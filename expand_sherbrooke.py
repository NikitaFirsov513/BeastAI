"""Expand training to 512 frames, preserving the 64-frame pilot and frozen validation."""
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib, io, json, random
from pathlib import Path
from PIL import Image
import waid
from fetch_sherbrooke import fetch_image

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data' / 'sherbrooke'
OUT = DATA / 'train512'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_selection(pilot):
    coco = json.loads((DATA / 'source_metadata/Ann_S_wds_Train.json').read_text(encoding='utf-8'))
    assert {c['id'] for c in coco['categories']} == {1}
    annotations = defaultdict(list)
    for ann in coco['annotations']:
        annotations[ann['image_id']].append(ann)
    metadata = json.loads((ROOT / 'data/dataset_candidates/sherbrooke.json').read_text(encoding='utf-8'))
    files = {f['dataFile']['filename']: f['dataFile'] for f in metadata['data']['latestVersion']['files']}
    old = json.loads((DATA / 'selection.json').read_text(encoding='utf-8'))
    chosen = [item for item in old if item['split'] == 'train']
    assert len(chosen) == 64
    chosen_names = {item['source_image']['file_name'] for item in chosen}
    valid_names = {item['source_image']['file_name'] for item in old if item['split'] == 'valid'}
    rng = random.Random(51242)
    for positive in (True, False):
        existing = sum(bool(item['annotations']) == positive for item in chosen)
        pool = [img for img in coco['images'] if bool(annotations[img['id']]) == positive and img['file_name'] not in chosen_names]
        for image in rng.sample(pool, 256 - existing):
            assert image['file_name'] not in valid_names
            entry = files[image['file_name']]
            chosen.append({'split': 'train', 'source_image': image, 'source_file_id': entry['id'], 'source_checksum': entry.get('checksum'), 'annotations': annotations[image['id']]})
            chosen_names.add(image['file_name'])
    chosen.sort(key=lambda item: item['source_image']['file_name'])
    assert len(chosen) == len(chosen_names) == 512
    assert sum(bool(item['annotations']) for item in chosen) == 256
    OUT.mkdir(parents=True, exist_ok=True)
    frozen = OUT / 'selection.json'
    if frozen.exists():
        assert json.loads(frozen.read_text(encoding='utf-8')) == chosen, 'Frozen expanded selection differs'
    else:
        waid.save_json(frozen, chosen)
    return chosen


def process(item):
    source = item['source_image']
    name = Path(source['file_name']).with_suffix('.png').name
    path = DATA / 'images/train' / name
    url = f"https://borealisdata.ca/api/access/datafile/{item['source_file_id']}?imageThumb=1024"
    if path.exists():
        content = path.read_bytes()
    else:
        content = fetch_image(url)
        with Image.open(io.BytesIO(content)) as image:
            image.load()
            if image.width != 1024 or abs(image.width/image.height-source['width']/source['height']) > .002:
                raise ValueError('Unexpected image geometry: ' + name)
        temporary = path.with_suffix('.png.part')
        temporary.write_bytes(content)
        temporary.replace(path)
    with Image.open(io.BytesIO(content)) as image:
        image.load()
        width, height = image.size
    assert width == 1024 and abs(width/height-source['width']/source['height']) < .002
    labels = []
    for ann in item['annotations']:
        assert ann['category_id'] == 1 and not ann.get('ignore', 0) and not ann.get('iscrowd', 0)
        x, y, w, h = ann['bbox']; sw, sh = source['width'], source['height']
        x1, y1, x2, y2 = max(0, x), max(0, y), min(sw, x+w), min(sh, y+h)
        assert x2 > x1 and y2 > y1
        labels.append(f'0 {(x1+x2)/2/sw:.8f} {(y1+y2)/2/sh:.8f} {(x2-x1)/sw:.8f} {(y2-y1)/sh:.8f}')
    payload = ('\n'.join(labels) + ('\n' if labels else '')).encode('utf-8')
    labelpath = DATA / 'labels/train' / (Path(name).stem + '.txt')
    if labelpath.exists():
        assert labelpath.read_bytes() == payload, 'Existing label differs: ' + name
    else:
        labelpath.write_bytes(payload)
    def entry(kind, filename, content):
        return {'path': f'SHERBROOKE/{kind}/train/{filename}', 'size': len(content), 'sha': waid.blob_hash(content), 'sha256': hashlib.sha256(content).hexdigest()}
    return {'split': 'train', 'name': name, 'species': 'red_deer', 'image': entry('images', name, content), 'label': entry('labels', labelpath.name, payload), 'source_file_id': item['source_file_id'], 'source_name': source['file_name'], 'source_checksum': item['source_checksum'], 'source_url': url, 'source_original_size': [source['width'], source['height']], 'original_size': [width, height], 'reference_count': len(labels)}


def main():
    waid.configure_dataset('sherbrooke')
    pilot = waid.get_index()
    pilot_index_sha = digest(DATA / 'index.json')
    pilot_pairs = pilot['pairs']
    for pair in pilot_pairs:
        for kind in ('image', 'label'):
            assert digest(waid.local_file(pair[kind])) == pair[kind]['sha256']
    assert Counter(p['split'] for p in pilot_pairs) == {'train': 64, 'valid': 32}
    selection = prepare_selection(pilot)
    old_train_names = {p['name'] for p in pilot_pairs if p['split'] == 'train'}
    print(json.dumps({'training_images': 512, 'positive': 256, 'negative': 256, 'validation_unchanged': 32, 'new_downloads': 448}), flush=True)
    train = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(process, item) for item in selection]
        for future in as_completed(futures):
            pair = future.result(); train.append(pair)
            if len(train) % 16 == 0 or len(train) == 512:
                print(f"[{len(train)}/512] verified; {sum(p['image']['size'] for p in train)/2**20:.1f} MiB", flush=True)
    train.sort(key=lambda item: item['name'])
    valid = [p for p in pilot_pairs if p['split'] == 'valid']
    assert {p['image']['sha256'] for p in train}.isdisjoint(p['image']['sha256'] for p in valid), 'Exact duplicate crosses split'
    assert old_train_names.issubset(p['name'] for p in train)
    for old_pair in pilot_pairs:
        for kind in ('image', 'label'):
            assert digest(waid.local_file(old_pair[kind])) == old_pair[kind]['sha256']
    assert digest(DATA / 'index.json') == pilot_index_sha
    pairs = train + valid
    revision = hashlib.sha256(json.dumps([(p['split'], p['name'], p['image']['sha256'], p['label']['sha256']) for p in pairs]).encode()).hexdigest()
    expanded = dict(pilot)
    expanded.update(pairs=pairs, revision=revision, parent_pilot_revision=pilot['revision'], selection=[
        '512 training frames: 256 positive and 256 negative, including all original 64 pilot training frames.',
        'Original author train/validation split retained. The original 32 validation frames and all their labels are unchanged.',
        'Training-only sampling seed 51242; no selection using model outputs.',
        'Sampling balances animal-containing and background training frames. Validation mixture is unchanged.',
        'Official full-frame 1024-pixel image derivatives. Missing higher-resolution details cannot be recovered by resizing.',
        'Same red-deer farm in Quebec; more frames do not create new geographic or species diversity.',
        '32 validation images are a reused development set, not an independent final test.'
    ], download_image_bytes=sum(p['image']['size'] for p in pairs))
    waid.save_json(OUT / 'index.json', expanded)
    def manifest(path, entries):
        path.write_text(''.join(waid.local_file(p['image']).as_posix()+'\n' for p in entries), encoding='utf-8')
    manifest(OUT / 'train.txt', train)
    manifest(OUT / 'valid.txt', valid)
    # Freeze the old pilot to explicit files before adding any future experiment.
    manifest(DATA / 'pilot_train.txt', [p for p in pilot_pairs if p['split'] == 'train'])
    manifest(DATA / 'pilot_valid.txt', valid)
    def yaml(path, trainpath, valpath):
        path.write_text(f'path: {DATA.as_posix()}\ntrain: {trainpath.as_posix()}\nval: {valpath.as_posix()}\nnames:\n  0: animal\n', encoding='utf-8')
    yaml(DATA / 'dataset.yaml', DATA / 'pilot_train.txt', DATA / 'pilot_valid.txt')
    yaml(OUT / 'dataset.yaml', OUT / 'train.txt', OUT / 'valid.txt')
    summary = {'training_images': len(train), 'positive_training_images': sum(p['reference_count'] > 0 for p in train), 'negative_training_images': sum(p['reference_count'] == 0 for p in train), 'training_boxes': sum(p['reference_count'] for p in train), 'validation_images': len(valid), 'validation_boxes': sum(p['reference_count'] for p in valid), 'validation_unchanged': True, 'pilot_index_unchanged': True, 'new_images': 448, 'new_image_bytes': sum(p['image']['size'] for p in train if p['name'] not in old_train_names), 'expanded_revision': revision, 'parent_pilot_revision': pilot['revision'], 'dataset_yaml': str(OUT / 'dataset.yaml'), 'license': pilot['license']}
    waid.save_json(OUT / 'summary.json', summary)
    (OUT / 'README.md').write_text('# Расширенная учебная подборка\n\n512 учебных кадров: 256 с оленями и 256 без животных. Все первые 64 учебных кадра сохранены. Проверочные 32 кадра и 115 рамок полностью совпадают с первым опытом.\n\nИзображения и разметка хранятся отдельно в ../images и ../labels; train.txt и valid.txt явно задают состав. dataset.yaml первого опыта теперь использует отдельные списки из 64/32 файлов: добавление изображений не меняет его состав.\n\nИсточник — прежняя RGB-съёмка оленьей фермы в Канаде. Лицензия CC BY-NC 4.0, исследовательский учебный опыт. Новых видов и мест съёмки здесь нет.\n\nКонтрольные суммы и исходные ID: index.json. Выбор до обучения: selection.json. Итоги проверки: summary.json. Подборка готова для обучения, новые веса ещё не обучались.\n', encoding='utf-8')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
