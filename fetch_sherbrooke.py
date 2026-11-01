"""Prepare a small frozen Sherbrooke RGB detection pilot from official image derivatives."""
from collections import defaultdict,Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import random
import time
import urllib.error
import urllib.request
from PIL import Image
import waid

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data'/'sherbrooke'
COUNTS={'train':(48,16),'valid':(8,24)}


def fetch_image(url):
    for attempt in range(3):
        try:
            request=urllib.request.Request(url,headers={'User-Agent':'BeastAI-educational-pilot'})
            with urllib.request.urlopen(request,timeout=60) as response:
                content=response.read(8*1024**2+1)
            if len(content)>8*1024**2:raise ValueError('Official derivative exceeded size budget')
            return content
        except (OSError,TimeoutError,urllib.error.URLError):
            if attempt==2:raise
            time.sleep(attempt+1)


def main():
    metadata=json.loads((ROOT/'data/dataset_candidates/sherbrooke.json').read_text(encoding='utf-8'))
    version=metadata['data']['latestVersion']
    files={f['dataFile']['filename']:f['dataFile'] for f in version['files']}
    selection=[];cocos={}
    for split,suffix in [('train','Train'),('valid','Val')]:
        coco=json.loads((DATA/'source_metadata'/f'Ann_S_wds_{suffix}.json').read_text(encoding='utf-8'))
        assert {c['id'] for c in coco['categories']}=={1}
        by_image=defaultdict(list)
        for ann in coco['annotations']:
            by_image[ann['image_id']].append(ann)
        positive=[image for image in coco['images'] if by_image[image['id']]]
        negative=[image for image in coco['images'] if not by_image[image['id']]]
        rng=random.Random(42 if split=='train' else 143)
        chosen=rng.sample(positive,COUNTS[split][0])+rng.sample(negative,COUNTS[split][1])
        for image in sorted(chosen,key=lambda item:item['file_name']):
            entry=files[image['file_name']]
            selection.append({'split':split,'source_image':image,'source_file_id':entry['id'],'source_checksum':entry.get('checksum'),'annotations':by_image[image['id']]})
        cocos[split]={'source_images':len(coco['images']),'source_positive_images':len(positive),'source_negative_images':len(negative),'source_boxes':len(coco['annotations'])}
    DATA.mkdir(parents=True,exist_ok=True)
    # Freeze selection before model runs; reruns must not silently change validation membership.
    frozen=DATA/'selection.json'
    if frozen.exists():
        assert json.loads(frozen.read_text(encoding='utf-8'))==selection,'Frozen pilot selection differs'
    else:
        waid.save_json(frozen,selection)

    def process(item):
        split=item['split'];source=item['source_image'];name=Path(source['file_name']).with_suffix('.png').name
        path=DATA/'images'/split/name;path.parent.mkdir(parents=True,exist_ok=True)
        url=f"https://borealisdata.ca/api/access/datafile/{item['source_file_id']}?imageThumb=1024"
        if path.exists():
            content=path.read_bytes()
        else:
            content=fetch_image(url)
            temporary=path.with_suffix('.png.part');temporary.write_bytes(content);temporary.replace(path)
        with Image.open(io.BytesIO(content)) as image:
            image.load();width,height=image.size
        if width!=1024 or abs(width/height-source['width']/source['height'])>.002:
            raise ValueError('Unexpected derivative geometry: '+name)
        labels=[]
        for annotation in item['annotations']:
            if annotation.get('ignore',0) or annotation.get('iscrowd',0):
                raise ValueError('Pilot contains ignored/crowd annotation; needs explicit evaluation policy')
            x,y,w,h=annotation['bbox'];sw,sh=source['width'],source['height']
            x1,y1,x2,y2=max(0,x),max(0,y),min(sw,x+w),min(sh,y+h)
            if x2<=x1 or y2<=y1:raise ValueError('Invalid COCO box')
            labels.append(f'0 {(x1+x2)/2/sw:.8f} {(y1+y2)/2/sh:.8f} {(x2-x1)/sw:.8f} {(y2-y1)/sh:.8f}')
        text=('\n'.join(labels)+ ('\n' if labels else '')).encode('utf-8')
        labelpath=DATA/'labels'/split/(Path(name).stem+'.txt');labelpath.parent.mkdir(parents=True,exist_ok=True);labelpath.write_bytes(text)
        def entry(kind,filename,payload):return {'path':f'SHERBROOKE/{kind}/{split}/{filename}','size':len(payload),'sha':waid.blob_hash(payload),'sha256':hashlib.sha256(payload).hexdigest()}
        return {'split':split,'name':name,'species':'red_deer','image':entry('images',name,content),'label':entry('labels',labelpath.name,text),'source_file_id':item['source_file_id'],'source_name':source['file_name'],'source_checksum':item['source_checksum'],'source_url':url,'source_original_size':[source['width'],source['height']],'original_size':[width,height],'reference_count':len(labels)}
    pairs=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        for i,pair in enumerate(pool.map(process,selection),1):
            pairs.append(pair)
            if i<=2 or i%8==0 or i==len(selection):print(f"[{i}/{len(selection)}] {pair['split']} {pair['name']} deer={pair['reference_count']}",flush=True)
    hashes={split:{p['image']['sha256'] for p in pairs if p['split']==split} for split in ('train','valid')}
    assert hashes['train'].isdisjoint(hashes['valid']),'Exact image duplicate crosses split'
    revision=hashlib.sha256(json.dumps([(p['split'],p['name'],p['image']['sha256'],p['label']['sha256']) for p in pairs]).encode()).hexdigest()
    index={'dataset':'sherbrooke','prefix':'SHERBROOKE','title':'Sherbrooke RGB — олени: учебное дообучение','revision':revision,'classes':['animal'],'category_names':{'red_deer':'Благородный олень'},'pairs':pairs,'source_urls':['https://doi.org/10.5683/SP3/TELIYD','https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0284449'],'source_version':f"{version['versionNumber']}.{version['versionMinorNumber']}",'license':version.get('license'),'annotation_coverage':'Detection dataset: manual per-image COCO boxes and background-only images, as supplied by the authors. Completeness is assumed according to detection protocol, not independently certified.','split_provenance':'Authors original summer train/validation split. Paper states images were grouped by flight before splitting. Flight IDs are not exposed in these COCO filenames, so individual-flight membership is not independently re-audited.','selection':['Frozen deterministic subset selected before any model outputs.','Train: 48 positive + 16 negative images. Validation: 8 positive + 24 negative images.','This deliberately enriched pilot does not preserve original animal occurrence rates. Precision depends on this mixture.','Official imageThumb=1024 derivatives are downloaded instead of 7952x5304 originals. Coordinates are normalized from source dimensions, preserving full-frame geometry.','Input experiments are limited to 640/1024. Original details above derivative resolution would require fetching larger images.','Images were collected on a red-deer farm in Quebec, Canada. Species occurs in Eastern Europe; habitat, management and geography are not representative of Eastern European wildlife.','Only a small pilot: not a deployment or final accuracy claim.','No random reassignment across author train/validation splits; no exact image-content duplicate crosses them.'],'source_counts':cocos,'download_image_bytes':sum(p['image']['size'] for p in pairs)}
    waid.save_json(DATA/'index.json',index)
    (DATA/'dataset.yaml').write_text(f'path: {DATA.as_posix()}\ntrain: images/train\nval: images/valid\nnames:\n  0: animal\n',encoding='utf-8')
    print(json.dumps({'images':len(pairs),'boxes':{s:sum(p['reference_count'] for p in pairs if p['split']==s) for s in COUNTS},'image_MiB':index['download_image_bytes']/1024**2,'index_revision':revision}),flush=True)


if __name__=='__main__':main()