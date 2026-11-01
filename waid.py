"""Download WAID image/label pairs and inspect them in a local browser.

Python 3.10+, standard library only. Nothing is installed into Python.
"""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path, PurePosixPath
import random
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "waid"
INDEX = DATA / "index.json"
REPO = "https://api.github.com/repos/xiaohuicui/WAID"
RAW = "https://raw.githubusercontent.com/xiaohuicui/WAID"
SPLITS = ("train", "valid", "test")
DATASET = "waid"
PREFIX = "WAID"


def configure_dataset(name):
    global DATA, INDEX, DATASET, PREFIX
    if name not in ("waid", "bambi", "sherbrooke"):
        raise ValueError("Unknown dataset")
    DATASET, PREFIX = name, name.upper()
    DATA = ROOT / "data" / name
    INDEX = DATA / "index.json"



def fetch(url):
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "BeastAI-WAID-viewer"})
            with urllib.request.urlopen(request, timeout=35) as response:
                return response.read()
        except (urllib.error.URLError, TimeoutError, OSError):
            if attempt == 2:
                raise
            time.sleep(0.5 * (attempt + 1))


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def get_index():
    if INDEX.exists():
        return json.loads(INDEX.read_text(encoding="utf-8"))
    if DATASET != "waid":
        raise ValueError(f"No local {DATASET} dataset. Run its fetch script first.")
    print("Reading GitHub file list (no image archive)...", flush=True)
    commit = json.loads(fetch(REPO + "/commits/main"))
    revision = commit["sha"]
    tree = json.loads(fetch(REPO + "/git/trees/" + commit["commit"]["tree"]["sha"] + "?recursive=1"))
    if tree.get("truncated"):
        raise ValueError("GitHub returned an incomplete file list. No downloads started.")
    files = {}
    for entry in tree["tree"]:
        if entry["type"] != "blob":
            continue
        parts = PurePosixPath(entry["path"]).parts
        if len(parts) == 4 and parts[0] == "WAID" and parts[1] in ("images", "labels") and parts[2] in SPLITS:
            files[entry["path"]] = {"path": entry["path"], "sha": entry["sha"], "size": entry["size"]}
    classes = fetch(RAW + "/" + revision + "/WAID/classes.txt").decode("utf-8-sig").splitlines()
    pairs = []
    for path, image in sorted(files.items()):
        parts = PurePosixPath(path).parts
        if parts[1] != "images" or PurePosixPath(path).suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        label_path = "WAID/labels/" + parts[2] + "/" + PurePosixPath(parts[3]).with_suffix(".txt").name
        if label_path not in files:
            raise ValueError("Image has no matching label file: " + path)
        pairs.append({"split": parts[2], "name": parts[3], "image": image, "label": files[label_path]})
    if not pairs:
        raise ValueError("No image/label pairs found. Repository structure may have changed.")
    index = {"revision": revision, "classes": classes, "pairs": pairs}
    save_json(INDEX, index)
    return index


def local_file(entry):
    parts = PurePosixPath(entry["path"]).parts
    if len(parts) != 4 or parts[0] != PREFIX or parts[1] not in ("images", "labels") or parts[2] not in SPLITS or any(p in (".", "..") or "\\" in p for p in parts):
        raise ValueError("Unsafe dataset path")
    path = DATA.joinpath(*parts[1:]).resolve()
    if not path.is_relative_to(DATA.resolve()):
        raise ValueError("Dataset path escapes data directory")
    return path


def blob_hash(content):
    return hashlib.sha1(b"blob " + str(len(content)).encode("ascii") + b"\0" + content).hexdigest()


def download_file(entry, revision):
    path = local_file(entry)
    if path.exists() and path.stat().st_size == entry["size"] and blob_hash(path.read_bytes()) == entry["sha"]:
        return False
    content = fetch(RAW + "/" + revision + "/" + urllib.parse.quote(entry["path"], safe="/"))
    if len(content) != entry["size"] or blob_hash(content) != entry["sha"]:
        raise ValueError("Downloaded file failed integrity check: " + entry["path"])
    if entry["path"].startswith("WAID/images/") and content.startswith(b"version https://git-lfs.github.com/spec"):
        raise ValueError("Image is a Git LFS pointer, not actual image data")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(content)
    temporary.replace(path)
    return True


def sample_pairs(pairs, limit):
    # Fixed seed and round-robin splits make a small, reproducible preview.
    rng = random.Random(42)
    groups = {split: [p for p in pairs if p["split"] == split] for split in SPLITS}
    for group in groups.values():
        rng.shuffle(group)
    result = []
    while len(result) < min(limit, len(pairs)):
        for split in SPLITS:
            if groups[split] and len(result) < limit:
                result.append(groups[split].pop())
    return result


def summary(index):
    counts = Counter(pair["split"] for pair in index["pairs"])
    total = sum(p[k]["size"] for p in index["pairs"] for k in ("image", "label"))
    print("WAID revision:", index["revision"])
    print("Image/label pairs:", len(index["pairs"]), dict(counts))
    print(f"All original images + labels: {total / 1024**3:.2f} GiB (without Git history).")
    print("Folder:", DATA)


def download(index, limit, all_files):
    pairs = index["pairs"] if all_files else sample_pairs(index["pairs"], limit)
    size = sum(p[k]["size"] for p in pairs for k in ("image", "label"))
    print(f"Selected {len(pairs)} pairs, {size / 1024**2:.1f} MiB before skipping existing files.", flush=True)

    def get_pair(pair):
        changed = sum(download_file(pair[key], index["revision"]) for key in ("image", "label"))
        return pair["name"], changed

    changed = 0
    # Four connections limit pressure on disk, RAM, and the public host.
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i, (name, count) in enumerate(pool.map(get_pair, pairs), 1):
            changed += count
            if i <= 3 or i % 25 == 0 or i == len(pairs):
                print(f"[{i}/{len(pairs)}] {name}", flush=True)
    print(f"Done. New/repaired files: {changed}. Images and labels are stored separately.")
    print("Preview only: this sample is not a reliable quality benchmark or a flight-based split.")


def parse_labels(path, classes):
    if not path.exists():
        return {"status": "missing", "boxes": [], "errors": ["Нет файла разметки"]}
    boxes, errors = [], []
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeError) as error:
        return {"status": "invalid", "boxes": [], "errors": [str(error)]}
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            parts = line.split()
            if len(parts) != 5:
                raise ValueError("ожидалось пять чисел")
            class_id = int(parts[0])
            x, y, width, height = map(float, parts[1:])
            if not (0 <= class_id < len(classes)):
                raise ValueError("неизвестный номер категории")
            if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < width <= 1 and 0 < height <= 1):
                raise ValueError("координаты или размеры вне диапазона")
            # Preserve original coordinates; SVG clips boxes at image boundaries.
            boxes.append({"class_id": class_id, "class_name": classes[class_id], "x": x - width / 2, "y": y - height / 2, "width": width, "height": height})
        except ValueError as error:
            errors.append(f"Строка {line_number}: {error}")
    return {"status": "invalid" if errors else ("ok" if boxes else "empty"), "boxes": boxes, "errors": errors}


def downloaded_pairs(index):
    names = set()
    for split in SPLITS:
        folder = DATA / 'images' / split
        if folder.is_dir():
            names.update((split, path.name) for path in folder.iterdir() if path.is_file())
    return [pair for pair in index['pairs'] if (pair['split'], pair['name']) in names and local_file(pair['image']).is_file()]


def make_handler(index):
    available = downloaded_pairs(index)
    wild_species = {'red_deer','roe_deer','wild_boar','moose','fox','wolf','hare','european_bison'}
    categories = {(p['split'],p['name']): ({p['species']} if p.get('species') else {b['class_name'] for b in parse_labels(local_file(p['label']),index['classes'])['boxes']}) for p in available}
    category_counts = Counter(name for names in categories.values() for name in names)
    wild_count = sum(bool(names.intersection(wild_species)) for names in categories.values())
    allowed_images = {"/data/" + "/".join(PurePosixPath(p["image"]["path"]).parts[1:]): local_file(p["image"]) for p in available}

    class Handler(BaseHTTPRequestHandler):
        def send_content(self, content, content_type, status=200):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(content)

        def do_GET(self):
            url = urllib.parse.urlsplit(self.path)
            path = urllib.parse.unquote(url.path)
            try:
                if path == "/":
                    self.send_content((ROOT / "viewer.html").read_bytes(), "text/html; charset=utf-8")
                elif path == "/api/items":
                    query = urllib.parse.parse_qs(url.query)
                    split = query.get("split", [""])[0]
                    needle = query.get("q", [""])[0].lower()
                    category = query.get("category", [""])[0]
                    page = max(0, int(query.get("page", ["0"])[0]))
                    selected = [p for p in available if (not split or p["split"] == split) and needle in p["name"].lower() and (not category or (bool(categories[(p["split"],p["name"])].intersection(wild_species)) if category in ("wild","europe") else category in categories[(p["split"],p["name"])]))]
                    pages = max(1, (len(selected) + 11) // 12)
                    page = min(page, pages - 1)
                    baseline_path = (ROOT / 'data' / 'baseline' if DATASET == 'waid' else DATA / 'baseline') / 'latest.json'
                    baseline = json.loads(baseline_path.read_text(encoding='utf-8')) if baseline_path.exists() else None
                    if baseline and baseline.get('dataset_revision') != index['revision']:
                        raise ValueError('Prediction report belongs to a different dataset revision')
                    items = []
                    for pair in selected[page * 12:(page + 1) * 12]:
                        label = parse_labels(local_file(pair["label"]), index["classes"])
                        prediction = baseline['items'].get(pair['split']+'/'+pair['name']) if baseline else None
                        if prediction and (prediction['image_git_sha'] != pair['image']['sha'] or prediction['label_git_sha'] != pair['label']['sha']):
                            raise ValueError('Prediction checksum differs from dataset index')
                        items.append({"flight_id": pair.get("flight_id"), "prediction": prediction, "name": pair["name"], "split": pair["split"], "src": "/data/" + "/".join(PurePosixPath(pair["image"]["path"]).parts[1:]), **label})
                    baseline_summary = {key: baseline[key] for key in ('model','conditions','timing_summary','validation_preview')} if baseline else None
                    analysis_path = DATA / 'analysis.json'
                    analysis = json.loads(analysis_path.read_text(encoding='utf-8')) if analysis_path.exists() else None
                    if analysis and analysis.get('dataset_revision') != index['revision']:
                        raise ValueError('Automatic comparison belongs to another dataset revision')
                    if analysis:
                        analysis = {key:value for key,value in analysis.items() if key != 'frame_results'}
                    threshold_path = DATA / 'threshold-analysis.json'
                    threshold_analysis = json.loads(threshold_path.read_text(encoding='utf-8')) if threshold_path.exists() else None
                    if threshold_analysis and (threshold_analysis.get('dataset_revision') != index['revision'] or not baseline or threshold_analysis.get('model_sha256') != baseline.get('model_sha256')):
                        threshold_analysis = None
                    if threshold_analysis:
                        threshold_analysis = {key:value for key,value in threshold_analysis.items() if key != 'frame_results'}
                    result = {"threshold_analysis": threshold_analysis, "analysis": analysis, "dataset_title": index.get("title", "WAID"), "dataset": DATASET, "category_names": index.get("category_names", {}), "source_urls": index.get("source_urls", ["https://github.com/xiaohuicui/WAID"]), "category_counts": dict(category_counts), "wild_downloaded": wild_count, "baseline": baseline_summary, "total": len(selected), "downloaded": len(available), "repository_total": len(index["pairs"]), "page": page, "pages": pages, "revision": index["revision"], "items": items}
                    self.send_content(json.dumps(result, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")
                elif path in allowed_images:
                    file = allowed_images[path]
                    self.send_content(file.read_bytes(), "image/png" if file.suffix.lower() == ".png" else "image/jpeg")
                else:
                    self.send_content(b"Not found", "text/plain", 404)
            except (ValueError, OSError) as error:
                self.send_content(str(error).encode("utf-8"), "text/plain; charset=utf-8", 400)

        def log_message(self, format, *args):
            pass

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("inspect", help="Read file list and report complete download size")
    download_parser = sub.add_parser("download", help="Download a small preview, or all pairs with --all")
    download_parser.add_argument("--limit", type=int, default=30)
    download_parser.add_argument("--all", action="store_true", help="Download the complete dataset")
    serve_parser = sub.add_parser("serve", help="Open a local viewer for downloaded files")
    serve_parser.add_argument("--dataset", choices=("waid","bambi","sherbrooke"), default="waid")
    serve_parser.add_argument("--port", type=int, default=8765)
    serve_parser.add_argument("--open", action="store_true")
    args = parser.parse_args()
    if args.command == "download" and args.limit < 1:
        parser.error("--limit must be positive")
    try:
        if args.command == "serve":
            configure_dataset(args.dataset)
        if args.command == "serve" and not INDEX.exists():
            raise ValueError("No local dataset. Run: python waid.py download --limit 30")
        index = get_index()
        if args.command == "inspect":
            summary(index)
        elif args.command == "download":
            summary(index)
            download(index, args.limit, args.all)
        else:
            server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(index))
            address = f"http://127.0.0.1:{server.server_port}/"
            print("Viewer:", address, "\nKeep this window open. Ctrl+C stops the viewer.", flush=True)
            if args.open:
                webbrowser.open(address)
            try:
                server.serve_forever()
            finally:
                server.server_close()
    except KeyboardInterrupt:
        print("\nStopped. Completed downloads are preserved.")
    except (OSError, ValueError, KeyError) as error:
        print("Error:", error, file=sys.stderr)
        print("You can rerun the command; valid completed files will be skipped.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
