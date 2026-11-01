"""Run a reproducible, single-frame pretrained YOLO11n baseline on the local preview."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

import waid

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / 'data' / 'baseline'
ANIMAL_NAMES = {'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe'}


def animal_class_ids(names):
    return [key for key,value in names.items() if value in ANIMAL_NAMES or value == 'animal']


def overlap(a, b):
    def corners(box):
        return max(0, box['x']), max(0, box['y']), min(1, box['x'] + box['width']), min(1, box['y'] + box['height'])
    ax, ay, ar, ab = corners(a)
    bx, by, br, bb = corners(b)
    intersection = max(0, min(ar, br) - max(ax, bx)) * max(0, min(ab, bb) - max(ay, by))
    union = max(0, ar-ax)*max(0, ab-ay) + max(0, br-bx)*max(0, bb-by) - intersection
    return intersection / union if union > 0 else 0


def match_boxes(predictions, truth, threshold=0.5):
    unmatched = set(range(len(truth)))
    matched = []
    for prediction in sorted(predictions, key=lambda p: p['score'], reverse=True):
        best = max(unmatched, key=lambda i: overlap(prediction, truth[i]), default=None)
        if best is not None and overlap(prediction, truth[best]) >= threshold:
            unmatched.remove(best)
            matched.append(best)
    return {'tp': len(matched), 'fp': len(predictions)-len(matched), 'fn': len(unmatched), 'matched_truth': matched}


def percentile(values, fraction):
    values = sorted(values)
    position = (len(values)-1)*fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return values[lower] + (values[upper]-values[lower])*(position-lower)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=('waid','bambi','sherbrooke'), default='waid')
    parser.add_argument('--validation-only', action='store_true', help='Process only the frozen validation images')
    parser.add_argument('--weights', help='Local weights for comparing a fine-tuned model')
    parser.add_argument('--model-label', help='Readable model label in the report')
    parser.add_argument('--no-latest', action='store_true', help='Save diagnostic report without replacing the viewer baseline')
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    waid.configure_dataset(args.dataset)
    output = OUTPUT if args.dataset == "waid" else waid.DATA / "baseline"
    if args.imgsz < 32 or args.imgsz % 32 or not 0 < args.conf < 1 or args.repeats < 1:
        parser.error('Use positive repeats, 0 < conf < 1, and imgsz divisible by 32.')
    for key, folder in [('YOLO_CONFIG_DIR','ultralytics-config'), ('MPLCONFIGDIR','matplotlib-cache'), ('TORCH_HOME','torch-cache')]:
        path = ROOT / 'data' / folder
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    os.environ['YOLO_AUTOINSTALL'] = 'false'
    import cv2
    import numpy as np
    import psutil
    import torch
    import torchvision
    import ultralytics
    from ultralytics import YOLO, settings
    settings.update({'sync': False})
    if not torch.cuda.is_available():
        raise RuntimeError('NVIDIA GPU is not available in PyTorch; baseline was not silently switched to CPU.')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    index = waid.get_index()
    splits = ('valid',) if args.validation_only else ('train','valid')
    pairs = [p for p in waid.downloaded_pairs(index) if p['split'] in splits]
    if not pairs or not any(p['split']=='valid' for p in pairs):
        raise RuntimeError('Download the preview first, including valid images.')
    weights = Path(args.weights).resolve() if args.weights else ROOT / 'data' / 'models' / 'yolo11n.pt'
    weights.parent.mkdir(parents=True, exist_ok=True)
    if not weights.exists():
        if args.weights:
            raise FileNotFoundError('Requested local weights do not exist: '+str(weights))
        content = waid.fetch('https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt')
        temporary = weights.with_suffix('.pt.part')
        temporary.write_bytes(content)
        temporary.replace(weights)
    load_start = time.perf_counter()
    model = YOLO(str(weights))
    animal_ids = animal_class_ids(model.names)
    if not animal_ids:
        raise ValueError('Model has no configured animal output category')
    config = dict(imgsz=args.imgsz, rect=False, device=0, half=False, batch=1, conf=args.conf, iou=0.7, agnostic_nms=True, classes=animal_ids, max_det=300, verbose=False, save=False)

    def predict(path):
        torch.cuda.synchronize()
        started = time.perf_counter()
        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError('Could not decode image: ' + str(path))
        read_ms = (time.perf_counter()-started)*1000
        result = model.predict(source=image, **config)[0]
        height, width = image.shape[:2]
        boxes = []
        # One transfer includes coordinates, confidence, and category; included in total time.
        for x1,y1,x2,y2,score,category in result.boxes.data.detach().cpu().tolist():
            boxes.append({'x':x1/width,'y':y1/height,'width':(x2-x1)/width,'height':(y2-y1)/height,'score':score,'class_id':int(category),'class_name':model.names[int(category)]})
        torch.cuda.synchronize()
        elapsed = (time.perf_counter()-started)*1000
        return boxes, {'total_ms':elapsed,'read_decode_ms':read_ms, **{key+'_ms':float(value) for key,value in result.speed.items()}}, [width,height]

    first_path = waid.local_file(pairs[0]['image'])
    _, cold, _ = predict(first_path)
    load_and_first_ms = (time.perf_counter()-load_start)*1000
    for _ in range(5):
        predict(first_path)
    torch.cuda.reset_peak_memory_stats()
    items, timings = {}, []
    for number,pair in enumerate(pairs,1):
        image_path = waid.local_file(pair['image'])
        truth = waid.parse_labels(waid.local_file(pair['label']), index['classes'])
        if truth['status'] not in ('ok','empty'):
            raise ValueError('Invalid reference labels: ' + pair['name'])
        repeated = []
        for _ in range(args.repeats):
            boxes, timing, shape = predict(image_path)
            repeated.append(timing)
            timings.append(timing)
        matched = match_boxes(boxes,truth['boxes'])
        scale = min(args.imgsz/shape[0],args.imgsz/shape[1])
        small_ids = {i for i,b in enumerate(truth['boxes']) if b['width']*shape[0]*scale*b['height']*shape[1]*scale < 32**2}
        items[pair['split']+'/'+pair['name']] = {'boxes':boxes,'timings':repeated,'flight_id':pair.get('flight_id'),'original_size':shape,'image_git_sha':pair['image']['sha'],'label_git_sha':pair['label']['sha'],'reference_boxes':len(truth['boxes']),'counts':{key:matched[key] for key in ('tp','fp','fn')},'small_reference':len(small_ids),'small_matched':len(small_ids.intersection(matched['matched_truth']))}
        print(f'[{number}/{len(pairs)}] {pair["split"]}: reference={len(truth["boxes"])} predicted={len(boxes)} total={statistics.median(t["total_ms"] for t in repeated):.1f} ms',flush=True)
    valid = [value for key,value in items.items() if key.startswith('valid/')]
    counts = {key:sum(v['counts'][key] for v in valid) for key in ('tp','fp','fn')}
    counts.update({'images':len(valid),'precision':counts['tp']/(counts['tp']+counts['fp']) if counts['tp']+counts['fp'] else None,'recall':counts['tp']/(counts['tp']+counts['fn']) if counts['tp']+counts['fn'] else None})
    counts['small_reference'] = sum(v['small_reference'] for v in valid)
    counts['small_matched'] = sum(v['small_matched'] for v in valid)
    counts['small_recall'] = counts['small_matched']/counts['small_reference'] if counts['small_reference'] else None
    if args.dataset == 'bambi':
        counts['unmatched_predictions'] = counts['fp']
        counts['precision'] = None
        counts['reference_box_recall_at_iou_0_5'] = counts['recall']
        counts['quality_scope'] = 'Partial reviewed tracks, not exhaustive frame annotation. Unmatched predictions are not necessarily false positives.'
    timing_summary = {key:{'median':statistics.median(t[key] for t in timings),'p95':percentile([t[key] for t in timings],.95)} for key in timings[0]}
    try:
        driver = subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,memory.total,power.draw','--format=csv,noheader'],capture_output=True,text=True,timeout=10).stdout.strip()
    except (OSError,subprocess.TimeoutExpired):
        driver = 'unavailable'
    report = {'created_utc':datetime.now(timezone.utc).isoformat(),'dataset':args.dataset,'selection_notes':index.get('selection',[]),'source_urls':index.get('source_urls',[]),'annotation_coverage':index.get('annotation_coverage','see dataset notes'),'dataset_revision':index['revision'],'model':args.model_label or ('YOLO11n fine-tuned' if args.weights else 'YOLO11n COCO pretrained'),'model_path':str(weights.relative_to(ROOT)),'model_sha256':hashlib.sha256(weights.read_bytes()).hexdigest(),'model_file_bytes':weights.stat().st_size,'parameters':sum(p.numel() for p in model.model.parameters()),'conditions':{'gpu':torch.cuda.get_device_name(0),'driver_snapshot':driver,'cpu':platform.processor(),'os':platform.platform(),'ram_total_bytes':psutil.virtual_memory().total,'ram_available_bytes':psutil.virtual_memory().available,'python':platform.python_version(),'torch':torch.__version__,'torchvision':torchvision.__version__,'ultralytics':ultralytics.__version__,'cuda':torch.version.cuda,'input_shape':[1,3,args.imgsz,args.imgsz],'precision':'FP32; TF32 disabled','model_dtype':str(next(model.model.parameters()).dtype),'batch_size':1,'cpu_threads':torch.get_num_threads(),'confidence_threshold':args.conf,'nms_iou':.7,'class_agnostic_nms':True,'match_iou':.5,'repeats':args.repeats,'warmup_frames':5,'power_energy':'not measured; GPU power snapshot is not per-frame energy'},'memory':{'gpu_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'gpu_peak_reserved_bytes':torch.cuda.max_memory_reserved(),'process_rss_bytes':psutil.Process().memory_info().rss},'cold_first_frame':cold,'load_and_first_frame_ms':load_and_first_ms,'timing_summary':timing_summary,'validation_preview':counts,'notes':['Small downloaded preview only, not a reliable final benchmark.','Only valid contributes to reported quality. test is not processed.','Ground-truth categories are collapsed to animal for matching.','Only COCO animal predictions are accepted. COCO does not include all WAID categories.','Small means box area below 32*32 pixels after aspect-preserving input resize.','Total latency includes file read/decode, preprocessing, GPU inference, NMS, animal boxes copied to CPU and converted to normalized coordinates.','Model load, warmup, reference parsing, quality scoring, JSON writing and browser rendering are outside warm total latency.','Repeated reads may use the operating system file cache. Flight independence is not yet audited.','GPU peak allocation excludes memory used by the driver and other processes.'],'items':items}
    output.mkdir(parents=True,exist_ok=True)
    name = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-yolo11n-'+str(args.imgsz)+'-conf'+str(args.conf)+'.json'
    waid.save_json(output/name,report)
    if not args.no_latest:
        waid.save_json(output/'latest.json',report)
    print(json.dumps({'report':str(output/name),'validation_preview':counts,'total_latency_ms':timing_summary['total_ms']},indent=2),flush=True)


if __name__=='__main__':
    main()