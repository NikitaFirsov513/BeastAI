"""External development check on existing BAMBI valid flights; partial annotation, no FP claim."""
from pathlib import Path
from collections import Counter
import hashlib,json,os
import baseline,waid
from tiled_inference import map_box,merge_boxes
ROOT=Path(__file__).resolve().parent;DATA=ROOT/'data/bambi';OUT=ROOT/'data/roe_expanded/evaluation'
for key,folder in [('YOLO_CONFIG_DIR','ultralytics-config'),('MPLCONFIGDIR','matplotlib-cache'),('TORCH_HOME','torch-cache')]:os.environ[key]=str(ROOT/'data'/folder)
os.environ['YOLO_AUTOINSTALL']='false'

def main():
    import cv2,numpy as np,torch
    from ultralytics import YOLO,settings
    settings.update({'sync':False});torch.set_num_threads(4);cv2.setNumThreads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    index=json.loads((DATA/'index.json').read_text(encoding='utf-8'));pairs=[p for p in index['pairs'] if p['split']=='valid'];assert len(pairs)==19
    run=json.loads((ROOT/'data/roe_expanded/latest_run.json').read_text(encoding='utf-8'));models=[('before',run['comparison_weights'],run['comparison_weights_sha256']),('after',run['weights'],run['weights_sha256'])];reports={}
    def local(entry):
        parts=Path(entry['path']).parts;assert parts[0]=='BAMBI';path=(DATA/Path(*parts[1:])).resolve();assert path.is_relative_to(DATA);assert hashlib.sha256(path.read_bytes()).hexdigest()==entry['sha256'];return path
    for stage,path,expected_sha in models:
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==expected_sha;model=YOLO(path);totals=Counter();items={}
        for p in pairs:
            image_path=local(p['image']);label_path=local(p['label']);truth=waid.parse_labels(label_path,['animal'])['boxes'];image=cv2.imdecode(np.fromfile(image_path,dtype=np.uint8),cv2.IMREAD_COLOR);h,w=image.shape[:2]
            r=model.predict(source=image,imgsz=1024,rect=False,device=0,half=False,batch=1,conf=.25,iou=.7,agnostic_nms=True,classes=[0],max_det=300,verbose=False,save=False)[0]
            boxes=[map_box(row,(0,0,w,h),(w,h),model.names) for row in r.boxes.data.detach().cpu().tolist()];boxes=merge_boxes([b for b in boxes if b],.5,300);match=baseline.match_boxes(boxes,truth,.5);small={i for i,b in enumerate(truth) if b['width']*w*b['height']*h*min(1024/w,1024/h)**2<32**2}
            counts={'reference_boxes':len(truth),'matched':match['tp'],'unmatched_references':match['fn'],'unmatched_predictions_not_false_positive_count':match['fp'],'small_reference':len(small),'small_matched':len(small&set(match['matched_truth']))};totals.update(counts)
            items[p['name']]={'flight_id':p['flight_id'],'reference':truth,'predictions':boxes,'counts':counts,'image_sha256':p['image']['sha256'],'label_sha256':p['label']['sha256']}
        reports[stage]={'weights':path,'weights_sha256':expected_sha,'summary':dict(totals),'items':items};print(stage,dict(totals),flush=True);del model
    output={'scope':'External source DEVELOPMENT diagnostic, only19 existing BAMBI valid frames on flights13,236,240. No BAMBI images used in this model training.','dataset_revision':index['revision'],'flights':sorted({p['flight_id'] for p in pairs}),'images':19,'annotation_policy':'Only reviewed tracks are annotated. Reference recall is measurable for this selected subset; unmatched predictions are not automatically false positives. Precision and false-alert rate are undefined here.','conditions':{'input_shape':[1,3,1024,1024],'batch_size':1,'precision':'FP32, TF32 disabled','confidence_threshold':.25,'match_iou':.5,'detector_nms_iou':.7,'additional_merge_iou':.5},'limitations':['Selected clear-positive frames only, not representative empty backgrounds or all visibility conditions.','Multiple frames can show the same animals; counts are annotated appearances, not independent video events.','Flight IDs exist, but geographic/site independence and full annotation coverage are not verified.','This is development evidence, not a new final independent test.','No performance claim on a future1TOPS device.'],'models':reports}
    waid.save_json(OUT/'bambi_external_development.json',output)

if __name__=='__main__':main()
