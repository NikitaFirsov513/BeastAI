"""Oracle crop diagnostic on training references only; NOT full-frame detection quality."""
from pathlib import Path
import hashlib,json,os
import baseline,waid
from tiled_inference import map_box,merge_boxes
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'data/roe_expanded';EVAL=OUT/'evaluation'
for key,folder in [('YOLO_CONFIG_DIR','ultralytics-config'),('MPLCONFIGDIR','matplotlib-cache'),('TORCH_HOME','torch-cache')]:os.environ[key]=str(ROOT/'data'/folder)
os.environ['YOLO_AUTOINSTALL']='false'

def main():
    import cv2,numpy as np,torch
    from ultralytics import YOLO,settings
    settings.update({'sync':False});torch.set_num_threads(4);cv2.setNumThreads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    index=json.loads((OUT/'index.json').read_text(encoding='utf-8'));run=json.loads((OUT/'latest_run.json').read_text(encoding='utf-8'));fit=json.loads((EVAL/'training_fit_diagnostic.json').read_text(encoding='utf-8'))['models']['after']
    weights=Path(run['weights']);assert hashlib.sha256(weights.read_bytes()).hexdigest()==run['weights_sha256'];model=YOLO(str(weights))
    samples=[]
    for e in index['roe_training']:
        truth=waid.parse_labels(Path(e['label_path']),['animal'])['boxes'];image=None
        for number,gt in enumerate(truth):
            w,h=e.get('dimensions',[1280,1280]);scale=min(1024/w,1024/h);area=gt['width']*w*gt['height']*h*scale**2
            if area>=16**2:continue
            if image is None:image=cv2.imdecode(np.fromfile(e['image_path'],dtype=np.uint8),cv2.IMREAD_COLOR);assert image is not None and image.shape[:2]==(h,w)
            full=fit['items'][e['name']];full_iou=max((baseline.overlap(gt,b) for b in full['predictions']),default=0.);entry={'image':e['name'],'reference_index':number,'reference':gt,'area_at1024':area,'full_frame_best_emitted_iou':full_iou,'crops':{}}
            for crop in (256,512,768):
                tw,th=min(crop,w),min(crop,h);cx=(gt['x']+gt['width']/2)*w;cy=(gt['y']+gt['height']/2)*h;x=max(0,min(w-tw,int(round(cx-tw/2))));y=max(0,min(h-th,int(round(cy-th/2))))
                contained=gt['x']*w>=x-1e-4 and gt['y']*h>=y-1e-4 and (gt['x']+gt['width'])*w<=x+tw+1e-4 and (gt['y']+gt['height'])*h<=y+th+1e-4
                if not contained:entry['crops'][str(crop)]={'target_contained':False};continue
                result=model.predict(source=image[y:y+th,x:x+tw],imgsz=1024,rect=False,device=0,half=False,batch=1,conf=.25,iou=.7,agnostic_nms=True,classes=[0],max_det=300,verbose=False,save=False)[0]
                boxes=[map_box(row,(x,y,x+tw,y+th),(w,h),model.names) for row in result.boxes.data.detach().cpu().tolist()];boxes=merge_boxes([b for b in boxes if b],.5,300)
                best=max((baseline.overlap(gt,b) for b in boxes),default=0.)
                entry['crops'][str(crop)]={'target_contained':True,'window':[x,y,x+tw,y+th],'best_emitted_iou':best,'target_matched':best>=.5,'predictions':boxes}
            samples.append(entry)
    assert len(samples)==30
    summary={'references':len(samples),'full_frame_targets_with_emitted_iou_at_least0.5':sum(r['full_frame_best_emitted_iou']>=.5 for r in samples),'by_oracle_crop':{str(crop):{'contained_targets':sum(r['crops'][str(crop)]['target_contained'] for r in samples),'matched_targets':sum(r['crops'][str(crop)].get('target_matched',False) for r in samples)} for crop in (256,512,768)}}
    output={'dataset_revision':index['revision'],'model_sha256':run['weights_sha256'],'scope':'Known-location crop diagnostic on30 tiny TRAINING references only. Crop centers use ground truth; this is NOT a detector that searches an unknown frame and NOT validation recall.','conditions':{'network_input':[1,3,1024,1024],'source_crop_sizes':[256,512,768],'confidence_threshold':.25,'match_iou':.5,'precision':'FP32, TF32 off','small_reference_rule':'Fixed full-frame1024 area below16 squared'},'limitations':['Known target location makes this an oracle experiment, not an operational pipeline.','Training samples cannot estimate generalization.','Upsampling does not restore missing source detail.','No false-box rate or end-to-end latency is claimed for deployment.'],'summary':summary,'items':samples}
    waid.save_json(EVAL/'oracle_crop_training_diagnostic.json',output);print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
