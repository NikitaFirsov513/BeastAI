"""One-pass training-fit diagnosis after the frozen held-out pilot; no timing claim."""
from project_paths import PROJECT_ROOT
from pathlib import Path
import hashlib,json,os
import waid,baseline
from evaluate_crop_pilot import windows
from tiled_inference import map_box,merge_boxes
ROOT=PROJECT_ROOT;OUT=ROOT/'data/crop_pilot'
for key,folder in [('YOLO_CONFIG_DIR','ultralytics-config'),('MPLCONFIGDIR','matplotlib-cache'),('TORCH_HOME','torch-cache')]:os.environ[key]=str(ROOT/'data'/folder)
os.environ['YOLO_AUTOINSTALL']='false'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    import cv2,numpy as np,torch
    from ultralytics import YOLO,settings
    target=OUT/'evaluation/training_fit.json'
    if target.exists():raise RuntimeError('Keep completed diagnosis')
    settings.update({'sync':False});torch.set_num_threads(4);cv2.setNumThreads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    index=json.loads((OUT/'index.json').read_text(encoding='utf-8'));run=json.loads((OUT/'latest_run.json').read_text(encoding='utf-8'));entries=index['roe_training_sources'];models={}
    cfg=dict(imgsz=1024,rect=False,device=0,half=False,batch=1,conf=.25,iou=.7,agnostic_nms=True,classes=[0],max_det=300,verbose=False,save=False)
    for stage,weight_key in [('before','base_weights'),('after','weights')]:
        assert sha(run[weight_key])==run[weight_key+'_sha256'];model=YOLO(run[weight_key]);items={}
        for number,e in enumerate(entries,1):
            assert sha(e['image_path'])==e['image_sha256'] and sha(e['label_path'])==e['label_sha256']
            image=cv2.imdecode(np.fromfile(e['image_path'],dtype=np.uint8),cv2.IMREAD_COLOR);h,w=image.shape[:2];boxes=[]
            for window in windows(w,h,'tiles'):
                x1,y1,x2,y2=window;r=model.predict(source=image[y1:y2,x1:x2],**cfg)[0];assert len(r.boxes)<300
                for row in r.boxes.data.detach().cpu().tolist():
                    b=map_box(row,window,(w,h),model.names)
                    if b:boxes.append(b)
            boxes=merge_boxes(boxes,.5,300);assert len(boxes)<300
            truth=waid.parse_labels(Path(e['label_path']),['animal'])['boxes'];match=baseline.match_boxes(boxes,truth,.5);found=set(match['matched_truth']);groups={g:{'reference':0,'found':0} for g in ('below16','16to32','32to64','atleast64')}
            for i,b in enumerate(truth):
                area=b['width']*b['height']*1024**2;g='below16' if area<16**2 else '16to32' if area<32**2 else '32to64' if area<64**2 else 'atleast64';groups[g]['reference']+=1;groups[g]['found']+=int(i in found)
            items[e['name']]={'image_sha256':e['image_sha256'],'label_sha256':e['label_sha256'],'reference':truth,'boxes':boxes,'counts':{k:match[k] for k in ('tp','fp','fn')},'size_groups':groups}
            if number%8==0:print(f'{stage} fit {number}/58',flush=True)
        counts={k:sum(r['counts'][k] for r in items.values()) for k in ('tp','fp','fn')};assert counts['tp']+counts['fn']==89
        groups={g:{key:sum(r['size_groups'][g][key] for r in items.values()) for key in ('reference','found')} for g in ('below16','16to32','32to64','atleast64')}
        assert groups['below16']['reference']==30
        models[stage]={'model_sha256':sha(run[weight_key]),'counts':counts,'size_groups':groups,'items':items};del model;torch.cuda.empty_cache()
    result={'dataset_revision':index['revision'],'images':58,'reference_boxes':89,'models':models,'conditions':{'input':1024,'source_tile_size':512,'network_calls_per_frame':9,'batch':1,'precision':'FP32; TF32 off','confidence':.25,'local_nms':.7,'global_merge':.5,'match_iou':.5,'passes':1},'scope':'Same 58 TRAINING source images, searched with full fixed grid. Measures training fit, not generalization. No latency benchmark. No changes to checkpoints, controls or thresholds.'}
    assert all(sha(p)==h for p,h in json.loads((OUT/'protected_artifacts.json').read_text(encoding='utf-8')).items());waid.save_json(target,result)
    print(json.dumps({stage:{k:v for k,v in r.items() if k!='items'} for stage,r in models.items()}),flush=True)
if __name__=='__main__':main()
