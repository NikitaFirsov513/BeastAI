"""Check learning on the 27 unique roe training frames; this is NOT validation quality."""
from project_paths import PROJECT_ROOT
from pathlib import Path
from collections import Counter
import argparse,hashlib,json,os
import baseline,waid
from tiled_inference import map_box,merge_boxes
ROOT=PROJECT_ROOT;OUT=ROOT/'data/mixed_pilot'
for key,folder in [('YOLO_CONFIG_DIR','ultralytics-config'),('MPLCONFIGDIR','matplotlib-cache'),('TORCH_HOME','torch-cache')]:os.environ[key]=str(ROOT/'data'/folder)
os.environ['YOLO_AUTOINSTALL']='false'

def main():
    global OUT
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--experiment',choices=('mixed_pilot','roe_focus','roe_expanded'),default='mixed_pilot');args=parser.parse_args();OUT=ROOT/'data'/args.experiment
    import cv2,numpy as np,torch
    from ultralytics import YOLO,settings
    settings.update({'sync':False});torch.set_num_threads(4);cv2.setNumThreads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    index=json.loads((OUT/'index.json').read_text(encoding='utf-8'));run=json.loads((OUT/'latest_run.json').read_text(encoding='utf-8'))
    entries=index['roe_training'];assert entries
    result={'scope':'Training-fit diagnostic only. These images trained the model; this cannot estimate generalization.','dataset_revision':index['revision'],'conditions':{'input_shape':[1,3,1024,1024],'confidence_threshold':.25,'nms_iou':.7,'global_merge_iou':.5,'match_iou':.5,'precision':'FP32, TF32 disabled','batch_size':1},'models':{}}
    def size_group(area):return 'below16' if area<16**2 else '16to32' if area<32**2 else '32to64' if area<64**2 else 'atleast64'
    for stage,field in [('before','base_weights'),('after','weights')]:
        model=YOLO(run[field]);aggregate=Counter();items={};sizes={k:{'reference':0,'found':0} for k in ('below16','16to32','32to64','atleast64')}
        for e in entries:
            imagepath=Path(e['image_path']);labelpath=Path(e['label_path'])
            assert hashlib.sha256(imagepath.read_bytes()).hexdigest()==e['image_sha256'] and hashlib.sha256(labelpath.read_bytes()).hexdigest()==e['label_sha256']
            image=cv2.imdecode(np.fromfile(imagepath,dtype=np.uint8),cv2.IMREAD_COLOR);height,width=image.shape[:2]
            r=model.predict(source=image,imgsz=1024,rect=False,device=0,half=False,batch=1,conf=.25,iou=.7,agnostic_nms=True,classes=[0],max_det=300,verbose=False,save=False)[0]
            boxes=[map_box(row,(0,0,width,height),(width,height),model.names) for row in r.boxes.data.detach().cpu().tolist()];boxes=merge_boxes([b for b in boxes if b],.5,300)
            truth=waid.parse_labels(labelpath,['animal'])['boxes'];match=baseline.match_boxes(boxes,truth,.5);aggregate.update({k:match[k] for k in ('tp','fp','fn')})
            scale=min(1024/width,1024/height)
            for i,b in enumerate(truth):
                group=size_group(b['width']*width*b['height']*height*scale**2);sizes[group]['reference']+=1;sizes[group]['found']+=int(i in match['matched_truth'])
            coverage=[{'reference_index':i,'matched':i in match['matched_truth'],'best_emitted_prediction_iou':max((baseline.overlap(b,p) for p in boxes),default=0.)} for i,b in enumerate(truth)]
            items[e['name']]={'image_sha256':e['image_sha256'],'label_sha256':e['label_sha256'],'reference':truth,'predictions':boxes,'counts':{k:match[k] for k in ('tp','fp','fn')},'reference_coverage':coverage}
        result['models'][stage]={'weights_path':run[field],'weights_sha256':hashlib.sha256(Path(run[field]).read_bytes()).hexdigest(),'counts':dict(aggregate),'images':len(entries),'reference_boxes':sum(e['reference_boxes'] for e in entries),'size_groups':sizes,'items':items}
        print(stage,dict(aggregate),sizes,flush=True)
        del model
    comparison=json.loads((OUT/'evaluation/after.json').read_text(encoding='utf-8'))
    result['control_roe_size_groups']=comparison['sources']['roe']['quality']['size_groups']
    waid.save_json(OUT/'evaluation/training_fit_diagnostic.json',result)

if __name__=='__main__':main()
