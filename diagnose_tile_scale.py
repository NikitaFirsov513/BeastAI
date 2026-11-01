"""Diagnose crop geometry and scale sensitivity; quality-only, one pass, not timing benchmark."""
from collections import Counter
from pathlib import Path
import hashlib,json,os
import baseline,waid
from tiled_inference import tile_windows,map_box,merge_boxes
ROOT=Path(__file__).resolve().parent;DATA=ROOT/'data/sherbrooke';OUT=DATA/'tiling'

def main():
    for k,d in [('YOLO_CONFIG_DIR','ultralytics-config'),('MPLCONFIGDIR','matplotlib-cache'),('TORCH_HOME','torch-cache')]:os.environ[k]=str(ROOT/'data'/d)
    os.environ['YOLO_AUTOINSTALL']='false'
    import cv2,numpy as np,torch
    from ultralytics import YOLO,settings
    settings.update({'sync':False});torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    waid.configure_dataset('sherbrooke');index=waid.get_index();training=json.loads((DATA/'train512/latest_run.json').read_text(encoding='utf-8'))
    model=YOLO(training['weights']);base=dict(rect=False,device=0,half=False,batch=1,conf=.25,iou=.7,agnostic_nms=True,max_det=300,verbose=False,save=False)
    modes=[('tiles_1024',1024,True),('tiles_640_native_scale',640,True),('full_1632_magnified',1632,False)]
    rows={name:{} for name,_,_ in modes};counts={name:Counter() for name,_,_ in modes}
    checks=0;nonzero_checks=0;max_difference=0.
    for pair in index['pairs']:
        if pair['split']!='valid':continue
        truth=waid.parse_labels(waid.local_file(pair['label']),index['classes'])['boxes']
        image=cv2.imdecode(np.fromfile(waid.local_file(pair['image']),dtype=np.uint8),cv2.IMREAD_COLOR);h,w=image.shape[:2]
        small={i for i,b in enumerate(truth) if b['width']*w*b['height']*h<32**2}
        for name,imgsz,tiled in modes:
            boxes=[];candidates=0
            windows=tile_windows(w,h) if tiled else [(0,0,w,h)]
            for window in windows:
                a,b,c,d=window;crop=image[b:d,a:c];result=model.predict(source=crop,imgsz=imgsz,**base)[0]
                assert result.orig_shape==crop.shape[:2] and np.array_equal(result.orig_img,crop)
                for raw in result.boxes.data.detach().cpu().tolist():
                    mapped=map_box(raw,window,(w,h),model.names)
                    if mapped is None:continue
                    boxes.append(mapped);candidates+=1
                    if name=='tiles_1024':
                        x1,y1,x2,y2=raw[:4]
                        local={'x':max(0,x1)/(c-a),'y':max(0,y1)/(d-b),'width':(min(c-a,x2)-max(0,x1))/(c-a),'height':(min(d-b,y2)-max(0,y1))/(d-b)}
                        for ref in truth:
                            gx,gy,gw,gh=ref['x']*w,ref['y']*h,ref['width']*w,ref['height']*h
                            if a<=gx and b<=gy and gx+gw<=c and gy+gh<=d:
                                local_ref={'x':(gx-a)/(c-a),'y':(gy-b)/(d-b),'width':gw/(c-a),'height':gh/(d-b)}
                                before=baseline.overlap(local,local_ref);after=baseline.overlap(mapped,ref)
                                checks+=1;nonzero_checks+=before>0;max_difference=max(max_difference,abs(before-after))
                                assert abs(before-after)<1e-10
            boxes=merge_boxes(boxes,.5,300);matched=baseline.match_boxes(boxes,truth,.5)
            counts[name].update({k:matched[k] for k in ('tp','fp','fn')});counts[name]['small_tp']+=len(small.intersection(matched['matched_truth']));counts[name]['candidates_before_merge']+=candidates
            rows[name][pair['name']]={'counts':{k:matched[k] for k in ('tp','fp','fn')},'candidates_before_merge':candidates,'boxes':boxes}
    assert counts['tiles_1024']['tp']==14 and counts['tiles_1024']['fp']==6 and counts['tiles_1024']['fn']==101
    assert checks>0 and nonzero_checks>0
    output={'model_sha256':training['weights_sha256'],'dataset_revision':index['revision'],'quality_only':True,'repeats':1,'confidence_threshold':.25,'local_nms_iou':.7,'global_merge_iou':.5,'conditions':{'gpu':torch.cuda.get_device_name(0),'precision':'FP32; TF32 disabled','batch_size':1},'geometry_verification':{'orig_shape_and_pixels_equal':True,'iou_invariance_checks':checks,'positive_overlap_checks':nonzero_checks,'max_iou_difference':max_difference,'prior_tiles_result_reproduced':True},'modes':{name:{'input_size':imgsz,'crops':tiled,'counts':dict(counts[name]),'items':rows[name]} for name,imgsz,tiled in modes},'notes':['Diagnostic quality run only. No latency comparison is inferred from this one-pass experiment.','A larger network input does not recover source details absent from the downloaded 1024px derivatives.']}
    waid.save_json(OUT/'scale_diagnosis.json',output)
    print(json.dumps({'geometry':output['geometry_verification'],'counts':{name:dict(c) for name,c in counts.items()}}),flush=True)

if __name__=='__main__':main()
