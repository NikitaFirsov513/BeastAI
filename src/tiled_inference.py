"""Compare full-frame and four overlapping crops using one frozen detector, batch 1."""
from project_paths import PROJECT_ROOT
from datetime import datetime,timezone
from pathlib import Path
from collections import Counter
import hashlib,json,math,os,platform,statistics,time,subprocess
import baseline,waid
ROOT=PROJECT_ROOT
DATA=ROOT/'data/sherbrooke';OUT=DATA/'tiling'
MODES=('full','full_merge','tiles','tiles_native640')
LABELS={'full':'Целый кадр','full_merge':'Целый кадр + фильтр повторов','tiles':'4 увеличенных фрагмента','tiles_native640':'4 фрагмента без увеличения'}


def axis_starts(length,size,overlap=.25):
    if length<=0 or size<=0 or not 0<=overlap<1:raise ValueError('Invalid crop geometry')
    size=min(size,length);last=length-size;step=max(1,int(size*(1-overlap)))
    starts=list(range(0,last+1,step))
    if starts[-1]!=last:starts.append(last)
    return starts


def tile_windows(width,height):
    tw,th=min(640,width),min(448,height)
    return [(x,y,x+tw,y+th) for y in axis_starts(height,th) for x in axis_starts(width,tw)]


def map_box(row,window,shape,class_names):
    x1,y1,x2,y2,score,category=row
    left,top,right,bottom=window;w,h=shape
    x1=max(0.,min(right-left,x1));x2=max(0.,min(right-left,x2))
    y1=max(0.,min(bottom-top,y1));y2=max(0.,min(bottom-top,y2))
    if x2<=x1 or y2<=y1:return None
    return {'x':(left+x1)/w,'y':(top+y1)/h,'width':(x2-x1)/w,'height':(y2-y1)/h,'score':float(score),'class_id':int(category),'class_name':class_names[int(category)]}


def merge_boxes(boxes,iou=.5,max_det=300):
    ordered=sorted(boxes,key=lambda b:b['score'],reverse=True);kept=[]
    for box in ordered:
        if all(baseline.overlap(box,other)<=iou for other in kept):
            kept.append(box)
            if len(kept)>=max_det:break
    return kept


def main():
    for key,folder in [('YOLO_CONFIG_DIR','ultralytics-config'),('MPLCONFIGDIR','matplotlib-cache'),('TORCH_HOME','torch-cache')]:
        path=ROOT/'data'/folder;path.mkdir(parents=True,exist_ok=True);os.environ[key]=str(path)
    os.environ['YOLO_AUTOINSTALL']='false'
    import cv2,numpy as np,psutil,torch,torchvision,ultralytics
    from ultralytics import YOLO,settings
    settings.update({'sync':False})
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no silent CPU fallback')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    waid.configure_dataset('sherbrooke');index=waid.get_index();pairs=index['pairs']
    assert Counter(p['split'] for p in pairs)=={'train':64,'valid':32}
    training=json.loads((DATA/'train512/latest_run.json').read_text(encoding='utf-8'))
    weights=Path(training['weights']);weight_sha=hashlib.sha256(weights.read_bytes()).hexdigest()
    assert weight_sha==training['weights_sha256']=='ac1229f41c338a8c3913e792857c27fa40b5b7f52a17bd2772e4d4355b6beead'
    for pair in pairs:
        for kind in ('image','label'):
            assert hashlib.sha256(waid.local_file(pair[kind]).read_bytes()).hexdigest()==pair[kind]['sha256'],pair['name']
    run=OUT/'runs'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');run.mkdir(parents=True,exist_ok=False)
    plan={'model_sha256':weight_sha,'dataset_revision':index['revision'],'modes':list(MODES),'network_input_by_mode':{mode:[1,3,640,640] if mode=='tiles_native640' else [1,3,1024,1024] for mode in MODES},'confidence_threshold':.25,'local_nms_iou':.7,'global_merge_iou':.5,'crop_size':[640,448],'axis_minimum_overlap':.25,'batch_size':1,'repeats':3,'warmup_frames_per_mode':5,'small_reference_rule':'Fixed full-frame scale at 1024, area < 32**2','checkpoint':'Same fixed epoch20 train512 weights; no new training','timing_scope':'All original 96 frames, quality only original valid32. Mode order rotates per frame and repeat.'}
    waid.save_json(run/'plan.json',plan)
    model=YOLO(str(weights));animal_ids=baseline.animal_class_ids(model.names)
    config=dict(imgsz=1024,rect=False,device=0,half=False,batch=1,conf=.25,iou=.7,agnostic_nms=True,classes=animal_ids,max_det=300,verbose=False,save=False)

    def predict(path,mode):
        torch.cuda.synchronize();start=time.perf_counter()
        image=cv2.imdecode(np.fromfile(path,dtype=np.uint8),cv2.IMREAD_COLOR)
        if image is None:raise ValueError('Decode failed: '+str(path))
        height,width=image.shape[:2];read_ms=(time.perf_counter()-start)*1000
        slicing_start=time.perf_counter()
        windows=tile_windows(width,height) if mode.startswith('tiles') else [(0,0,width,height)]
        views=[image[y1:y2,x1:x2] for x1,y1,x2,y2 in windows]
        slice_ms=(time.perf_counter()-slicing_start)*1000
        boxes=[];stages=Counter();local_cap=False
        for view,window in zip(views,windows):
            result=model.predict(source=view,**{**config,'imgsz':640 if mode=='tiles_native640' else 1024})[0]
            local_cap|=len(result.boxes)>=300
            for row in result.boxes.data.detach().cpu().tolist():
                mapped=map_box(row,window,(width,height),model.names)
                if mapped:boxes.append(mapped)
            for key,value in result.speed.items():stages[key+'_ms']+=float(value)
        candidates=len(boxes)
        merge_start=time.perf_counter()
        if mode!='full':boxes=merge_boxes(boxes,.5,300)
        merge_ms=(time.perf_counter()-merge_start)*1000
        torch.cuda.synchronize();total=(time.perf_counter()-start)*1000
        return boxes,{'total_ms':total,'read_decode_ms':read_ms,'slicing_ms':slice_ms,**dict(stages),'global_merge_ms':merge_ms},[width,height],{'network_calls':len(windows),'windows':[list(w) for w in windows],'candidates_before_merge':candidates,'per_crop_cap_reached':local_cap,'global_cap_reached':mode!='full' and len(boxes)>=300}

    first_path=waid.local_file(pairs[0]['image']);cold={}
    for mode in MODES:
        _,cold[mode],_,_=predict(first_path,mode)
        for _ in range(5):predict(first_path,mode)
    items={mode:{} for mode in MODES};timings={mode:[] for mode in MODES};peak={mode:0 for mode in MODES};reserved={mode:0 for mode in MODES};rss={mode:0 for mode in MODES}
    for number,pair in enumerate(pairs):
        truth=waid.parse_labels(waid.local_file(pair['label']),index['classes'])
        assert truth['status'] in ('ok','empty')
        stored={mode:[] for mode in MODES};outputs={}
        for repeat in range(3):
            offset=(number+repeat)%len(MODES);order=MODES[offset:]+MODES[:offset]
            for mode in order:
                torch.cuda.reset_peak_memory_stats()
                boxes,timing,shape,extra=predict(waid.local_file(pair['image']),mode)
                peak[mode]=max(peak[mode],torch.cuda.max_memory_allocated());reserved[mode]=max(reserved[mode],torch.cuda.max_memory_reserved());rss[mode]=max(rss[mode],psutil.Process().memory_info().rss)
                timings[mode].append(timing);stored[mode].append(timing);outputs[mode]=(boxes,shape,extra)
        for mode in MODES:
            boxes,shape,extra=outputs[mode];matched=baseline.match_boxes(boxes,truth['boxes'],.5)
            scale=min(1024/shape[0],1024/shape[1]);small={i for i,b in enumerate(truth['boxes']) if b['width']*shape[0]*scale*b['height']*shape[1]*scale<32**2}
            items[mode][pair['split']+'/'+pair['name']]={'boxes':boxes,'timings':stored[mode],'flight_id':pair.get('flight_id'),'original_size':shape,'image_git_sha':pair['image']['sha'],'label_git_sha':pair['label']['sha'],'reference_boxes':len(truth['boxes']),'counts':{k:matched[k] for k in ('tp','fp','fn')},'small_reference':len(small),'small_matched':len(small.intersection(matched['matched_truth'])),'tiling':extra}
        if (number+1)%8==0 or number==0:
            print(f'[{number+1}/96] all four modes measured',flush=True)
            waid.save_json(OUT/'status.json',{'state':'benchmarking','images_completed':number+1,'images':96,'run':str(run)})
    try:driver=subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,memory.total,power.draw,temperature.gpu','--format=csv,noheader'],capture_output=True,text=True,timeout=10).stdout.strip()
    except (OSError,subprocess.TimeoutExpired):driver='unavailable'
    reports={}
    for mode in MODES:
        valid=[v for key,v in items[mode].items() if key.startswith('valid/')]
        counts={k:sum(v['counts'][k] for v in valid) for k in ('tp','fp','fn')}
        counts.update(images=32,precision=counts['tp']/(counts['tp']+counts['fp']) if counts['tp']+counts['fp'] else None,recall=counts['tp']/115,small_reference=sum(v['small_reference'] for v in valid),small_matched=sum(v['small_matched'] for v in valid))
        counts['small_recall']=counts['small_matched']/counts['small_reference']
        c={'gpu':torch.cuda.get_device_name(0),'driver_snapshot':driver,'cpu':platform.processor(),'os':platform.platform(),'ram_total_bytes':psutil.virtual_memory().total,'python':platform.python_version(),'torch':torch.__version__,'torchvision':torchvision.__version__,'ultralytics':ultralytics.__version__,'cuda':torch.version.cuda,'input_shape':[1,3,640,640] if mode=='tiles_native640' else [1,3,1024,1024],'precision':'FP32; TF32 disabled','model_dtype':str(next(model.model.parameters()).dtype),'batch_size':1,'cpu_threads':4,'confidence_threshold':.25,'nms_iou':.7,'class_agnostic_nms':True,'match_iou':.5,'repeats':3,'warmup_frames':5,'power_energy':'not measured; power snapshot is not per-frame energy','tiling_mode':mode,'global_merge_iou':.5 if mode!='full' else None,'network_calls_per_frame':4 if mode.startswith('tiles') else 1,'crop_size':[640,448] if mode.startswith('tiles') else None}
        report={'created_utc':datetime.now(timezone.utc).isoformat(),'dataset':'sherbrooke','dataset_revision':index['revision'],'source_urls':index['source_urls'],'selection_notes':index['selection'],'annotation_coverage':index['annotation_coverage'],'model':LABELS[mode]+' · YOLO11n, 512 снимков','model_path':str(weights.relative_to(ROOT)),'model_sha256':weight_sha,'model_file_bytes':weights.stat().st_size,'parameters':sum(p.numel() for p in model.model.parameters()),'conditions':c,'memory':{'gpu_peak_allocated_bytes':peak[mode],'gpu_peak_reserved_bytes':reserved[mode],'process_rss_bytes':rss[mode]},'cold_first_frame':cold[mode],'timing_summary':{k:{'median':statistics.median(t[k] for t in timings[mode]),'p95':baseline.percentile([t[k] for t in timings[mode]],.95)} for k in timings[mode][0]},'validation_preview':counts,'items':items[mode],'notes':['Full latency includes file read/decode, crop views, all sequential network calls, CPU coordinate mapping and global duplicate filtering.','Model load, warmup, label reading, quality matching, JSON and viewer are excluded. Repeated reads may use OS cache.','Full_merge is full frame followed by the exact same extra overlap filter used for tiled results; it does not re-run the original internal NMS at another setting.','GPU peak allocated is a maximum over individual runs; process RAM is RSS sampled outside the timed prediction.','Additional global filtering uses class-agnostic greedy NMS at IoU 0.5. Crops retain internal detector NMS 0.7.','Fixed small objects are determined at full-frame 1024 scale, not recalculated after crop magnification.']}
        waid.save_json(run/(mode+'.json'),report);reports[mode]=str(run/(mode+'.json'))
        print(json.dumps({'mode':mode,'quality':counts,'latency':report['timing_summary']['total_ms'],'gpu_MiB':peak[mode]/2**20}),flush=True)
    assert hashlib.sha256(weights.read_bytes()).hexdigest()==weight_sha
    assert all(not v['tiling']['per_crop_cap_reached'] and not v['tiling']['global_cap_reached'] for report_mode in items.values() for v in report_mode.values())
    waid.save_json(OUT/'latest_run.json',{'run':str(run),'reports':reports,'plan':plan})
    waid.save_json(OUT/'status.json',{'state':'benchmarks_completed','run':str(run),'images_completed':96,'images':96})

if __name__=='__main__':main()
