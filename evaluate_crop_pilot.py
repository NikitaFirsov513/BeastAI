"""Compare full-frame vs fixed-grid search, without reference access during prediction."""
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import argparse,hashlib,json,os,platform,statistics,subprocess,time
import waid,baseline
from tiled_inference import axis_starts,map_box,merge_boxes
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'data/crop_pilot'
for key,folder in [('YOLO_CONFIG_DIR','ultralytics-config'),('MPLCONFIGDIR','matplotlib-cache'),('TORCH_HOME','torch-cache')]:os.environ[key]=str(ROOT/'data'/folder)
os.environ['YOLO_AUTOINSTALL']='false'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def windows(width,height,mode):
    if mode=='full':return [(0,0,width,height)]
    if mode!='tiles':raise ValueError('Unknown search mode')
    size=512
    return [(x,y,x+min(size,width),y+min(size,height)) for y in axis_starts(height,size,.25) for x in axis_starts(width,size,.25)]
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',required=True,choices=('before','after'));args=parser.parse_args()
    import cv2,numpy as np,psutil,torch,torchvision,ultralytics
    from ultralytics import YOLO,settings
    settings.update({'sync':False});torch.set_num_threads(4);cv2.setNumThreads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    assert ROOT.drive.lower()=='d:' and torch.cuda.is_available()
    target=OUT/'evaluation'/(args.stage+'.json')
    if target.exists():raise RuntimeError('Preserve completed evaluation')
    index=json.loads((OUT/'index.json').read_text(encoding='utf-8'));plan=json.loads((OUT/'experiment_plan.json').read_text(encoding='utf-8'));run=json.loads((OUT/'latest_run.json').read_text(encoding='utf-8'))
    key='base_weights' if args.stage=='before' else 'weights';weights=Path(run[key]);assert sha(weights)==run[key+'_sha256']
    assert index['revision']==run['dataset_revision']==plan['dataset_revision']
    entries=[{**e,'evaluation_split':split} for split,rows in [('control',index['validation']),('fresh',index['fresh_validation'])] for e in rows]
    refs={}
    for e in entries:
        assert sha(e['image_path'])==e['image_sha256'] and sha(e['label_path'])==e['label_sha256']
        refs[e['source']+'/'+e['name']]=waid.parse_labels(Path(e['label_path']),['animal'])['boxes']
    load_start=time.perf_counter();model=YOLO(str(weights));load_ms=(time.perf_counter()-load_start)*1000;assert model.names=={0:'animal'}
    cfg=dict(imgsz=1024,rect=False,device=0,half=False,batch=1,conf=.25,iou=.7,agnostic_nms=True,classes=[0],max_det=300,verbose=False,save=False)
    def predict(path,mode):
        # No labels, source metadata or target centers are available here.
        torch.cuda.synchronize();start=time.perf_counter()
        image=cv2.imdecode(np.fromfile(path,dtype=np.uint8),cv2.IMREAD_COLOR)
        if image is None:raise RuntimeError('Image decode failed')
        h,w=image.shape[:2];read_ms=(time.perf_counter()-start)*1000
        slice_start=time.perf_counter();grid=windows(w,h,mode);views=[image[y1:y2,x1:x2] for x1,y1,x2,y2 in grid];slice_ms=(time.perf_counter()-slice_start)*1000
        boxes=[];stages=Counter();cpu_ms=0.;max_local=0
        for view,window in zip(views,grid):
            result=model.predict(source=view,**cfg)[0];max_local=max(max_local,len(result.boxes))
            cpu_start=time.perf_counter()
            for row in result.boxes.data.detach().cpu().tolist():
                b=map_box(row,window,(w,h),model.names)
                if b:boxes.append(b)
            cpu_ms+=(time.perf_counter()-cpu_start)*1000
            for k,v in result.speed.items():stages[k+'_ms']+=float(v)
        candidates=len(boxes);merge_start=time.perf_counter();boxes=merge_boxes(boxes,.5,300);merge_ms=(time.perf_counter()-merge_start)*1000
        torch.cuda.synchronize();total_ms=(time.perf_counter()-start)*1000
        assert max_local<300 and len(boxes)<300,'Prediction cap reached'
        return boxes,{'total_ms':total_ms,'read_decode_ms':read_ms,'slicing_ms':slice_ms,**dict(stages),'cpu_box_mapping_ms':cpu_ms,'global_merge_ms':merge_ms},[w,h],{'network_calls':len(grid),'windows':[list(v) for v in grid],'candidates_before_merge':candidates,'max_local_predictions':max_local}
    # Geometric regression: full coverage independent of target locations, even for smaller inputs.
    for w,h in [(1280,1280),(1024,683),(300,240)]:
        cover=np.zeros((h,w),np.uint8)
        for x1,y1,x2,y2 in windows(w,h,'tiles'):cover[y1:y2,x1:x2]+=1
        assert cover.min()>=1
    cold={}
    for mode in ('full','tiles'):
        _,cold[mode],_,_=predict(entries[0]['image_path'],mode)
        for _ in range(5):predict(entries[0]['image_path'],mode)
    items={mode:{} for mode in ('full','tiles')};peak=Counter();reserved=Counter();rss=Counter()
    for number,e in enumerate(entries):
        name=e['source']+'/'+e['name'];truth=refs[name];stored={mode:[] for mode in items}
        for repeat in range(3):
            order=('full','tiles') if (number+repeat)%2==0 else ('tiles','full')
            for mode in order:
                torch.cuda.reset_peak_memory_stats();boxes,timing,shape,extra=predict(e['image_path'],mode)
                peak[mode]=max(peak[mode],torch.cuda.max_memory_allocated());reserved[mode]=max(reserved[mode],torch.cuda.max_memory_reserved());rss[mode]=max(rss[mode],psutil.Process().memory_info().rss)
                matched=baseline.match_boxes(boxes,truth,.5);counts={k:matched[k] for k in ('tp','fp','fn')}
                stored[mode].append({'boxes':boxes,'timing':timing,'shape':shape,'extra':extra,'counts':counts,'found':matched['matched_truth']})
        for mode,observations in stored.items():
            assert all(o['counts']==observations[0]['counts'] for o in observations),'Repeat counts changed'
            r=observations[0];found=set(r['found']);w,h=r['shape'];scale=min(1024/w,1024/h)
            groups={key:{'reference':0,'found':0} for key in ('below16','16to32','32to64','atleast64')}
            small=set()
            for i,b in enumerate(truth):
                area=b['width']*w*b['height']*h*scale**2;group='below16' if area<16**2 else '16to32' if area<32**2 else '32to64' if area<64**2 else 'atleast64'
                groups[group]['reference']+=1;groups[group]['found']+=int(i in found)
                if area<32**2:small.add(i)
            items[mode][name]={**e,'reference':truth,'boxes':r['boxes'],'counts':r['counts'],'repeat_counts':[o['counts'] for o in observations],'timings':[o['timing'] for o in observations],'original_size':r['shape'],'small_reference':len(small),'small_matched':len(small&found),'size_groups':groups,'matched_reference_indices':r['found'],'search':r['extra']}
        if (number+1)%4==0:
            waid.save_json(OUT/'evaluation_status.json',{'state':'evaluating','stage':args.stage,'images_completed':number+1,'images':80});print(f'{args.stage}: {number+1}/80 full+tiles',flush=True)
    def timing_summary(times):return {k:{'median':statistics.median(t[k] for t in times),'p95':baseline.percentile([t[k] for t in times],.95)} for k in times[0]}
    modes={}
    for mode,rows_dict in items.items():
        partitions={}
        for split in ('control','fresh'):
            sources={}
            for source in ('sherbrooke','roe'):
                rows=[r for r in rows_dict.values() if r['evaluation_split']==split and r['source']==source]
                if not rows:continue
                count={k:sum(r['counts'][k] for r in rows) for k in ('tp','fp','fn')};nref=sum(len(r['reference']) for r in rows)
                assert count['tp']+count['fn']==nref
                quality={'images':len(rows),'reference_boxes':nref,**count,'precision':count['tp']/(count['tp']+count['fp']) if count['tp']+count['fp'] else None,'recall':count['tp']/nref,'small_reference':sum(r['small_reference'] for r in rows),'small_matched':sum(r['small_matched'] for r in rows),'background_images':sum(not r['reference'] for r in rows),'false_boxes_on_background':sum(r['counts']['fp'] for r in rows if not r['reference']),'size_groups':{g:{k:sum(r['size_groups'][g][k] for r in rows) for k in ('reference','found')} for g in rows[0]['size_groups']}}
                sources[source]={'quality':quality,'timing_summary':timing_summary([t for r in rows for t in r['timings']]),'network_calls_per_frame':sorted({r['search']['network_calls'] for r in rows})}
            partitions[split]=sources
        modes[mode]={'partitions':partitions,'timing_summary':timing_summary([t for r in rows_dict.values() for t in r['timings']]),'memory':{'gpu_peak_allocated_bytes':peak[mode],'gpu_peak_reserved_bytes':reserved[mode],'process_rss_bytes':rss[mode]},'items':rows_dict}
    if args.stage=='before':
        for split,filename in [('control','after.json'),('fresh','fresh_after.json')]:
            previous=json.loads((ROOT/'data/roe_expanded/evaluation'/filename).read_text(encoding='utf-8'))
            for source,data in previous['sources'].items():
                assert {k:modes['full']['partitions'][split][source]['quality'][k] for k in ('tp','fp','fn')}=={k:data['quality'][k] for k in ('tp','fp','fn')},'Frozen full-frame baseline changed'
    conditions={'gpu':torch.cuda.get_device_name(0),'driver_snapshot':subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,memory.total,power.draw,temperature.gpu','--format=csv,noheader'],capture_output=True,text=True,timeout=10).stdout.strip(),'cpu':platform.processor(),'os':platform.platform(),'ram_total_bytes':psutil.virtual_memory().total,'python':platform.python_version(),'torch':torch.__version__,'torchvision':torchvision.__version__,'ultralytics':ultralytics.__version__,'cuda':torch.version.cuda,'input_shape':[1,3,1024,1024],'batch_size':1,'precision':'FP32; AMP and TF32 off','actual_model_dtype':str(next(model.model.parameters()).dtype),'cpu_threads':4,'opencv_threads':cv2.getNumThreads(),'confidence_threshold':.25,'nms_iou':.7,'global_merge_iou':.5,'match_iou':.5,'class_agnostic_nms':True,'small_reference_scale':1024,'tile_source_size':[512,512],'minimum_tile_overlap':.25,'network_calls_by_mode':{'full':1,'tiles':{'roe':9,'sherbrooke':6}},'repeats':3,'warmup_frames_per_mode':5,'order':'Modes alternate per frame/repeat; before/after separate sequential processes. Clocks and power state not fixed.','energy':'Not measured. Power snapshot is not per-frame energy.'}
    limitations=['All 80 frames excluded from this training. Roe visual groups are not verified flight IDs; this is development evidence, not independent final testing.','Full frame search uses original pixels and fixed grid only, not reference-centered crops.','Frozen small group uses the full-frame1024 scale in all modes.','Warm latency includes image read/decode, slicing, input preparation, every network call, internal NMS, CPU mapping/transfer and global duplicate filtering.','Model load, cold first call, warmup, reference reading/scoring, reports and browser excluded. Repeated reads may use OS file cache.','GPU allocated memory excludes driver/other processes; RSS includes Python/Torch.','Counts concern animal appearances in images, not video observation events.','No energy measurement or future1TOPS speed prediction.']
    result={'created_utc':datetime.now(timezone.utc).isoformat(),'stage':args.stage,'dataset_revision':index['revision'],'source_dataset_revision':index['source_dataset_revision'],'model_path':str(weights),'model_sha256':sha(weights),'model_file_bytes':weights.stat().st_size,'parameters':sum(p.numel() for p in model.model.parameters()),'conditions':conditions,'model_constructor_ms':load_ms,'cold_first_frame':cold,'quality_repeat_counts_stable':True,'modes':modes,'limitations':limitations}
    assert all(sha(p)==h for p,h in json.loads((OUT/'protected_artifacts.json').read_text(encoding='utf-8')).items())
    waid.save_json(target,result);waid.save_json(OUT/'evaluation_status.json',{'state':args.stage+'_evaluated','images':80});print(json.dumps({'stage':args.stage,'modes':{mode:r['partitions'] for mode,r in modes.items()}}),flush=True)
if __name__=='__main__':main()
