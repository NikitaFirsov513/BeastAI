"""Measure one fixed model on the same 64 control frames, batch 1 and FP32."""
from pathlib import Path
from collections import Counter
from datetime import datetime,timezone
import argparse,hashlib,json,os,platform,statistics,subprocess,time
import baseline,waid
from tiled_inference import map_box,merge_boxes
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'data/mixed_pilot';EVAL=OUT/'evaluation'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    global OUT,EVAL
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--model',choices=('before','previous','after'),required=True);parser.add_argument('--experiment',choices=('mixed_pilot','roe_focus','roe_expanded'),default='mixed_pilot');parser.add_argument('--split',choices=('control','fresh'),default='control');parser.add_argument('--imgsz',type=int,choices=(1024,1280),default=1024);args=parser.parse_args()
    OUT=ROOT/'data'/args.experiment;EVAL=OUT/'evaluation'
    assert ROOT.drive.lower()=='d:'
    for key,folder in [('YOLO_CONFIG_DIR','ultralytics-config'),('MPLCONFIGDIR','matplotlib-cache'),('TORCH_HOME','torch-cache')]:os.environ[key]=str(ROOT/'data'/folder)
    os.environ['YOLO_AUTOINSTALL']='false'
    import cv2,numpy as np,torch,torchvision,ultralytics,psutil
    from ultralytics import YOLO,settings
    settings.update({'sync':False});torch.set_num_threads(4);cv2.setNumThreads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    index=json.loads((OUT/'index.json').read_text(encoding='utf-8'));entries=index['validation'] if args.split=='control' else index['fresh_validation'];assert entries
    if args.split=='control':assert len(entries)==64
    training=json.loads((OUT/'latest_run.json').read_text(encoding='utf-8'))
    weight_key={'before':('comparison_weights' if 'comparison_weights' in training else 'base_weights'),'previous':'previous_mixed_weights','after':'weights'}[args.model]
    weights=Path(training[weight_key]);weight_sha=sha(weights)
    assert weight_sha==training[weight_key+'_sha256']
    assert index['revision']==training['dataset_revision']
    references={}
    for e in entries:
        assert sha(e['image_path'])==e['image_sha256'] and sha(e['label_path'])==e['label_sha256']
        parsed=waid.parse_labels(Path(e['label_path']),['animal']);assert parsed['status'] in ('ok','empty')
        assert len(parsed['boxes'])==e['reference_boxes'];references[e['source']+'/'+e['name']]=parsed['boxes']
    EVAL.mkdir(exist_ok=True)
    model=YOLO(str(weights));assert model.names=={0:'animal'}
    config=dict(imgsz=args.imgsz,rect=False,device=0,half=False,batch=1,conf=.25,iou=.7,agnostic_nms=True,classes=[0],max_det=300,verbose=False,save=False)
    def predict(path):
        torch.cuda.synchronize();start=time.perf_counter()
        image=cv2.imdecode(np.fromfile(path,dtype=np.uint8),cv2.IMREAD_COLOR)
        if image is None:raise ValueError('Decode failed: '+path)
        height,width=image.shape[:2];read_ms=(time.perf_counter()-start)*1000
        result=model.predict(source=image,**config)[0];assert result.orig_shape==(height,width)
        transfer_start=time.perf_counter();boxes=[]
        for row in result.boxes.data.detach().cpu().tolist():
            box=map_box(row,(0,0,width,height),(width,height),model.names)
            if box:boxes.append(box)
        transfer_ms=(time.perf_counter()-transfer_start)*1000;candidates=len(boxes)
        merge_start=time.perf_counter();boxes=merge_boxes(boxes,.5,300);merge_ms=(time.perf_counter()-merge_start)*1000
        torch.cuda.synchronize();total=(time.perf_counter()-start)*1000
        return boxes,{'total_ms':total,'read_decode_ms':read_ms,**{k+'_ms':float(v) for k,v in result.speed.items()},'cpu_boxes_ms':transfer_ms,'additional_merge_ms':merge_ms},[width,height],candidates
    load_start=time.perf_counter();_,cold,_,_=predict(entries[0]['image_path']);first_call=(time.perf_counter()-load_start)*1000
    for _ in range(5):predict(entries[0]['image_path'])
    torch.cuda.reset_peak_memory_stats();items={};all_times=[];rss=0
    for number,e in enumerate(entries,1):
        key=e['source']+'/'+e['name'];truth=references[key];times=[];counts=[];outputs=[];caps=[]
        for repeat in range(3):
            boxes,timing,shape,candidates=predict(e['image_path']);times.append(timing);all_times.append(timing);outputs.append(boxes);caps.append(candidates)
            match=baseline.match_boxes(boxes,truth,.5);counts.append({k:match[k] for k in ('tp','fp','fn')})
        assert all(c==counts[0] for c in counts),'Quality counts vary across repeats'
        assert max(caps)<300,'Detection cap reached'
        boxes=outputs[0];matched=baseline.match_boxes(boxes,truth,.5);scale=min(1024/shape[0],1024/shape[1])
        sizes={i:b['width']*shape[0]*b['height']*shape[1]*scale**2 for i,b in enumerate(truth)}
        small={i for i,area in sizes.items() if area<32**2};found=set(matched['matched_truth'])
        groups={name:{'reference':0,'found':0} for name in ('below16','16to32','32to64','atleast64')}
        for i,area in sizes.items():
            group='below16' if area<16**2 else '16to32' if area<32**2 else '32to64' if area<64**2 else 'atleast64'
            groups[group]['reference']+=1;groups[group]['found']+=int(i in found)
        items[key]={**e,'reference':truth,'boxes':boxes,'counts':counts[0],'repeat_counts':counts,'timings':times,'original_size':shape,'small_reference':len(small),'small_matched':len(small&found),'size_groups':groups,'max_candidates_before_merge':max(caps)}
        rss=max(rss,psutil.Process().memory_info().rss)
        if number%8==0:print(f'{args.model}: {number}/{len(entries)}',flush=True)
    def time_summary(times):return {k:{'median':statistics.median(t[k] for t in times),'p95':baseline.percentile([t[k] for t in times],.95)} for k in times[0]}
    source_reports={}
    for source in ('sherbrooke','roe'):
        rows=[r for r in items.values() if r['source']==source]
        if not rows:continue
        expected_ref=sum(len(r['reference']) for r in rows);counts={k:sum(r['counts'][k] for r in rows) for k in ('tp','fp','fn')}
        assert counts['tp']+counts['fn']==expected_ref
        quality={'images':len(rows),'reference_boxes':expected_ref,**counts,'precision':counts['tp']/(counts['tp']+counts['fp']) if counts['tp']+counts['fp'] else None,'recall':counts['tp']/expected_ref,'small_reference':sum(r['small_reference'] for r in rows),'small_matched':sum(r['small_matched'] for r in rows),'background_images':sum(not r['reference'] for r in rows),'false_boxes_on_background':sum(r['counts']['fp'] for r in rows if not r['reference']),'size_groups':{group:{key:sum(r['size_groups'][group][key] for r in rows) for key in ('reference','found')} for group in rows[0]['size_groups']}}
        source_reports[source]={'quality':quality,'timing_summary':time_summary([t for r in rows for t in r['timings']])}
    if args.model=='before' and args.split=='control' and args.imgsz==1024:
        expected=training.get('expected_base_quality',{'sherbrooke':{'tp':83,'fp':3,'fn':32},'roe':{'tp':0,'fp':0,'fn':44}})
        for source,counts in expected.items():assert {k:source_reports[source]['quality'][k] for k in ('tp','fp','fn')}==counts,'Base control was not reproduced'
    if args.model=='previous':
        assert {k:source_reports['sherbrooke']['quality'][k] for k in ('tp','fp','fn')}=={'tp':94,'fp':11,'fn':21},'Previous mixed deer control changed'
        assert {k:source_reports['roe']['quality'][k] for k in ('tp','fp','fn')}=={'tp':2,'fp':3,'fn':42},'Previous mixed roe control changed'
    for e in entries:assert sha(e['image_path'])==e['image_sha256'] and sha(e['label_path'])==e['label_sha256']
    assert sha(weights)==weight_sha
    driver=subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,memory.total,power.draw,temperature.gpu','--format=csv,noheader'],capture_output=True,text=True,timeout=10).stdout.strip()
    conditions={'gpu':torch.cuda.get_device_name(0),'driver_snapshot':driver,'cpu':platform.processor(),'os':platform.platform(),'ram_total_bytes':psutil.virtual_memory().total,'python':platform.python_version(),'torch':torch.__version__,'torchvision':torchvision.__version__,'ultralytics':ultralytics.__version__,'cuda':torch.version.cuda,'cpu_threads':4,'opencv_threads':cv2.getNumThreads(),'input_shape':[1,3,args.imgsz,args.imgsz],'batch_size':1,'precision':'FP32; AMP and TF32 disabled','actual_model_dtype':str(next(model.model.parameters()).dtype),'confidence_threshold':.25,'nms_iou':.7,'global_merge_iou':.5,'match_iou':.5,'class_agnostic_nms':True,'network_calls_per_frame':1,'repeats':3,'warmup_frames':5,'model_order':'before and after measured in separate processes, sequentially; clocks and power state not fixed','energy':'not measured; driver power snapshot is not energy per frame'}
    limitations=['Development controls reused across experiments; not a representative or independent final test.','Visual grouping is not verified flight separation; undetected related copies may remain.','Counts describe annotated animal appearances in images, not independent animal observation events.','Cannot validate video event recall 90%, false events per minute, tracking or mapping.','Full latency includes file read/decode, preprocessing, network, internal NMS, CPU box transfer/mapping and extra duplicate filtering.','Model loading, cold first call, warmup, label reading/scoring, report output and browser excluded from warm latency. Repeated reads may use OS cache.','CPU stage timings and GPU internal timings do not exhaustively partition total latency.','Per-frame energy and future board performance unmeasured; laptop latency does not predict a 1 TOPS device.']
    report={'created_utc':datetime.now(timezone.utc).isoformat(),'experiment':args.experiment,'stage':args.model,'evaluation_split':args.split,'small_reference_input':1024,'dataset_revision':index['revision'],'model_path':str(weights.relative_to(ROOT)),'model_sha256':weight_sha,'model_file_bytes':weights.stat().st_size,'parameters':sum(p.numel() for p in model.model.parameters()),'conditions':conditions,'memory':{'gpu_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'gpu_peak_reserved_bytes':torch.cuda.max_memory_reserved(),'process_rss_bytes':rss},'cold_first_frame':cold,'first_prediction_call_ms':first_call,'timing_summary':time_summary(all_times),'sources':source_reports,'quality_repeat_counts_stable':True,'limitations':limitations,'items':items}
    filename=('fresh_' if args.split=='fresh' else '')+args.model+('_'+str(args.imgsz) if args.imgsz!=1024 else '')+'.json'
    waid.save_json(EVAL/filename,report)
    print(json.dumps({'stage':args.model,'sources':source_reports,'memory':report['memory']}),flush=True)

if __name__=='__main__':main()
