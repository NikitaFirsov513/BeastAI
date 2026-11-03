"""Frozen full-frame comparison of compact YOLO and DroneMegaDetector, no training."""
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import gc,hashlib,json,os,platform,statistics,subprocess,time
import waid,baseline
from tiled_inference import map_box,merge_boxes
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'data/dronemegadetector'
for k,v in [('HF_HOME',str(OUT/'hf-cache')),('TORCH_HOME',str(OUT/'torch-cache')),('MPLCONFIGDIR',str(OUT/'matplotlib-cache')),('YOLO_CONFIG_DIR',str(ROOT/'data/ultralytics-config')),('YOLO_AUTOINSTALL','false'),('HF_HUB_OFFLINE','1'),('TRANSFORMERS_OFFLINE','1')]:os.environ[k]=v

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def map_animal_rows(xyxy,scores,classes,width,height):
    result=[]
    for coordinates,score,category in zip(xyxy,scores,classes):
        if int(category)!=1:continue
        b=map_box([*map(float,coordinates),float(score),0],(0,0,width,height),(width,height),{0:'animal'})
        if b:b['native_class_id']=1;result.append(b)
    return result

def main():
    import cv2,numpy as np,torch,torchvision,psutil,rfdetr,importlib.metadata
    from ultralytics import YOLO,settings
    assert ROOT.drive.lower()=='d:' and torch.cuda.is_available()
    target=OUT/'evaluation/comparison.json'
    if target.exists():raise RuntimeError('Retain completed reference comparison')
    settings.update({'sync':False});torch.set_num_threads(4);cv2.setNumThreads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    index=json.loads((OUT/'index.json').read_text(encoding='utf-8'));plan=json.loads((OUT/'experiment_plan.json').read_text(encoding='utf-8'));probe=json.loads((OUT/'runtime_probe.json').read_text(encoding='utf-8'));assert probe['passed']
    assert index['revision']==plan['dataset_revision'];entries=[{**e,'evaluation_split':split} for split,rows in [('control',index['validation']),('fresh',index['fresh_validation'])] for e in rows];assert len(entries)==80
    references={}
    for e in entries:
        for k,h in [('image_path','image_sha256'),('label_path','label_sha256')]:assert sha(e[k])==e[h]
        parsed=waid.parse_labels(Path(e['label_path']),['animal']);assert not parsed['errors'];assert len(parsed['boxes'])==e['reference_boxes'];references[e['source']+'/'+e['name']]=parsed['boxes']
    for k,c in plan['models'].items():assert sha(c['path'])==c['sha256']
    class Runner:
        def __init__(self,key):
            self.key=key;self.cfg=plan['models'][key];self.load_ms=0.;self.cold=None;start=time.perf_counter()
            if key=='baseline':self.model=YOLO(self.cfg['path'])
            else:
                self.model=rfdetr.from_checkpoint(self.cfg['path'],device='cuda',amp=False)
                assert self.model.class_names==['person','animal','vehicle'];assert self.model.model_config.resolution==1280;self.model.model.model.float().eval()
            self.load_ms=(time.perf_counter()-start)*1000
        def module(self):return self.model.model if self.key=='baseline' else self.model.model.model
        def predict(self,path):
            # This function accepts no annotations, flight IDs or target positions.
            torch.cuda.synchronize();start=time.perf_counter();image=cv2.imdecode(np.fromfile(path,np.uint8),cv2.IMREAD_COLOR)
            if image is None:raise RuntimeError('Decode failed')
            h,w=image.shape[:2];read_ms=(time.perf_counter()-start)*1000
            conversion_start=time.perf_counter();input_image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB) if self.key=='reference' else image;conversion_ms=(time.perf_counter()-conversion_start)*1000
            api_start=time.perf_counter()
            with torch.inference_mode(),torch.autocast('cuda',enabled=False):
                if self.key=='baseline':
                    output=self.model.predict(source=input_image,imgsz=1024,rect=False,device=0,half=False,batch=1,conf=.25,iou=.7,agnostic_nms=True,classes=[0],max_det=300,verbose=False,save=False)[0]
                else:output=self.model.predict(input_image,threshold=.25,include_source_image=False)
            torch.cuda.synchronize();api_ms=(time.perf_counter()-api_start)*1000
            post_start=time.perf_counter()
            if self.key=='baseline':
                raw=output.boxes.data.detach().cpu().tolist();boxes=[map_box(r,(0,0,w,h),(w,h),{0:'animal'}) for r in raw];boxes=[b for b in boxes if b];raw_count=len(raw);other={}
            else:
                assert output.class_id is not None and output.confidence is not None
                raw_count=len(output);other=dict(Counter(str(int(c)) for c in output.class_id if int(c)!=1));boxes=map_animal_rows(output.xyxy,output.confidence,output.class_id,w,h)
            mapping_ms=(time.perf_counter()-post_start)*1000;pre_merge=len(boxes);merge_start=time.perf_counter();boxes=merge_boxes(boxes,.5,300);merge_ms=(time.perf_counter()-merge_start)*1000
            torch.cuda.synchronize();total_ms=(time.perf_counter()-start)*1000
            # RF-DETR returns top300 candidates across classes; flag clipping rather than hiding it.
            return boxes,{'total_ms':total_ms,'read_decode_ms':read_ms,'colour_conversion_ms':conversion_ms,'detector_api_ms':api_ms,'cpu_box_mapping_ms':mapping_ms,'additional_merge_ms':merge_ms},[w,h],{'raw_predictions_all_classes':raw_count,'target_candidates_before_merge':pre_merge,'non_target_predictions':other,'prediction_cap_reached':raw_count>=300 or len(boxes)>=300}
        def warmup(self):
            _,self.cold,_,_=self.predict(entries[0]['image_path'])
            for _ in range(5):self.predict(entries[0]['image_path'])
            assert next(self.module().parameters()).dtype==torch.float32
    # Isolated process allocations, after warmup, on two explicitly named representative inputs.
    isolated={};representatives=[next(e for e in entries if e['source']==s) for s in ('sherbrooke','roe')]
    for key in ('baseline','reference'):
        runner=Runner(key);runner.warmup();torch.cuda.reset_peak_memory_stats();rss=0
        for e in representatives:
            for _ in range(3):runner.predict(e['image_path']);rss=max(rss,psutil.Process().memory_info().rss)
        isolated[key]={'gpu_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'gpu_peak_reserved_bytes':torch.cuda.max_memory_reserved(),'process_rss_bytes':rss,'parameter_elements':sum(p.numel() for p in runner.module().parameters()),'actual_model_dtype':str(next(runner.module().parameters()).dtype),'probe_images':[e['source']+'/'+e['name'] for e in representatives],'scope':'Only this model loaded. Two representative input shapes, three repeats each after warmup; not a maximum over all80 images. CUDA context/driver not in PyTorch allocations.'}
        del runner;gc.collect();torch.cuda.empty_cache();waid.save_json(OUT/'status.json',{'state':'isolated_memory_checked','model':key})
    runners={};loads={};cold={}
    for key in ('baseline','reference'):
        runners[key]=Runner(key);runners[key].warmup();loads[key]=runners[key].load_ms;cold[key]=runners[key].cold
    driver_args=['nvidia-smi','--query-gpu=name,driver_version,memory.total,power.draw,temperature.gpu,clocks.sm','--format=csv,noheader']
    driver_start=subprocess.run(driver_args,capture_output=True,text=True,timeout=15).stdout.strip();items={k:{} for k in runners};aggregate_peak=0;aggregate_reserved=0;aggregate_rss=0
    for number,e in enumerate(entries):
        name=e['source']+'/'+e['name'];truth=references[name];results={k:[] for k in runners}
        for repeat in range(3):
            order=('baseline','reference') if (number+repeat)%2==0 else ('reference','baseline')
            for key in order:
                torch.cuda.reset_peak_memory_stats();boxes,timing,shape,extra=runners[key].predict(e['image_path']);matched=baseline.match_boxes(boxes,truth,.5)
                aggregate_peak=max(aggregate_peak,torch.cuda.max_memory_allocated());aggregate_reserved=max(aggregate_reserved,torch.cuda.max_memory_reserved());aggregate_rss=max(aggregate_rss,psutil.Process().memory_info().rss)
                assert not extra['prediction_cap_reached'],'Prediction cap reached; quality may be censored'
                results[key].append({'boxes':boxes,'timing':timing,'shape':shape,'extra':extra,'counts':{k:matched[k] for k in ('tp','fp','fn')},'found':matched['matched_truth']})
        for key,observations in results.items():
            r=observations[0];assert all(o['counts']==r['counts'] for o in observations),'Repeat quality changed'
            w,h=r['shape'];scale=min(1024/w,1024/h);found=set(r['found']);small=set();groups={g:{'reference':0,'found':0} for g in ('below16','16to32','32to64','atleast64')}
            for i,b in enumerate(truth):
                area=b['width']*w*b['height']*h*scale**2;g='below16' if area<16**2 else '16to32' if area<32**2 else '32to64' if area<64**2 else 'atleast64';groups[g]['reference']+=1;groups[g]['found']+=int(i in found)
                if area<32**2:small.add(i)
            items[key][name]={**e,'reference':truth,'boxes':r['boxes'],'counts':r['counts'],'repeat_counts':[o['counts'] for o in observations],'timings':[o['timing'] for o in observations],'original_size':r['shape'],'small_reference':len(small),'small_matched':len(small&found),'size_groups':groups,'matched_reference_indices':r['found'],'prediction_details':r['extra']}
        if (number+1)%4==0:
            waid.save_json(OUT/'status.json',{'state':'evaluating','images_completed':number+1,'images':80,'repeats':3});print(f'Compared {number+1}/80 both models',flush=True)
    def timing_summary(times):return {k:{'median':statistics.median(t[k] for t in times),'p95':baseline.percentile([t[k] for t in times],.95)} for k in times[0]}
    models={}
    for key,rows_dict in items.items():
        partitions={}
        for split in ('control','fresh'):
            sources={}
            for source in ('sherbrooke','roe'):
                rows=[r for r in rows_dict.values() if r['evaluation_split']==split and r['source']==source]
                if not rows:continue
                counts={k:sum(r['counts'][k] for r in rows) for k in ('tp','fp','fn')};nref=sum(len(r['reference']) for r in rows)
                q={'images':len(rows),'reference_boxes':nref,**counts,'precision':counts['tp']/(counts['tp']+counts['fp']) if counts['tp']+counts['fp'] else None,'recall':counts['tp']/nref,'small_reference':sum(r['small_reference'] for r in rows),'small_matched':sum(r['small_matched'] for r in rows),'background_images':sum(not r['reference'] for r in rows),'false_boxes_on_background':sum(r['counts']['fp'] for r in rows if not r['reference']),'size_groups':{g:{k:sum(r['size_groups'][g][k] for r in rows) for k in ('reference','found')} for g in rows[0]['size_groups']}}
                assert counts['tp']+counts['fn']==nref;sources[source]={'quality':q,'timing_summary':timing_summary([t for r in rows for t in r['timings']])}
            partitions[split]=sources
        cfg=plan['models'][key];models[key]={'model_path':cfg['path'],'model_sha256':sha(cfg['path']),'model_file_bytes':Path(cfg['path']).stat().st_size,'input_shape':[1,3,cfg['input'],cfg['input']],'model_label':'YOLO11n · основной кандидат' if key=='baseline' else 'DroneMegaDetector · RF-DETR Medium','isolated_memory_probe':isolated[key],'load_ms':loads[key],'cold_first_frame':cold[key],'partitions':partitions,'timing_summary':timing_summary([t for r in rows_dict.values() for t in r['timings']]),'items':rows_dict}
    for split,filename in [('control','after.json'),('fresh','fresh_after.json')]:
        original=json.loads((ROOT/'data/roe_expanded/evaluation'/filename).read_text(encoding='utf-8'))
        for source,r in original['sources'].items():assert {k:models['baseline']['partitions'][split][source]['quality'][k] for k in ('tp','fp','fn')}=={k:r['quality'][k] for k in ('tp','fp','fn')},'Frozen baseline not reproduced'
    assert all(sha(p)==h for p,h in json.loads((OUT/'protected_artifacts.json').read_text(encoding='utf-8')).items())
    result={'created_utc':datetime.now(timezone.utc).isoformat(),'dataset_revision':index['revision'],'training':False,'models':models,'conditions':{'gpu':torch.cuda.get_device_name(0),'driver_snapshot_start':driver_start,'driver_snapshot_end':subprocess.run(driver_args,capture_output=True,text=True,timeout=15).stdout.strip(),'cpu':platform.processor(),'os':platform.platform(),'ram_total_bytes':psutil.virtual_memory().total,'python':platform.python_version(),'torch':torch.__version__,'torchvision':torchvision.__version__,'rfdetr':importlib.metadata.version('rfdetr'),'transformers':importlib.metadata.version('transformers'),'ultralytics':importlib.metadata.version('ultralytics'),'numpy':np.__version__,'opencv':cv2.__version__,'cuda':torch.version.cuda,'batch_size':1,'precision':'FP32; AMP and TF32 disabled','cpu_threads':4,'opencv_threads':cv2.getNumThreads(),'confidence_threshold':.25,'baseline_internal_nms_iou':.7,'reference_internal_nms':'Native RF-DETR set prediction, no added internal NMS','additional_merge_iou':.5,'match_iou':.5,'network_calls_per_frame':1,'repeats':3,'warmup_per_model':5,'fixed_small_reference_scale':1024,'timing_order':'Both models resident. Model order alternates per frame and repeat. No concurrent training.','isolated_memory_probe':True,'power_energy':'Not measured; instantaneous driver snapshots do not measure energy per frame.'},'co_resident_benchmark_memory':{'gpu_peak_allocated_bytes':aggregate_peak,'gpu_peak_reserved_bytes':aggregate_reserved,'process_rss_bytes':aggregate_rss,'scope':'Both models loaded; this aggregate maximum over all80 cannot be attributed to either model alone.'},'limitations':['Model pretraining overlap with our80 controls not ruled out; not an independent external-generalization claim.','Native inputs and preprocessing differ (YOLO1024 letterbox, reference1280 square resize). This compares complete configurations, not architecture at equal compute.','Fixed0.25 scores are not calibrated probabilities shared between models. No threshold selection on controls.','Roe visual groups are not verified flight separation; reused development controls.','Full wall latency includes read/decode, colour conversion, detector preparation/network/postprocessing, host box mapping and extra duplicate filtering.','Cold start, loading, warmup, reference reading/scoring, JSON output and browser excluded. Repeated files may be cached.','Time modes interleave but clocks/power/OS load not locked. Isolated memory probes cover two shapes; aggregate timed peak with both models also recorded.','Image box appearances do not establish video90% event recall, false events per minute or future1TOPS performance.']}
    waid.save_json(target,result);waid.save_json(OUT/'status.json',{'state':'evaluated','images':80,'training':False});print(json.dumps({key:r['partitions'] for key,r in models.items()}),flush=True)
if __name__=='__main__':
    try:main()
    except Exception as error:waid.save_json(OUT/'status.json',{'state':'failed','error':str(error)});raise
