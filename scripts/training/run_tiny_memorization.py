"""Five-image fixed-input memorization diagnostic, never a generalization/deployment score."""
from project_paths import PROJECT_ROOT
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,time
import waid,baseline
from tiled_inference import map_box,merge_boxes
ROOT=PROJECT_ROOT;OUT=ROOT/'data/tiny_memorization';SHIFT_TO_GRID=False
for key,folder in [('YOLO_CONFIG_DIR','ultralytics-config'),('MPLCONFIGDIR','matplotlib-cache'),('TORCH_HOME','torch-cache')]:os.environ[key]=str(ROOT/'data'/folder)
os.environ['YOLO_AUTOINSTALL']='false'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    import cv2,numpy as np,psutil,torch,ultralytics
    from ultralytics import YOLO,settings
    from ultralytics.data.dataset import YOLODataset
    from ultralytics.data.augment import LetterBox
    from ultralytics.models.yolo.detect import DetectionTrainer
    from ultralytics.utils.tal import make_anchors,TaskAlignedAssigner
    from ultralytics.utils.ops import xywh2xyxy
    assert ROOT.drive.lower()=='d:';OUT.mkdir(exist_ok=True)
    if (OUT/'latest_run.json').exists():raise RuntimeError('Diagnostic already completed; retain it and choose another experiment directory')
    settings.update({'sync':False});torch.set_num_threads(4);cv2.setNumThreads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    source=ROOT/'data/roe_expanded';index=json.loads((source/'index.json').read_text(encoding='utf-8'));run=json.loads((source/'latest_run.json').read_text(encoding='utf-8'));byname={e['name']:e for e in index['roe_training']}
    names=['03650_1.jpg','07152_1.jpg','15276_1.jpg','19700_1.jpg','00128_1.jpg'];assert len(set(names))==5
    frozen={str(source/name):sha(source/name) for name in ('index.json','latest_run.json','experiment_plan.json','evaluation/published.json','evaluation/after.json','evaluation/fresh_after.json')};frozen[run['weights']]=sha(run['weights']);assert frozen[run['weights']]==run['weights_sha256']
    for e in index['validation']+index['fresh_validation']:
        frozen[e['image_path']]=e['image_sha256'];frozen[e['label_path']]=e['label_sha256']
    waid.save_json(OUT/'protected_artifacts.json',frozen)
    base=Path(run['weights']);probe=YOLO(str(base));strides=probe.model.stride.cpu();del probe
    anchors,scale=make_anchors([torch.zeros(1,1,1024//int(s),1024//int(s)) for s in strides],strides);anchors*=scale
    original_groups={}
    if SHIFT_TO_GRID:
        original=json.loads((ROOT/'data/tiny_memorization/input_verification.json').read_text(encoding='utf-8'));original_groups={r['name']:r['eligible_anchor_centers_per_reference'] for r in original['samples']}
    entries=[]
    for name in names:
        e=byname[name];assert sha(e['image_path'])==e['image_sha256'] and sha(e['label_path'])==e['label_sha256']
        image=cv2.imdecode(np.fromfile(e['image_path'],dtype=np.uint8),cv2.IMREAD_COLOR);assert image.shape[:2]==(1280,1280)
        resized=LetterBox(new_shape=(1024,1024),auto=False,scaleup=True,stride=32)(image=image)
        assert resized.shape==(1024,1024,3)
        dx=dy=0;source_label=Path(e['label_path']).read_text(encoding='utf-8');label_lines=source_label.splitlines()
        if SHIFT_TO_GRID:
            boxes=torch.tensor([[float(v) for v in line.split()[1:]] for line in label_lines],dtype=torch.float32);corners=xywh2xyxy(boxes)*1024
            choices=[]
            for tx in range(-4,5):
                for ty in range(-4,5):
                    shifted=corners+torch.tensor([tx,ty,tx,ty],dtype=torch.float32)
                    if shifted[:,:2].min()<0 or shifted[:,2:].max()>1024:continue
                    counts=TaskAlignedAssigner.select_candidates_in_gts(anchors,shifted.unsqueeze(0))[0].sum(1)
                    choices.append((-(counts>0).sum().item(),abs(tx)+abs(ty),abs(tx),abs(ty),tx,ty))
            chosen=min(choices);dx,dy=chosen[-2:];assert -chosen[0]==len(boxes),'Could not expose all targets with a small translation'
            shifted_image=np.full_like(resized,114);sx=max(0,-dx);sy=max(0,-dy);ex=min(1024,1024-dx);ey=min(1024,1024-dy);shifted_image[sy+dy:ey+dy,sx+dx:ex+dx]=resized[sy:ey,sx:ex]
            assert np.array_equal(shifted_image[sy+dy:ey+dy,sx+dx:ex+dx],resized[sy:ey,sx:ex]);resized=shifted_image
            lines=[]
            for line in label_lines:
                values=line.split();cx,cy,w,h=map(float,values[1:]);lines.append(f'0 {cx+dx/1024:.8f} {cy+dy/1024:.8f} {w:.8f} {h:.8f}')
            source_label='\n'.join(lines)+'\n'
        path=OUT/'images'/Path(name).with_suffix('.png');path.parent.mkdir(exist_ok=True);ok,payload=cv2.imencode('.png',resized);assert ok
        path.write_bytes(payload.tobytes());assert np.array_equal(cv2.imdecode(np.fromfile(path,dtype=np.uint8),cv2.IMREAD_COLOR),resized)
        label=OUT/'labels'/Path(name).with_suffix('.txt');label.parent.mkdir(exist_ok=True);label.write_text(source_label,encoding='utf-8')
        entries.append({'name':path.name,'source_name':name,'source':'roe','role':'large_control' if name=='00128_1.jpg' else 'tiny','image_path':str(path),'label_path':str(label),'image_sha256':sha(path),'label_sha256':sha(label),'source_image_sha256':e['image_sha256'],'fixed_integer_shift':[dx,dy],'reference_boxes':e['reference_boxes'],'dimensions':[1024,1024]})
    assert sum(e['reference_boxes'] for e in entries if e['role']=='tiny')==8
    (OUT/'train.txt').write_text(''.join(Path(e['image_path']).as_posix()+'\n' for e in entries),encoding='utf-8')
    config=OUT/'dataset.yaml';config.write_text(f'path: {ROOT.as_posix()}\ntrain: {(OUT/"train.txt").as_posix()}\nval: {(OUT/"train.txt").as_posix()}\nnames:\n  0: animal\n',encoding='utf-8')
    class FixedInputTrainer(DetectionTrainer):
        def build_dataset(self,img_path,mode='train',batch=None):
            return YOLODataset(img_path=img_path,data=self.data,imgsz=self.args.imgsz,augment=False,rect=False,cache=False,batch_size=batch,stride=int(self.model.stride.max()),hyp=self.args,pad=0,task='detect')
    eligibility={};expected_by_path={str(Path(e['image_path']).resolve()):e for e in entries}
    def verify_dataset(dataset):
        checked=[]
        for i,path in enumerate(dataset.im_files):
            e=expected_by_path[str(Path(path).resolve())];a=dataset[i];b=dataset[i]
            assert torch.equal(a['img'],b['img']) and torch.equal(a['bboxes'],b['bboxes']) and list(a['img'].shape)==[3,1024,1024]
            decoded=cv2.imdecode(np.fromfile(path,dtype=np.uint8),cv2.IMREAD_COLOR);expected=torch.from_numpy(decoded[:,:,::-1].transpose(2,0,1).copy());assert torch.equal(a['img'],expected)
            assert len(a['cls'])==e['reference_boxes'] and (a['cls']==0).all()
            boxes=xywh2xyxy(a['bboxes'])*1024;mask=TaskAlignedAssigner.select_candidates_in_gts(anchors,boxes.unsqueeze(0))[0];counts=mask.sum(1).to(torch.int64).tolist();eligibility[e['name']]=counts
            checked.append({'name':e['name'],'role':e['role'],'reference_boxes':e['reference_boxes'],'eligible_anchor_centers_per_reference':counts,'input_sha256':hashlib.sha256(a['img'].numpy().tobytes()).hexdigest(),'transforms':[type(t).__name__ for t in dataset.transforms.transforms]})
        expected_zero=0 if SHIFT_TO_GRID else 5
        assert sum(c==0 for e in entries if e['role']=='tiny' for c in eligibility[e['name']])==expected_zero
        waid.save_json(OUT/'input_verification.json',{'passed':True,'samples':checked,'native_model_strides':strides.tolist(),'native_assigner':'TaskAlignedAssigner.select_candidates_in_gts','square_pixels_match_standard_inference_letterbox':not SHIFT_TO_GRID,'fixed_integer_translation':SHIFT_TO_GRID,'repeat_inputs_identical':True,'zero_candidate_tiny_references':expected_zero,'eligible_tiny_references':8-expected_zero,'important_limit':'Geometric candidates are necessary, not sufficient, for actual positive assignment. Counts apply to this fixed no-augmentation input, not all positions in earlier augmented training.'})
    ds=YOLODataset(img_path=str(OUT/'train.txt'),data={'names':{0:'animal'},'channels':3},imgsz=1024,augment=False,rect=False,cache=False,batch_size=5,pad=0);verify_dataset(ds);del ds
    revision=hashlib.sha256(json.dumps([(e['name'],e['image_sha256'],e['label_sha256']) for e in entries]).encode()).hexdigest()
    waid.save_json(OUT/'index.json',{'revision':revision,'entries':entries,'source_dataset_revision':index['revision'],'scope':'Only training data selected before diagnostic predictions; four tiny-target images plus one large control. PNGs are lossless fixed1024 preprocessing caches.','training_and_evaluation_same_images':True})
    options=dict(run['training_options']);options.update(data=str(config),epochs=200,patience=201,batch=5,nbs=5,lr0=.001,lrf=1.,momentum=.9,weight_decay=0.,warmup_epochs=0.,warmup_bias_lr=.001,save=False,save_period=-1,val=False,project=str(OUT/'training'),name=datetime.now(timezone.utc).strftime('tiny_%Y%m%dT%H%M%SZ'),hsv_h=0.,hsv_s=0.,hsv_v=0.,scale=0.,translate=0.,fliplr=0.,flipud=0.,degrees=0.,shear=0.,perspective=0.,mosaic=0.,mixup=0.,cutmix=0.,copy_paste=0.,bgr=0.,workers=0,cache=False,rect=False,amp=False,plots=False)
    plan={'created_utc':datetime.now(timezone.utc).isoformat(),'dataset_revision':revision,'base_weights':str(base),'base_weights_sha256':sha(base),'fixed_integer_translation':SHIFT_TO_GRID,'group_cohorts_defined_by_original_unshifted_input':True,'images':5,'tiny_reference_boxes':8,'large_reference_boxes':1,'epochs':200,'batches_per_epoch':1,'expected_optimizer_steps':200,'training_options':options,'augmentation_policy':'Custom trainer uses only LetterBox and Format. Source images are cached losslessly at the exact standard-inference pixel input.','checkpoint_policy':'Fixed last epoch200; not a best-epoch selection.','evaluation_policy':'Same five TRAINING images, conf0.25, IoU0.5, batch1 FP32, same extra IoU0.5 duplicate filter. Scores do not estimate generalization.','hypothesis':'Some tiny references have no eligible anchor centers in fixed inputs; inspect learning separately for zero-candidate and candidate-supported targets.'}
    waid.save_json(OUT/'experiment_plan.json',plan)
    def evaluate(weights,stage):
        model=YOLO(str(weights));first=cv2.imdecode(np.fromfile(entries[0]['image_path'],dtype=np.uint8),cv2.IMREAD_COLOR)
        cfg=dict(imgsz=1024,rect=False,device=0,half=False,batch=1,conf=.25,iou=.7,agnostic_nms=True,classes=[0],max_det=300,verbose=False,save=False)
        model.predict(source=first,**cfg)
        for _ in range(5):model.predict(source=first,**cfg)
        items={};groups={key:{'references':0,'matched':0} for key in ('tiny_zero_candidates','tiny_with_candidates','large_control')};all_times=[]
        torch.cuda.reset_peak_memory_stats()
        for e in entries:
            truth=waid.parse_labels(Path(e['label_path']),['animal'])['boxes'];trials=[];timings=[];outputs=[]
            for repeat in range(3):
                torch.cuda.synchronize();start=time.perf_counter();image=cv2.imdecode(np.fromfile(e['image_path'],dtype=np.uint8),cv2.IMREAD_COLOR);r=model.predict(source=image,**cfg)[0]
                boxes=[map_box(row,(0,0,1024,1024),(1024,1024),model.names) for row in r.boxes.data.detach().cpu().tolist()];boxes=merge_boxes([b for b in boxes if b],.5,300);torch.cuda.synchronize();timings.append((time.perf_counter()-start)*1000);outputs.append(boxes)
                m=baseline.match_boxes(boxes,truth,.5);trials.append({k:m[k] for k in ('tp','fp','fn')})
            assert all(c==trials[0] for c in trials);match=baseline.match_boxes(outputs[0],truth,.5);found=set(match['matched_truth'])
            for i in range(len(truth)):
                key='large_control' if e['role']=='large_control' else 'tiny_zero_candidates' if (original_groups[e['name']][i] if SHIFT_TO_GRID else eligibility[e['name']][i])==0 else 'tiny_with_candidates';groups[key]['references']+=1;groups[key]['matched']+=int(i in found)
            items[e['name']]={**e,'reference':truth,'boxes':outputs[0],'counts':trials[0],'timings_ms':timings,'matched_reference_indices':match['matched_truth'],'eligible_anchor_centers':eligibility[e['name']]};all_times+=timings
        totals={k:sum(row['counts'][k] for row in items.values()) for k in ('tp','fp','fn')}
        report={'stage':stage,'dataset_revision':revision,'model_path':str(weights),'model_sha256':sha(weights),'groups':groups,'grouping_policy':'Original unshifted eligibility cohorts' if SHIFT_TO_GRID else 'Current unshifted eligibility cohorts','counts':totals,'conditions':{'input_shape':[1,3,1024,1024],'precision':'FP32, TF32 off','batch_size':1,'confidence_threshold':.25,'match_iou':.5,'extra_merge_iou':.5,'gpu':torch.cuda.get_device_name(0),'cpu_threads':4,'repeats':3},'timing_summary':{'total_ms':{'median':__import__('statistics').median(all_times),'p95':baseline.percentile(all_times,.95)}},'memory':{'gpu_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'process_rss_bytes':psutil.Process().memory_info().rss},'items':items,'scope':'Memorization on the same five training images only; not validation quality. Latency includes read/decode, preparation, network and box processing; model load/warmup/reference scoring excluded.'}
        waid.save_json(OUT/(stage+'.json'),report);del model;torch.cuda.empty_cache();return report
    before=evaluate(base,'before');started=time.perf_counter();model=YOLO(str(base));torch.cuda.reset_peak_memory_stats()
    def on_start(trainer):verify_dataset(trainer.train_loader.dataset)
    def progress(trainer):
        n=int(trainer.epoch)+1
        if n%10==0 or n==1:waid.save_json(OUT/'status.json',{'state':'training','epoch_completed':n,'epochs':200,'elapsed_seconds':time.perf_counter()-started,'loss':trainer.tloss.detach().cpu().tolist()})
    model.add_callback('on_train_start',on_start);model.add_callback('on_fit_epoch_end',progress);waid.save_json(OUT/'status.json',{'state':'starting','epochs':200})
    model.train(trainer=FixedInputTrainer,**options);last=Path(model.trainer.save_dir)/'weights/last.pt';assert last.exists();train_seconds=time.perf_counter()-started;peak=torch.cuda.max_memory_allocated();waid.save_json(OUT/'latest_run.json',{'weights':str(last),'weights_sha256':sha(last),'base_weights':str(base),'base_weights_sha256':sha(base),'dataset_revision':revision,'epochs':200,'optimizer_steps':int(model.trainer.optimizer_step_count) if hasattr(model.trainer,'optimizer_step_count') else None,'training_seconds':train_seconds,'gpu_peak_allocated_bytes':peak,'scope':'Diagnostic only; keep production candidates unchanged.'});del model;torch.cuda.empty_cache()
    after=evaluate(last,'after');assert all(sha(path)==value for path,value in frozen.items())
    comparison={'dataset_revision':revision,'before':{k:v for k,v in before.items() if k!='items'},'after':{k:v for k,v in after.items() if k!='items'},'training_seconds':train_seconds,'previous_artifacts_and_controls_unchanged':True,'interpretation_limit':'Fixed-input training memorization; no held-out inference claims. Zero geometric candidates blocks direct positive assignment in this specific setup; does not prove a target can never be detected or never had positive assignments in earlier augmented training.'}
    waid.save_json(OUT/'comparison.json',comparison);waid.save_json(OUT/'status.json',{'state':'completed','epochs':200,'training_seconds':train_seconds,'before':before['counts'],'after':after['counts'],'groups':after['groups']});print(json.dumps(comparison),flush=True)

if __name__=='__main__':
    try:main()
    except Exception as error:
        waid.save_json(OUT/'status.json',{'state':'failed','error':str(error)});raise
