"""Prepare reserved new diagnostic groups and fine-tune on the expanded data, fixed20 epochs."""
from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone
import hashlib,json,os,random,time
import waid
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'data/roe_expanded'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    assert ROOT.drive.lower()=='d:';OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'latest_run.json').exists():raise RuntimeError('This expanded experiment already completed')
    waid.save_json(OUT/'training_status.json',{'state':'waiting_for_prepared_data','planned_epochs':20})
    while not (OUT/'index.json').exists():
        status=json.loads((OUT/'preparation_status.json').read_text(encoding='utf-8')) if (OUT/'preparation_status.json').exists() else {}
        if status.get('state')=='failed':raise RuntimeError('Data preparation failed: '+status.get('error',''))
        time.sleep(3)
    grouped=json.loads((OUT/'index.json').read_text(encoding='utf-8'));assert len(grouped['validation'])==64
    if (OUT/'grouped_index.json').exists():grouped=json.loads((OUT/'grouped_index.json').read_text(encoding='utf-8'))
    else:waid.save_json(OUT/'grouped_index.json',grouped)
    roe=grouped['roe_training'];new=[e for e in roe if e['id'].startswith('roe_new/')]
    hold_count=24 if len(new)>=72 else 16 if len(new)>=48 else 8 if len(new)>=24 else 0
    hold_count=min(hold_count,max(0,len(roe)-28))
    rng=random.Random(20261104);bins=defaultdict(list)
    for e in new:
        truth=waid.parse_labels(Path(e['label_path']),['animal'])['boxes'];area=max(b['width']*b['height']*1024**2 for b in truth)
        group=0 if area<16**2 else 1 if area<32**2 else 2 if area<64**2 else 3;bins[group].append(e)
    for values in bins.values():rng.shuffle(values)
    fresh=[]
    while len(fresh)<hold_count:
        for group in range(4):
            if len(fresh)<hold_count and bins[group]:fresh.append(bins[group].pop())
    fresh_ids={e['id'] for e in fresh};training_roe=[e for e in roe if e['id'] not in fresh_ids];deer=[e for e in grouped['training'] if e['source']=='sherbrooke']
    assert len(deer)==512 and len(training_roe)>27,'Expanded independent-component training pool is not larger than the previous27'
    validation=grouped['validation'];training=deer+training_roe
    assert {e['visual_group'] for e in training_roe}.isdisjoint(e['visual_group'] for e in fresh)
    assert {e['image_path'] for e in training}.isdisjoint(e['image_path'] for e in validation+fresh)
    target=max(512,len(training_roe))
    def repeated(entries,count):
        result=list(entries)*(count//len(entries))+rng.sample(list(entries),count%len(entries));rng.shuffle(result);return result
    exposure=repeated(deer,target)+repeated(training_roe,target)
    revision=hashlib.sha256(json.dumps({'grouped_revision':grouped['revision'],'train':[(e['image_path'],e['image_sha256'],e['label_sha256']) for e in training],'fresh':[e['id'] for e in fresh],'exposure':Counter(e['image_path'] for e in exposure)},sort_keys=True).encode()).hexdigest()
    index={**grouped,'revision':revision,'grouped_revision':grouped['revision'],'training':training,'roe_training':training_roe,'fresh_validation':fresh,'reserved_fresh_policy':'Selected from new unprotected visual components before model inference, stratified by maximum reference size; not flight-verified or representative.','preparation_selected_roe_before_holdout':len(roe)}
    waid.save_json(OUT/'index.json',index)
    waid.save_json(OUT/'reserved_fresh_groups.json',{'entries':fresh,'component_ids':[e['visual_group'] for e in fresh],'policy':index['reserved_fresh_policy'],'must_not_enter_training':True})
    for name,rows in [('train_unique.txt',training),('train_exposure.txt',exposure),('valid.txt',validation),('fresh.txt',fresh)]:
        (OUT/name).write_text(''.join(Path(e['image_path']).as_posix()+'\n' for e in rows),encoding='utf-8')
    config=OUT/'dataset.yaml';config.write_text(f'path: {ROOT.as_posix()}\ntrain: {(OUT/"train_exposure.txt").as_posix()}\nval: {(OUT/"valid.txt").as_posix()}\nnames:\n  0: animal\n',encoding='utf-8')
    base=json.loads((ROOT/'data/roe_focus/latest_run.json').read_text(encoding='utf-8'));base_report=json.loads((ROOT/'data/roe_focus/evaluation/after.json').read_text(encoding='utf-8'))
    comparison_weights=Path(base['weights']);assert sha(comparison_weights)==base['weights_sha256']==base_report['model_sha256']
    initial=json.loads((ROOT/'data/sherbrooke/train512/latest_run.json').read_text(encoding='utf-8')) if grouped.get('legacy_selected_now_connected_to_control',0) else base
    weights=Path(initial['weights']);assert sha(weights)==initial['weights_sha256']
    preserved=[]
    for folder in ('mixed_pilot','roe_focus'):
        for name in ('index.json','experiment_plan.json','latest_run.json','evaluation/before.json','evaluation/after.json','evaluation/comparison.json','evaluation/published.json','evaluation/COMPARISON.md'):
            p=ROOT/'data'/folder/name
            if p.exists():preserved.append(p)
        run=json.loads((ROOT/'data'/folder/'latest_run.json').read_text(encoding='utf-8'));preserved.extend([Path(run['weights']),Path(run['base_weights'])])
    preserved.append(ROOT/'data/sherbrooke/baseline/latest.json');frozen={str(p):sha(p) for p in preserved};waid.save_json(OUT/'protected_artifacts.json',frozen)
    for e in training+validation+fresh:assert sha(e['image_path'])==e['image_sha256'] and sha(e['label_path'])==e['label_sha256']
    for key,folder in [('YOLO_CONFIG_DIR','ultralytics-config'),('MPLCONFIGDIR','matplotlib-cache'),('TORCH_HOME','torch-cache')]:os.environ[key]=str(ROOT/'data'/folder)
    os.environ['YOLO_AUTOINSTALL']='false'
    import torch,cv2,ultralytics
    from ultralytics import YOLO,settings
    from ultralytics.data.dataset import YOLODataset
    from ultralytics.data.utils import check_det_dataset
    settings.update({'sync':False});torch.set_num_threads(4);cv2.setNumThreads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    name=datetime.now(timezone.utc).strftime('roe_expanded_%Y%m%dT%H%M%SZ');options=dict(base['training_options']);options.update(data=str(config),epochs=20,patience=20,project=str(OUT/'training'),name=name,warmup_epochs=1.,warmup_bias_lr=.0003)
    expected={source:{k:r['quality'][k] for k in ('tp','fp','fn')} for source,r in base_report['sources'].items()}
    plan={'created_utc':datetime.now(timezone.utc).isoformat(),'dataset_revision':revision,'base_weights':str(weights),'base_weights_sha256':initial['weights_sha256'],'comparison_weights':str(comparison_weights),'comparison_weights_sha256':base['weights_sha256'],'unique_training_images':len(training),'unique_roe_images':len(training_roe),'deer_training_images':512,'entries_per_epoch':len(exposure),'source_entries':{'sherbrooke':target,'roe':target},'fresh_diagnostic_images':len(fresh),'fresh_diagnostic_boxes':sum(e['reference_boxes'] for e in fresh),'epochs':20,'options':options,'checkpoint_policy':'Fixed last epoch20, not validation-selected best.','expected_base_quality':expected,'rationale':'Use the roe-focused candidate for comparison. If expanded grouping discovers legacy control links, initialize from deer-only weights; otherwise continue from the focused candidate. Train using more unique visual-component representatives, all512 deer frames and balanced source exposure. Fixed20 epochs; short low-bias warmup for continued fine-tuning. This changes data, initial weights and training budget, so it is not a one-factor causal experiment.','limitations':'Known related copies excluded. Visual groups are not flight IDs. Old64 controls are reused development data; newly reserved diagnostic groups are also not verified flight-independent.','manifest_sha256':sha(OUT/'train_exposure.txt')}
    waid.save_json(OUT/'experiment_plan.json',plan)
    data=check_det_dataset(str(config),autodownload=False);loader=YOLODataset(img_path=data['train'],data=data,imgsz=1024,augment=False,rect=False,cache=False,batch_size=2)
    counts=Counter(str(Path(p).resolve()) for p in loader.im_files);assert counts==Counter(str(Path(e['image_path']).resolve()) for e in exposure)
    expected_labels={str(Path(e['image_path']).resolve()):e['reference_boxes'] for e in training};assert all(len(l['cls'])==expected_labels[str(Path(l['im_file']).resolve())] and (l['cls']==0).all() for l in loader.labels)
    waid.save_json(OUT/'exposure_verification.json',{'passed':True,'unique_images':len(counts),'training_entries':len(loader),'source_entries':plan['source_entries'],'total_boxes_with_repeats':sum(len(l['cls']) for l in loader.labels),'fresh_groups_disjoint_from_training':True,'old_control_images':64});del loader
    waid.save_json(OUT/'dataset_ready.json',{'prepared':True,'unique_roe_training':len(training_roe),'fresh_diagnostic_images':len(fresh),'entries_per_epoch':len(exposure)})
    started=time.perf_counter();torch.cuda.reset_peak_memory_stats();model=YOLO(str(weights));assert model.names=={0:'animal'}
    def progress(trainer):waid.save_json(OUT/'training_status.json',{'state':'training','epoch_completed':int(trainer.epoch)+1,'epochs':20,'elapsed_seconds':time.perf_counter()-started,'run_directory':str(trainer.save_dir),'metrics':{k:float(v) for k,v in trainer.metrics.items()}})
    model.add_callback('on_fit_epoch_end',progress);waid.save_json(OUT/'training_status.json',{'state':'starting','epochs':20,'unique_roe_training':len(training_roe),'training_entries':len(exposure),'fresh_diagnostic_images':len(fresh),'run_name':name})
    model.train(**options)
    run=Path(model.trainer.save_dir);last=run/'weights/last.pt';assert last.exists() and sha(last)!=initial['weights_sha256'];assert all(sha(path)==value for path,value in frozen.items())
    for e in validation+fresh:assert sha(e['image_path'])==e['image_sha256'] and sha(e['label_path'])==e['label_sha256']
    result={'created_utc':datetime.now(timezone.utc).isoformat(),'dataset_revision':revision,'weights':str(last),'weights_sha256':sha(last),'base_weights':str(weights),'base_weights_sha256':initial['weights_sha256'],'comparison_weights':str(comparison_weights),'comparison_weights_sha256':base['weights_sha256'],'expected_base_quality':expected,'checkpoint_policy':plan['checkpoint_policy'],'training_seconds':time.perf_counter()-started,'training_options':options,'unique_training_images':len(training),'unique_roe_images':len(training_roe),'source_entries':plan['source_entries'],'entries_per_epoch':len(exposure),'validation_images':64,'fresh_validation_images':len(fresh),'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,'ultralytics':ultralytics.__version__,'gpu_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'gpu_peak_reserved_bytes':torch.cuda.max_memory_reserved(),'weights_file_bytes':last.stat().st_size,'precision':'FP32, AMP and TF32 disabled; inference explicitly FP32.','previous_artifacts_and_controls_unchanged':True}
    waid.save_json(OUT/'latest_run.json',result);waid.save_json(run/'experiment_summary.json',result);waid.save_json(OUT/'training_status.json',{'state':'completed','epoch_completed':20,'epochs':20,'elapsed_seconds':result['training_seconds'],'weights':str(last)})
    print(json.dumps(result),flush=True)

if __name__=='__main__':
    try:main()
    except Exception as error:
        waid.save_json(OUT/'training_status.json',{'state':'failed','error':str(error)});raise
