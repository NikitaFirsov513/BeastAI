"""Fixed 50/50 source-exposure experiment, same 620 entries and 10 epochs."""
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import hashlib,json,os,random,time
import waid
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'data/roe_focus';PREVIOUS=ROOT/'data/mixed_pilot'
BASE_SHA='ac1229f41c338a8c3913e792857c27fa40b5b7f52a17bd2772e4d4355b6beead'

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    assert ROOT.drive.lower()=='d:';OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'latest_run.json').exists():raise RuntimeError('Completed focus experiment exists; use a new experiment directory')
    for key,folder in [('YOLO_CONFIG_DIR','ultralytics-config'),('MPLCONFIGDIR','matplotlib-cache'),('TORCH_HOME','torch-cache')]:os.environ[key]=str(ROOT/'data'/folder)
    os.environ['YOLO_AUTOINSTALL']='false'
    import torch,cv2,ultralytics
    from ultralytics import YOLO,settings
    from ultralytics.data.dataset import YOLODataset
    from ultralytics.data.utils import check_det_dataset
    settings.update({'sync':False});torch.set_num_threads(4);cv2.setNumThreads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    old_index=json.loads((PREVIOUS/'index.json').read_text(encoding='utf-8'));old_run=json.loads((PREVIOUS/'latest_run.json').read_text(encoding='utf-8'))
    weights=Path(old_run['base_weights']);assert digest(weights)==BASE_SHA
    deer=[e for e in old_index['training'] if e['source']=='sherbrooke'];roe=old_index['roe_training']
    positive=sorted([e for e in deer if e['reference_boxes']],key=lambda e:e['name']);negative=sorted([e for e in deer if not e['reference_boxes']],key=lambda e:e['name'])
    assert len(positive)==len(negative)==256 and len(roe)==27
    rng=random.Random(20261102)
    selected_deer=sorted(rng.sample(positive,155)+rng.sample(negative,155),key=lambda e:e['name'])
    extra_roe=rng.sample(sorted(roe,key=lambda e:e['name']),13)
    exposure=selected_deer+roe*11+extra_roe
    training=selected_deer+roe;validation=old_index['validation']
    assert len(training)==337 and len(exposure)==620 and len(validation)==64
    assert Counter(e['source'] for e in exposure)=={'sherbrooke':310,'roe':310}
    assert {e['image_path'] for e in training}.isdisjoint(e['image_path'] for e in validation)
    for e in training+validation:assert digest(e['image_path'])==e['image_sha256'] and digest(e['label_path'])==e['label_sha256']
    protected=[PREVIOUS/'index.json',PREVIOUS/'experiment_plan.json',PREVIOUS/'latest_run.json',PREVIOUS/'train.txt',PREVIOUS/'valid.txt',PREVIOUS/'dataset.yaml',PREVIOUS/'evaluation/before.json',PREVIOUS/'evaluation/after.json',PREVIOUS/'evaluation/comparison.json',PREVIOUS/'evaluation/published.json',PREVIOUS/'evaluation/COMPARISON.md',weights,Path(old_run['weights']),ROOT/'data/sherbrooke/baseline/latest.json']
    frozen={str(p):digest(p) for p in protected};waid.save_json(OUT/'protected_artifacts.json',frozen)
    revision=digest(PREVIOUS/'index.json')
    revision=hashlib.sha256(json.dumps({'source_index_sha256':revision,'training':[(e['image_path'],e['image_sha256'],e['label_sha256']) for e in training],'exposures':Counter(e['image_path'] for e in exposure),'validation':[(e['image_path'],e['image_sha256'],e['label_sha256']) for e in validation]},sort_keys=True).encode()).hexdigest()
    index={'revision':revision,'source_dataset_revision':old_index['revision'],'training':training,'validation':validation,'roe_training':roe,'training_exposure':exposure,'sampling_seed':20261102,'scope':'Development controls unchanged. Known related groups excluded in source preparation; flight independence unverified.'}
    waid.save_json(OUT/'index.json',index)
    for name,entries in [('train.txt',training),('train_exposure.txt',exposure),('valid.txt',validation)]:
        (OUT/name).write_text(''.join(Path(e['image_path']).as_posix()+'\n' for e in entries),encoding='utf-8')
    config=OUT/'dataset.yaml';config.write_text(f'path: {ROOT.as_posix()}\ntrain: {(OUT/"train_exposure.txt").as_posix()}\nval: {(OUT/"valid.txt").as_posix()}\nnames:\n  0: animal\n',encoding='utf-8')
    name=datetime.now(timezone.utc).strftime('roe_focus_%Y%m%dT%H%M%SZ');options=dict(old_run['training_options']);options.update(data=str(config),project=str(OUT/'training'),name=name)
    assert all(v==old_run['training_options'][k] for k,v in options.items() if k not in ('data','project','name'))
    roe_counts=Counter(e['name'] for e in exposure if e['source']=='roe')
    plan={'created_utc':datetime.now(timezone.utc).isoformat(),'dataset_revision':revision,'source_dataset_revision':old_index['revision'],'base_weights':str(weights),'base_weights_sha256':BASE_SHA,'previous_mixed_weights':old_run['weights'],'previous_mixed_weights_sha256':old_run['weights_sha256'],'unique_training_images':337,'unique_deer_images':310,'unique_roe_images':27,'training_entries_per_epoch':620,'source_entries':{'sherbrooke':310,'roe':310},'deer_positive_negative':[155,155],'roe_repetitions_min_max':[min(roe_counts.values()),max(roe_counts.values())],'sampling_seed':20261102,'manifest_sha256':digest(OUT/'train_exposure.txt'),'options':options,'checkpoint_policy':old_run['checkpoint_policy'],'rationale':'Increase roe source exposure from 17.4% to 50%; keep 620 entries/epoch, 10 epochs, optimizer and initial weights identical to prior mixed run. Deer replay subset 155 positive and 155 negative is training-only deterministic selection.','limitations':'Same 27 unique roe frames, repeated 11-12 times; repeats are not independent data. Source balance and deer subset/diversity change together. Controls are developmental and not flight-independent.','evaluation_policy':'Fresh original, previous mixed and focused weights evaluated at input1024 conf0.25, batch1 FP32 on unchanged32+32 controls; check training fit separately.'}
    waid.save_json(OUT/'experiment_plan.json',plan)
    data=check_det_dataset(str(config),autodownload=False);loader=YOLODataset(img_path=data['train'],data=data,imgsz=1024,augment=False,rect=False,cache=False,batch_size=2)
    assert Counter(str(Path(p).resolve()) for p in loader.im_files)==Counter(str(Path(e['image_path']).resolve()) for e in exposure)
    expected={str(Path(e['image_path']).resolve()):e['reference_boxes'] for e in training}
    assert all(len(l['cls'])==expected[str(Path(l['im_file']).resolve())] and (l['cls']==0).all() for l in loader.labels)
    waid.save_json(OUT/'exposure_verification.json',{'passed':True,'unique_images':337,'training_entries':len(loader),'source_entries':plan['source_entries'],'roe_repetitions_min_max':plan['roe_repetitions_min_max'],'total_boxes_with_repeats':sum(len(l['cls']) for l in loader.labels),'validation_files_unchanged':validation==old_index['validation'],'training_options_unchanged_except_paths':True})
    del loader
    started=time.perf_counter();torch.cuda.reset_peak_memory_stats();model=YOLO(str(weights));assert model.names=={0:'animal'}
    def progress(trainer):waid.save_json(OUT/'training_status.json',{'state':'training','epoch_completed':int(trainer.epoch)+1,'epochs':10,'elapsed_seconds':time.perf_counter()-started,'run_directory':str(trainer.save_dir),'metrics':{k:float(v) for k,v in trainer.metrics.items()}})
    model.add_callback('on_fit_epoch_end',progress)
    waid.save_json(OUT/'training_status.json',{'state':'starting','epochs':10,'training_entries':620,'run_name':name})
    model.train(**options)
    run=Path(model.trainer.save_dir);last=run/'weights/last.pt';assert last.is_file() and digest(last)!=BASE_SHA
    assert all(digest(path)==expected_sha for path,expected_sha in frozen.items())
    for e in validation:assert digest(e['image_path'])==e['image_sha256'] and digest(e['label_path'])==e['label_sha256']
    result={'created_utc':datetime.now(timezone.utc).isoformat(),'dataset_revision':revision,'weights':str(last),'weights_sha256':digest(last),'base_weights':str(weights),'base_weights_sha256':BASE_SHA,'previous_mixed_weights':old_run['weights'],'previous_mixed_weights_sha256':old_run['weights_sha256'],'checkpoint_policy':plan['checkpoint_policy'],'training_seconds':time.perf_counter()-started,'training_options':options,'unique_training_images':337,'unique_deer_images':310,'unique_roe_images':27,'training_entries_per_epoch':620,'source_entries':plan['source_entries'],'roe_repetitions_min_max':plan['roe_repetitions_min_max'],'validation_images':64,'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,'ultralytics':ultralytics.__version__,'gpu_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'gpu_peak_reserved_bytes':torch.cuda.max_memory_reserved(),'weights_file_bytes':last.stat().st_size,'precision':'Training FP32, AMP and TF32 disabled. Inference explicitly FP32.','existing_weights_controls_and_previous_reports_unchanged':True}
    waid.save_json(OUT/'latest_run.json',result);waid.save_json(run/'experiment_summary.json',result)
    waid.save_json(OUT/'training_status.json',{'state':'completed','epoch_completed':10,'epochs':10,'elapsed_seconds':result['training_seconds'],'weights':str(last)})
    print(json.dumps(result),flush=True)

if __name__=='__main__':
    try:main()
    except Exception as error:
        waid.save_json(OUT/'training_status.json',{'state':'failed','error':str(error)})
        raise
