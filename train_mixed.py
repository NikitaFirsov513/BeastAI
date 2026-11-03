"""Fixed 10-epoch mixed fine-tune. No downloads; keep all existing weights and controls."""
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import hashlib,json,os,time
import waid
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'data/mixed_pilot'
BASE_SHA='ac1229f41c338a8c3913e792857c27fa40b5b7f52a17bd2772e4d4355b6beead'

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    assert ROOT.drive.lower()=='d:'
    for key,folder in [('YOLO_CONFIG_DIR','ultralytics-config'),('MPLCONFIGDIR','matplotlib-cache'),('TORCH_HOME','torch-cache')]:
        os.environ[key]=str(ROOT/'data'/folder)
    os.environ['YOLO_AUTOINSTALL']='false'
    import torch,cv2,ultralytics
    from ultralytics import YOLO,settings
    from ultralytics.data.dataset import YOLODataset
    from ultralytics.data.utils import check_det_dataset
    settings.update({'sync':False});torch.set_num_threads(4);cv2.setNumThreads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; CPU fallback is not automatic')
    index=json.loads((OUT/'index.json').read_text(encoding='utf-8'))
    original=json.loads((ROOT/'data/sherbrooke/train512/latest_run.json').read_text(encoding='utf-8'))
    weights=Path(original['weights']);assert digest(weights)==BASE_SHA
    assert len(index['training'])==539 and len(index['roe_training'])==27 and len(index['validation'])==64
    frozen_files={str(OUT/name):digest(OUT/name) for name in ('index.json','train.txt','valid.txt','dataset.yaml')}
    for e in index['training']+index['validation']:
        assert digest(e['image_path'])==e['image_sha256'] and digest(e['label_path'])==e['label_sha256']
    exposure=index['training']+index['roe_training']*3
    train_manifest=OUT/'train_exposure4.txt'
    train_manifest.write_text(''.join(Path(e['image_path']).as_posix()+'\n' for e in exposure),encoding='utf-8')
    config=OUT/'experiment_dataset.yaml'
    config.write_text(f'path: {ROOT.as_posix()}\ntrain: {train_manifest.as_posix()}\nval: {(OUT/"valid.txt").as_posix()}\nnames:\n  0: animal\n',encoding='utf-8')
    name=datetime.now(timezone.utc).strftime('mixed_%Y%m%dT%H%M%SZ')
    options=dict(original['training_options'])
    options.update(data=str(config),epochs=10,lr0=.0003,patience=10,project=str(OUT/'training'),name=name,exist_ok=False,pretrained=True,resume=False)
    plan={'created_utc':datetime.now(timezone.utc).isoformat(),'dataset_revision':index['revision'],'base_weights':str(weights),'base_weights_sha256':BASE_SHA,'unique_training_images':539,'unique_roe_images':27,'training_entries_per_epoch':len(exposure),'roe_entries_per_epoch':108,'roe_exposure_multiplier':4,'manifest_sha256':digest(train_manifest),'options':options,'checkpoint_policy':'Fixed final epoch 10; validation-selected best is not used.','rationale':'Start from existing deer weights. Lower learning rate for adaptation. Repeat scarce roe images four times while retaining all 512 deer frames; repeats are not independent data.','validation_scope':'Same 32 deer and 32 roe frames, developmental only; no verified flight independence.','evaluation_policy':'Compare both fixed models with identical full-frame inference, confidence 0.25, match IoU 0.5, batch1 FP32 and extra duplicate filter IoU0.5. Show candidate even if quality regresses; do not silently replace old model.'}
    waid.save_json(OUT/'experiment_plan.json',plan)
    data=check_det_dataset(str(config),autodownload=False)
    loader=YOLODataset(img_path=data['train'],data=data,imgsz=1024,augment=False,rect=False,cache=False,batch_size=2)
    counts=Counter(str(Path(p).resolve()) for p in loader.im_files)
    assert counts==Counter(str(Path(e['image_path']).resolve()) for e in exposure)
    expected={str(Path(e['image_path']).resolve()):e['reference_boxes'] for e in exposure}
    assert all(len(label['cls'])==expected[str(Path(label['im_file']).resolve())] for label in loader.labels)
    waid.save_json(OUT/'exposure_verification.json',{'passed':True,'training_entries':len(loader),'unique_images':len(counts),'source_entries':{'sherbrooke':512,'roe':108},'total_boxes_with_repeats':sum(len(l['cls']) for l in loader.labels),'manifest_sha256':digest(train_manifest)})
    del loader
    started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    model=YOLO(str(weights));assert model.names=={0:'animal'}
    status=OUT/'training_status.json'
    def progress(trainer):
        waid.save_json(status,{'state':'training','epoch_completed':int(trainer.epoch)+1,'epochs':10,'elapsed_seconds':time.perf_counter()-started,'run_directory':str(trainer.save_dir),'metrics':{k:float(v) for k,v in trainer.metrics.items()}})
    model.add_callback('on_fit_epoch_end',progress)
    waid.save_json(status,{'state':'starting','epochs':10,'unique_training_images':539,'training_entries':620,'run_name':name})
    model.train(**options)
    run=Path(model.trainer.save_dir);last=run/'weights/last.pt';assert last.is_file() and digest(last)!=BASE_SHA and digest(weights)==BASE_SHA
    assert all(digest(path)==sha for path,sha in frozen_files.items())
    for e in index['validation']:
        assert digest(e['image_path'])==e['image_sha256'] and digest(e['label_path'])==e['label_sha256']
    result={'created_utc':datetime.now(timezone.utc).isoformat(),'dataset_revision':index['revision'],'weights':str(last),'weights_sha256':digest(last),'base_weights':str(weights),'base_weights_sha256':BASE_SHA,'checkpoint_policy':plan['checkpoint_policy'],'training_seconds':time.perf_counter()-started,'training_options':options,'unique_training_images':539,'unique_roe_images':27,'training_entries_per_epoch':620,'roe_exposure_multiplier':4,'validation_images':64,'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,'ultralytics':ultralytics.__version__,'gpu_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'gpu_peak_reserved_bytes':torch.cuda.max_memory_reserved(),'weights_file_bytes':last.stat().st_size,'precision':'Training FP32, AMP and TF32 off; saved checkpoint may be FP16; inference explicitly FP32.','existing_weights_and_controls_unchanged':True}
    waid.save_json(OUT/'latest_run.json',result);waid.save_json(run/'experiment_summary.json',result)
    waid.save_json(status,{'state':'completed','epoch_completed':10,'epochs':10,'elapsed_seconds':result['training_seconds'],'weights':str(last)})
    print(json.dumps(result),flush=True)

if __name__=='__main__':
    try:main()
    except Exception as error:
        waid.save_json(OUT/'training_status.json',{'state':'failed','error':str(error)})
        raise
