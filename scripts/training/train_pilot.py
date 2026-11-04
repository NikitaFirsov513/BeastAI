"""One small, fixed 20-epoch single-class fine-tuning pilot. Validation membership is frozen."""
from project_paths import PROJECT_ROOT
from datetime import datetime,timezone
import argparse,hashlib,json,os
from pathlib import Path
import sys,time
import waid

ROOT=PROJECT_ROOT
DATA=ROOT/'data'/'sherbrooke'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expanded",action="store_true",help="Use frozen 512-frame training selection with original 32 validation frames")
    args=parser.parse_args()
    experiment=DATA/"train512" if args.expanded else DATA
    expected_train=512 if args.expanded else 64
    for key,folder in [('YOLO_CONFIG_DIR','ultralytics-config'),('MPLCONFIGDIR','matplotlib-cache'),('TORCH_HOME','torch-cache')]:
        path=ROOT/'data'/folder;path.mkdir(parents=True,exist_ok=True);os.environ[key]=str(path)
    os.environ['YOLO_AUTOINSTALL']='false'
    import torch,ultralytics
    from ultralytics import YOLO,settings
    settings.update({'sync':False})
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; training was not silently switched to CPU')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    waid.configure_dataset('sherbrooke');pilot=waid.get_index()
    index=json.loads((experiment/'index.json').read_text(encoding='utf-8')) if args.expanded else pilot
    assert [p for p in index['pairs'] if p['split']=='valid']==[p for p in pilot['pairs'] if p['split']=='valid']
    for pair in index['pairs']:
        for kind in ('image','label'):
            assert hashlib.sha256(waid.local_file(pair[kind]).read_bytes()).hexdigest()==pair[kind]['sha256'],pair['name']
    assert sum(p['split']=='train' for p in index['pairs'])==expected_train
    assert sum(p['split']=='valid' for p in index['pairs'])==32
    base=ROOT/'data'/'models'/'yolo11n.pt'
    before_hash=hashlib.sha256(base.read_bytes()).hexdigest()
    name=datetime.now(timezone.utc).strftime(('train512_' if args.expanded else 'pilot_')+'%Y%m%dT%H%M%SZ')
    options=dict(data=str(experiment/'dataset.yaml'),epochs=20,imgsz=1024,batch=2,nbs=8,device=0,workers=0,amp=False,cache=False,optimizer='AdamW',lr0=.001,lrf=.1,weight_decay=.0005,seed=42,deterministic=True,mosaic=0.,mixup=0.,copy_paste=0.,scale=.15,translate=.05,fliplr=.5,flipud=.5,degrees=0.,multi_scale=False,rect=False,close_mosaic=0,patience=20,plots=False,save=True,save_period=-1,project=str(DATA/'training'),name=name,exist_ok=False,verbose=True)
    start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    model=YOLO(str(base))
    status_path=experiment/'training_status.json'
    def progress(trainer):
        status={'state':'training','epoch_completed':int(trainer.epoch)+1,'epochs':20,'training_images':expected_train,'elapsed_seconds':time.perf_counter()-start,'run_directory':str(trainer.save_dir),'metrics':{k:float(v) for k,v in trainer.metrics.items()}}
        waid.save_json(status_path,status)
    model.add_callback('on_fit_epoch_end',progress)
    waid.save_json(status_path,{'state':'starting','epochs':20,'training_images':expected_train,'run_name':name})
    model.train(**options)
    elapsed=time.perf_counter()-start
    run=Path(model.trainer.save_dir);weights=run/'weights'/'last.pt'
    assert weights.is_file(),'No last-epoch checkpoint'
    assert hashlib.sha256(base.read_bytes()).hexdigest()==before_hash,'Original COCO weights changed'
    final_hash=hashlib.sha256(weights.read_bytes()).hexdigest()
    assert final_hash!=before_hash
    summary={'dataset_revision':index['revision'],'run_directory':str(run),'weights':str(weights),'weights_sha256':final_hash,'base_weights_sha256':before_hash,'checkpoint_policy':'Fixed last epoch 20, not validation-selected best checkpoint.','training_seconds':elapsed,'training_options':options,'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,'ultralytics':ultralytics.__version__,'cpu_threads':torch.get_num_threads(),'gpu_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'gpu_peak_reserved_bytes':torch.cuda.max_memory_reserved(),'precision':'Training FP32, AMP disabled; saved checkpoint may be FP16; inference is explicitly FP32.','source_split_provenance':index['split_provenance'],'selection_notes':index['selection'],'license':index['license'],'training_images':expected_train,'validation_images':32,'weights_file_bytes':weights.stat().st_size}
    summary['evaluation_pilot_revision']=pilot['revision']
    summary_path=experiment/'latest_run.json' if args.expanded else DATA/'training'/'latest_run.json'
    waid.save_json(summary_path,summary)
    waid.save_json(run/'experiment_summary.json',summary)
    waid.save_json(status_path,{'state':'completed','epoch_completed':20,'epochs':20,'training_images':expected_train,'elapsed_seconds':elapsed,'weights':str(weights)})
    print(json.dumps({'weights':str(weights),'training_seconds':elapsed,'weights_sha256':final_hash,'gpu_peak_MiB':summary['gpu_peak_allocated_bytes']/2**20}),flush=True)


if __name__=='__main__':main()