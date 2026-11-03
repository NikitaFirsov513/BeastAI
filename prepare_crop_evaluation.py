"""Finish own training checkpoint before evaluation; freeze all prior candidates."""
from pathlib import Path
import hashlib,json,os
import waid
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'data/crop_pilot'
for key,folder in [('YOLO_CONFIG_DIR','ultralytics-config'),('MPLCONFIGDIR','matplotlib-cache'),('TORCH_HOME','torch-cache')]:os.environ[key]=str(ROOT/'data'/folder)
os.environ['YOLO_AUTOINSTALL']='false'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    if (OUT/'checkpoint_preparation.json').exists():raise RuntimeError('Checkpoint already finalized')
    import torch
    from ultralytics.utils.torch_utils import strip_optimizer
    p=OUT/'latest_run.json';run=json.loads(p.read_text(encoding='utf-8'));weights=Path(run['weights']);assert sha(weights)==run['weights_sha256']
    assert not (OUT/'evaluation/before.json').exists() and not (OUT/'evaluation/after.json').exists()
    checkpoint=torch.load(weights,map_location='cpu',weights_only=False)
    assert checkpoint['epoch']==2
    assert checkpoint['updates']==run['optimizer_steps'] and run['optimizer_steps']>0
    report={'passed':True,'epochs':3,'optimizer_steps':run['optimizer_steps'],'checkpoint_ema_updates_before_strip':checkpoint['updates'],'training_checkpoint_sha256':sha(weights),'training_checkpoint_bytes':weights.stat().st_size,'scope':'Remove optimizer state before timing/model-size comparison. Inference explicitly FP32; file stores half-precision weights as in earlier candidates.'}
    del checkpoint;strip_optimizer(weights)
    run['weights_sha256']=sha(weights);run['weights_file_bytes']=weights.stat().st_size;run['optimizer_removed_before_evaluation']=True
    report.update({'inference_checkpoint_sha256':run['weights_sha256'],'inference_checkpoint_bytes':weights.stat().st_size})
    for path,h in json.loads((OUT/'protected_artifacts.json').read_text(encoding='utf-8')).items():assert sha(path)==h
    waid.save_json(p,run);waid.save_json(OUT/'checkpoint_preparation.json',report);print(json.dumps(report),flush=True)
if __name__=='__main__':main()
