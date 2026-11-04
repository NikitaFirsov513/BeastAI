"""Run pretrained baseline, fixed fine-tuning, and automatic evaluation sequentially."""
from project_paths import PROJECT_ROOT
import json,pathlib,subprocess,sys
import waid
ROOT=PROJECT_ROOT
DATA=ROOT/'data'/'sherbrooke'


def step(name,arguments):
    print('Running '+name,flush=True)
    log=DATA/(name+'.log')
    with log.open('w',encoding='utf-8') as stream:
        result=subprocess.run([sys.executable,str(PROJECT_ROOT/"run.py"),*arguments],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
    if result.returncode:
        print('\n'.join(log.read_text(encoding='utf-8',errors='replace').splitlines()[-35:]),flush=True)
        raise RuntimeError(name+' failed. See '+str(log))


def main():
    waid.configure_dataset('sherbrooke');index=waid.get_index()
    for pair in index['pairs']:
        for kind in ('image','label'):
            assert waid.blob_hash(waid.local_file(pair[kind]).read_bytes())==pair[kind]['sha']
    step('pretrained_1024',['baseline.py','--dataset','sherbrooke','--imgsz','1024','--model-label','YOLO11n COCO pretrained'])
    folder=DATA/'baseline';pretrained=json.loads((folder/'latest.json').read_text(encoding='utf-8'));waid.save_json(folder/'pretrained_1024.json',pretrained)
    print(json.dumps({'pretrained_validation':pretrained['validation_preview'],'median_total_ms':pretrained['timing_summary']['total_ms']['median']}),flush=True)
    step('training_20_epochs',['train_pilot.py'])
    training=json.loads((DATA/'training/latest_run.json').read_text(encoding='utf-8'))
    step('finetuned_1024',['baseline.py','--dataset','sherbrooke','--weights',training['weights'],'--imgsz','1024','--model-label','YOLO11n animal: fixed 20 epochs'])
    trained=json.loads((folder/'latest.json').read_text(encoding='utf-8'));waid.save_json(folder/'finetuned_1024.json',trained)
    step('finetuned_640',['baseline.py','--dataset','sherbrooke','--weights',training['weights'],'--imgsz','640','--no-latest','--model-label','YOLO11n animal: fixed 20 epochs'])
    reports=sorted(folder.glob('*-yolo11n-640-conf0.25.json'),key=lambda p:p.stat().st_mtime)
    waid.save_json(folder/'finetuned_640.json',json.loads(reports[-1].read_text(encoding='utf-8')))
    print(json.dumps({'trained_validation':trained['validation_preview'],'median_total_ms':trained['timing_summary']['total_ms']['median'],'weights':training['weights'],'training_seconds':training['training_seconds']}),flush=True)


if __name__=='__main__':main()