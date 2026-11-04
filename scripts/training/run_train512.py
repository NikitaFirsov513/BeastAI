"""Train the frozen 512-frame experiment, then benchmark three models sequentially."""
from project_paths import PROJECT_ROOT
from datetime import datetime, timezone
import json, pathlib, subprocess, sys
import waid
ROOT=PROJECT_ROOT
DATA=ROOT/'data/sherbrooke'; OUT=DATA/'train512'

def step(name,args):
    print('Running '+name,flush=True)
    waid.save_json(OUT/'experiment_status.json',{'state':'running','stage':name,'updated_utc':datetime.now(timezone.utc).isoformat()})
    with (OUT/(name+'.log')).open('w',encoding='utf-8') as stream:
        result=subprocess.run([sys.executable,str(PROJECT_ROOT/"run.py"),*args],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(name+' failed; see '+str(OUT/(name+'.log')))

def main():
    step('training',['train_pilot.py','--expanded'])
    training=json.loads((OUT/'latest_run.json').read_text(encoding='utf-8'))
    old=json.loads((DATA/'training/latest_run.json').read_text(encoding='utf-8'))
    (OUT/'baseline').mkdir(exist_ok=True)
    entries=[('pretrained_1024',None,'COCO pretrained'),('pilot64_1024',old['weights'],'YOLO11n: 64 images, 20 epochs'),('train512_1024',training['weights'],'YOLO11n: 512 images, 20 epochs')]
    for name,weights,label in entries:
        before=set((DATA/'baseline').glob('*-yolo11n-1024-conf0.25.json'))
        args=['baseline.py','--dataset','sherbrooke','--imgsz','1024','--conf','0.25','--repeats','3','--no-latest','--model-label',label]
        if weights:args+=['--weights',weights]
        step(name,args)
        created=set((DATA/'baseline').glob('*-yolo11n-1024-conf0.25.json'))-before
        assert len(created)==1,created
        report=json.loads(next(iter(created)).read_text(encoding='utf-8'))
        assert len(report['items'])==96
        waid.save_json(OUT/'baseline'/(name+'.json'),report)
    step('analysis',['analyze_train512.py'])
    step('publish',['publish_train512.py'])
    waid.save_json(OUT/'experiment_status.json',{'state':'completed','updated_utc':datetime.now(timezone.utc).isoformat(),'training_summary':str(OUT/'latest_run.json')})
    print('Training and all three benchmarks completed.',flush=True)

if __name__=='__main__':
    try:main()
    except Exception as error:
        waid.save_json(OUT/'experiment_status.json',{'state':'failed','error':str(error)})
        raise
