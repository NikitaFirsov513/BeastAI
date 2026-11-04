"""Complete this expanded experiment automatically after its data and training finish."""
from project_paths import PROJECT_ROOT
from pathlib import Path
from datetime import datetime,timezone
import json,subprocess,sys,time
import waid
ROOT=PROJECT_ROOT;OUT=ROOT/'data/roe_expanded'

def step(stage,args):
    waid.save_json(OUT/'comparison_status.json',{'state':'running','stage':stage,'updated_utc':datetime.now(timezone.utc).isoformat()})
    with (OUT/('comparison-'+stage+'.log')).open('w',encoding='utf-8') as stream:subprocess.run([sys.executable,str(PROJECT_ROOT/"run.py"),*args],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)

def main():
    waid.save_json(OUT/'comparison_status.json',{'state':'waiting_for_training'})
    deadline=time.monotonic()+4*3600
    while True:
        status=json.loads((OUT/'training_status.json').read_text(encoding='utf-8'))
        if status['state']=='failed':raise RuntimeError('Expanded training failed: '+status.get('error',''))
        if status['state']=='completed':break
        if time.monotonic()>deadline:raise TimeoutError('Expanded pipeline exceeded four-hour bound')
        time.sleep(3)
    for stage in ('before','after'):step(stage,['evaluate_mixed.py','--experiment','roe_expanded','--model',stage])
    step('after1280',['evaluate_mixed.py','--experiment','roe_expanded','--model','after','--imgsz','1280'])
    index=json.loads((OUT/'index.json').read_text(encoding='utf-8'))
    if index['fresh_validation']:
        for stage in ('before','after'):step('fresh-'+stage,['evaluate_mixed.py','--experiment','roe_expanded','--split','fresh','--model',stage])
        from publish_roe_expanded import choose_resolution
        if choose_resolution()=='after_1280':step('fresh-after1280',['evaluate_mixed.py','--experiment','roe_expanded','--split','fresh','--model','after','--imgsz','1280'])
    step('training-fit',['diagnose_mixed_training.py','--experiment','roe_expanded'])
    step('publish',['publish_roe_expanded.py'])
    waid.save_json(OUT/'comparison_status.json',{'state':'completed','updated_utc':datetime.now(timezone.utc).isoformat()})
    print('Expanded data, training and comparison complete',flush=True)

if __name__=='__main__':
    try:main()
    except Exception as error:
        waid.save_json(OUT/'comparison_status.json',{'state':'failed','error':str(error)});raise
