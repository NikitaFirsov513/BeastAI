"""Finish comparison after this fixed training run completes, without starting another training."""
from pathlib import Path
from datetime import datetime,timezone
import json,subprocess,sys,time
import waid
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'data/mixed_pilot'

def main():
    plan=json.loads((OUT/'experiment_plan.json').read_text(encoding='utf-8'));name=plan['options']['name'];deadline=time.monotonic()+3600
    waid.save_json(OUT/'comparison_status.json',{'state':'waiting_for_training','run_name':name})
    while True:
        status=json.loads((OUT/'training_status.json').read_text(encoding='utf-8'))
        if status['state']=='failed':raise RuntimeError('Training failed: '+status.get('error',''))
        if status['state']=='completed':
            training=json.loads((OUT/'latest_run.json').read_text(encoding='utf-8'))
            if Path(training['weights']).parents[1].name!=name:raise RuntimeError('Unexpected completed run')
            break
        if time.monotonic()>deadline:raise TimeoutError('Training did not complete within one hour')
        time.sleep(3)
    for stage,argv in [('before',['evaluate_mixed.py','--model','before']),('after',['evaluate_mixed.py','--model','after']),('publish',['publish_mixed.py'])]:
        waid.save_json(OUT/'comparison_status.json',{'state':'running','stage':stage,'run_name':name,'updated_utc':datetime.now(timezone.utc).isoformat()})
        with (OUT/('comparison-'+stage+'.log')).open('w',encoding='utf-8') as stream:
            subprocess.run([sys.executable,*argv],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
    waid.save_json(OUT/'comparison_status.json',{'state':'completed','run_name':name,'updated_utc':datetime.now(timezone.utc).isoformat()})
    print('Both models compared and reports published',flush=True)

if __name__=='__main__':
    try:main()
    except Exception as error:
        waid.save_json(OUT/'comparison_status.json',{'state':'failed','error':str(error)})
        raise
