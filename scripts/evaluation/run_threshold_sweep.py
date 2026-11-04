"""Capture threshold candidates once, then score saved predictions automatically."""
from project_paths import PROJECT_ROOT
import json,subprocess,sys
from pathlib import Path
import waid


def main():
    root=PROJECT_ROOT;data=root/'data/sherbrooke'
    training=json.loads((data/'training/latest_run.json').read_text(encoding='utf-8'))
    with (data/'threshold-probe.log').open('w',encoding='utf-8') as stream:
        subprocess.run([sys.executable,str(PROJECT_ROOT/"run.py"),'baseline.py','--dataset','sherbrooke','--weights',training['weights'],'--imgsz','1024','--conf','0.01','--repeats','1','--validation-only','--no-latest','--model-label','YOLO11n animal: threshold candidates'],cwd=root,stdout=stream,stderr=subprocess.STDOUT,check=True)
    reports=sorted((data/'baseline').glob('*-yolo11n-1024-conf0.01.json'),key=lambda p:p.stat().st_mtime)
    report=json.loads(reports[-1].read_text(encoding='utf-8'))
    assert report['model_sha256']==training['weights_sha256']
    waid.save_json(data/'baseline/threshold_candidates.json',report)
    subprocess.run([sys.executable,str(PROJECT_ROOT/"run.py"),'threshold_sweep.py'],cwd=root,check=True)


if __name__=='__main__':main()