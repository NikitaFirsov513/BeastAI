from project_paths import PROJECT_ROOT
import json,pathlib,subprocess,sys
root=PROJECT_ROOT;out=root/'data'/'bambi'/'baseline';out.mkdir(parents=True,exist_ok=True)
experiments=[('640-base',['--imgsz','640']),('1024-detail',['--imgsz','1024','--no-latest']),('640-low-threshold',['--imgsz','640','--conf','0.1','--no-latest'])]
for name,options in experiments:
    print('Running '+name,flush=True)
    with (out/(name+'.log')).open('w',encoding='utf-8') as log:
        subprocess.run([sys.executable,str(PROJECT_ROOT/"run.py"),'baseline.py','--dataset','bambi',*options],stdout=log,stderr=subprocess.STDOUT,check=True)
    reports=sorted((p for p in out.glob('*-yolo11n-*.json')),key=lambda p:p.stat().st_mtime)
    report=json.loads(reports[-1].read_text(encoding='utf-8'))
    print(json.dumps({'experiment':name,'file':reports[-1].name,'valid':report['validation_preview'],'time_ms':report['timing_summary']['total_ms']},ensure_ascii=False),flush=True)