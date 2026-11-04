"""Headless subprocess fixture: no model, screen capture or GUI."""
from pathlib import Path
import json,sys,time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from screen_overlay_win32 import StopSignal
from overlay_process import EVENT_PREFIX
signal=StopSignal(sys.argv[1],open_existing=True)
mode=sys.argv[2]
try:
    if mode=='exit':raise SystemExit(7)
    if mode=='partial':
        sys.stdout.write(EVENT_PREFIX+'{"generation":');sys.stdout.flush()
    else:
        print(EVENT_PREFIX+json.dumps({'generation':1001,'kind':'status','value':'fixture-ready'}),flush=True)
    if mode=='cooperative':signal.wait(60)
    else:
        while True:time.sleep(.1)
finally:signal.close()
