"""Own the detector process and bound shutdown without blocking the Tk thread."""
from pathlib import Path
import json
import logging
import os
import queue
import subprocess
import sys
import threading
import time
from project_paths import PROJECT_ROOT

EVENT_PREFIX = "BEASTAI_EVENT "

class DetectorWorker:
    def __init__(self,region,model_key,threshold,fps,frames,events,generation,*,grace_seconds=1.5,clock=time.monotonic):
        from screen_overlay_win32 import StopSignal
        self.region=region;self.key=model_key;self.threshold=threshold;self.fps=fps
        self.frames=frames;self.events=events;self.generation=generation
        self.cancel=StopSignal();self.process=None;self.reader=None;self.stderr=None
        self.clock=clock;self.grace_seconds=grace_seconds;self.deadline=None
        self.stop_reason=None;self.forced=False;self.closed=False
    @property
    def pid(self):return self.process.pid if self.process else None
    @property
    def exitcode(self):return self.process.poll() if self.process else None
    def is_alive(self):return self.process is not None and self.process.poll() is None
    def start(self,command=None):
        if self.process is not None:raise RuntimeError("Detector worker already started")
        out=PROJECT_ROOT/'data/screen_overlay';out.mkdir(parents=True,exist_ok=True)
        python=Path(sys.executable).with_name('python.exe')
        if not python.is_file():python=Path(sys.executable)
        config={'region':self.region.bbox,'model':self.key,'threshold':self.threshold,'fps':self.fps,'generation':self.generation,'stop_event':self.cancel.name}
        args=command if command is not None else [str(python),'-u',str(PROJECT_ROOT/'run.py'),'screen_overlay_worker','--config',json.dumps(config)]
        self.stderr=(out/f'worker_{os.getpid()}_{self.generation}.log').open('a',encoding='utf-8')
        environment=os.environ.copy();environment.update(PYTHONUTF8='1',PYTHONUNBUFFERED='1')
        try:
            self.process=subprocess.Popen(args,cwd=PROJECT_ROOT,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=self.stderr,text=True,encoding='utf-8',errors='replace',bufsize=1,env=environment,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        except BaseException:
            self.stderr.close();self.stderr=None;self.cancel.close();raise
        self.reader=threading.Thread(target=self._read_messages,daemon=True,name=f'BeastAI-result-{self.pid}')
        self.reader.start();logging.info('Detector process started pid=%s generation=%s',self.pid,self.generation)
    def _read_messages(self):
        # Only this background reader touches the pipe. A partial line can never block Tk.
        try:
            with self.process.stdout as stream:
                for line in stream:
                    if not line.startswith(EVENT_PREFIX):continue
                    try:
                        message=json.loads(line[len(EVENT_PREFIX):]);kind=message['kind'];value=message['value']
                        if message['generation']!=self.generation:continue
                        if kind=='frame':
                            try:self.frames.put_nowait(value)
                            except queue.Full:
                                try:self.frames.get_nowait()
                                except queue.Empty:pass
                                try:self.frames.put_nowait(value)
                                except queue.Full:pass
                        elif kind in ('status','warning','ready','error','stopped'):
                            self.events.put((self.generation,kind,value))
                    except (ValueError,KeyError,TypeError):logging.warning('Ignored malformed detector event')
        except (OSError,ValueError):logging.exception('Detector result pipe closed unexpectedly')
    def request_stop(self,reason='user_stop'):
        if self.deadline is not None:return
        self.stop_reason=reason;self.deadline=self.clock()+self.grace_seconds
        self.cancel.set();logging.info('Detector stop requested pid=%s reason=%s',self.pid,reason)
    def poll_shutdown(self):
        if self.deadline is None or not self.is_alive() or self.clock()<self.deadline:return
        self.forced=True
        logging.warning('Detector exceeded stop deadline; terminating owned pid=%s',self.pid)
        try:self.process.terminate()
        except OSError:
            if self.is_alive():logging.exception('Could not terminate detector')
        self.deadline=self.clock()+.5
    def close(self):
        if self.closed:return
        if self.is_alive():raise RuntimeError('Cannot release a live detector process')
        if self.process:self.process.wait(timeout=0)
        if self.stderr:self.stderr.close();self.stderr=None
        self.cancel.close();self.closed=True
    def force_shutdown(self):
        # Exception/finally path: do not leave the child behind if the GUI fails.
        if self.is_alive():
            self.forced=True;self.process.terminate()
            try:self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill();self.process.wait(timeout=1)
        self.close()
