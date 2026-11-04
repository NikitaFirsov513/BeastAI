"""Detector subprocess entry point; exchanges boxes and status, never screen images."""
import argparse
import json
import logging
import sys
from overlay_process import EVENT_PREFIX
from screen_overlay import DetectorTask, Region, OUT
from screen_overlay_win32 import StopSignal, enable_physical_pixels

class EventStream:
    def __init__(self,generation):self.generation=generation
    def send(self,kind,value):
        sys.stdout.write(EVENT_PREFIX+json.dumps({'generation':self.generation,'kind':kind,'value':value},ensure_ascii=True)+'\n')
        sys.stdout.flush()
    def put(self,event):self.send(event[1],event[2])

class FrameStream:
    def __init__(self,events):self.events=events
    def put_nowait(self,value):self.events.send('frame',value)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True)
    config=json.loads(parser.parse_args().config)
    logging.basicConfig(level=logging.INFO,stream=sys.stderr,format='%(asctime)s %(levelname)s %(message)s')
    enable_physical_pixels();OUT.mkdir(parents=True,exist_ok=True)
    signal=StopSignal(config['stop_event'],open_existing=True)
    try:
        events=EventStream(config['generation'])
        task=DetectorTask(Region(*config['region']),config['model'],config['threshold'],config['fps'],FrameStream(events),events,config['generation'],signal)
        if not signal.is_set():task.run()
        return 1 if task.failed else 0
    finally:signal.close()

if __name__=='__main__':raise SystemExit(main())
