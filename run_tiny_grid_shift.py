from pathlib import Path
import hashlib,json
import run_tiny_memorization as experiment
root=Path.cwd();out=root/'data/tiny_grid_shift';out.mkdir(exist_ok=True);protected={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'data/tiny_memorization').glob('*.json')}
try:
 experiment.OUT=out;experiment.SHIFT_TO_GRID=True;experiment.main()
 assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==sha for p,sha in protected.items())
except Exception as error:
 experiment.waid.save_json(out/'status.json',{'state':'failed','error':str(error)});raise
