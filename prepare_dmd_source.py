from pathlib import Path
import json,urllib.request,hashlib
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'data/dronemegadetector';OUT.mkdir(exist_ok=True)
def get(url):
    req=urllib.request.Request(url,headers={'User-Agent':'BeastAI-Diagnostic-Comparison'})
    with urllib.request.urlopen(req,timeout=45) as r:return r.read()
for name,url in [('repo.json','https://api.github.com/repos/ConservationDronesAI/DroneMegaDetector'),('release.json','https://api.github.com/repos/ConservationDronesAI/DroneMegaDetector/releases/tags/v1.1'),('rfdetr-pypi.json','https://pypi.org/pypi/rfdetr/json'),('source-tree.json','https://api.github.com/repos/ConservationDronesAI/DroneMegaDetector/contents/src')]:
    data=get(url);(OUT/name).write_bytes(data)
repo=json.loads((OUT/'repo.json').read_bytes());release=json.loads((OUT/'release.json').read_bytes());pypi=json.loads((OUT/'rfdetr-pypi.json').read_bytes())
print(json.dumps({'assets':[{'name':x['name'],'size':x['size'],'digest':x.get('digest'),'url':x['browser_download_url']} for x in release['assets']],'rfdetr_version':pypi['info']['version'],'rfdetr_python':pypi['info']['requires_python'],'rfdetr_dependencies':pypi['info']['requires_dist'],'source_files':[x['name'] for x in json.loads((OUT/'source-tree.json').read_bytes())]},ensure_ascii=True))
for filename in ('inference-batch.py','inference-demo.py'):
    try:
        data=get('https://raw.githubusercontent.com/ConservationDronesAI/DroneMegaDetector/main/src/'+filename);(OUT/filename).write_bytes(data);print(filename,len(data))
    except Exception as e:print(type(e).__name__,str(e))
