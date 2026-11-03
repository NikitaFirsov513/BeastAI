from pathlib import Path
import json,urllib.request,hashlib
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'data/dronemegadetector'
def request(url):return urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'BeastAI-Diagnostic-Comparison'}),timeout=60)
release=json.loads((OUT/'release.json').read_bytes());asset=release['assets'][0];assert asset['size']<200*1024**2
(OUT/'weights').mkdir(exist_ok=True)
for name,url,expected,digest in [(asset['name'],asset['browser_download_url'],asset['size'],asset['digest'].split(':')[1]),('paper.pdf','https://wildlabs.net/sites/default/files/2026-10/2026.10.12.758430v1.full_.pdf',None,None)]:
    p=OUT/'weights'/name if name.endswith('.pth') else OUT/name
    if p.exists() and (not digest or hashlib.sha256(p.read_bytes()).hexdigest()==digest):print('Cached',name,p.stat().st_size,flush=True);continue
    temporary=p.with_suffix(p.suffix+'.part');count=0;limit=200*1024**2 if name.endswith('.pth') else 40*1024**2
    with request(url) as response,temporary.open('wb') as f:
        while True:
            data=response.read(1024**2)
            if not data:break
            count+=len(data)
            if count>limit:raise RuntimeError('Download budget exceeded')
            f.write(data)
    if expected:assert count==expected
    checksum=hashlib.sha256(temporary.read_bytes()).hexdigest()
    if digest:assert checksum==digest
    temporary.replace(p);print(json.dumps({'path':str(p),'bytes':count,'sha256':checksum}),flush=True)
