"""Bounded larger roe candidate download: one verified ZIP range per JPEG."""
from pathlib import Path
from collections import defaultdict,Counter
from datetime import datetime,timezone
import hashlib,io,json,random,struct,time,urllib.error,urllib.request,zipfile,zlib
from PIL import Image
from remote_zip import RemoteZipReader
import waid
ROOT=Path(__file__).resolve().parent;DATA=ROOT/'data/roe';OUT=DATA/'expanded_candidates';BUDGET=640*1024**2

def main():
    assert ROOT.drive.lower()=='d:';OUT.mkdir(parents=True,exist_ok=True)
    cached=json.loads((DATA/'audit/source_record.json').read_text(encoding='utf-8'))
    with urllib.request.urlopen('https://zenodo.org/api/records/18206908',timeout=35) as response:record=json.load(response)
    source=next(f for f in record['files'] if f['key']=='Dataset_young_deer_RGB.zip');previous=next(f for f in cached['files'] if f['key']==source['key'])
    assert source['size']==previous['size'] and source['checksum']==previous['checksum']
    waid.save_json(OUT/'source_record.json',record)
    coco=json.loads((DATA/'source_metadata/annotation_coco.json').read_text(encoding='utf-8'));images={im['id']:im for im in coco['images']};anns=defaultdict(list)
    for ann in coco['annotations']:
        if ann['category_id']==1:anns[ann['image_id']].append(ann)
    already={i['name'] for i in json.loads((DATA/'audit/summary.json').read_text(encoding='utf-8'))['sample']}
    already.update(i['name'] for i in json.loads((DATA/'mixed_candidates/index.json').read_text(encoding='utf-8'))['entries'])
    bins=defaultdict(list)
    for iid,im in images.items():
        if im['file_name'] in already:continue
        areas=[a['bbox'][2]*a['bbox'][3]*min(1024/im['width'],1024/im['height'])**2 for a in anns[iid]]
        if not areas:continue
        area=max(areas);group='below16' if area<16**2 else '16to32' if area<32**2 else '32to64' if area<64**2 else 'atleast64';bins[group].append(iid)
    rng=random.Random(20261103);selection=[]
    for group in ('below16','16to32','32to64','atleast64'):
        selection.extend({'image_id':iid,'name':images[iid]['file_name'],'size_group':group} for iid in rng.sample(sorted(bins[group]),128))
    selection.sort(key=lambda x:x['name']);assert len(selection)==512 and not already.intersection(i['name'] for i in selection)
    plan={'created_utc':datetime.now(timezone.utc).isoformat(),'images':512,'seed':20261103,'source_size':source['size'],'source_checksum':source['checksum'],'body_budget_bytes':BUDGET,'selection':selection,'note':'Equal maximum-target-size strata; selection does not use model predictions. Candidates are not yet training data. API/header traffic excluded from body accounting.'}
    p=OUT/'selection.json'
    if p.exists():
        saved=json.loads(p.read_text(encoding='utf-8'));assert saved['selection']==selection and saved['source_checksum']==source['checksum'];plan=saved
    else:waid.save_json(p,plan)
    reader=RemoteZipReader(source['links']['self'],source['size'],budget=BUDGET,minimum_interval=2.1)
    counter=OUT/'range_body_accounting.json';prior=json.loads(counter.read_text(encoding='utf-8')).get('cumulative_bytes',0) if counter.exists() else 0
    reader.budget=BUDGET-prior
    with zipfile.ZipFile(reader) as archive:
        infos={i.filename:i for i in archive.infolist()}
    assert reader.bytes_received<4*1024**2
    entries=[];started=time.perf_counter()
    def account():waid.save_json(counter,{'cumulative_bytes':prior+reader.bytes_received,'this_run_bytes':reader.bytes_received,'prior_runs_bytes':prior,'budget_bytes':BUDGET,'successful_requests_this_run':reader.requests,'rate_limit_waits':reader.rate_limit_waits,'scope':'Successful range response bodies, excludes API responses and HTTP headers.'})
    account()
    for number,item in enumerate(selection,1):
        name=item['name'];info=infos[name];dest=OUT/'images'/name;dest.parent.mkdir(parents=True,exist_ok=True)
        assert Path(name).name==name and len(name.encode())<128 and info.compress_size+65792<16*1024**2
        if dest.exists():content=dest.read_bytes()
        else:
            reader.seek(info.header_offset);blob=reader.read(info.compress_size+65792);account()
            values=struct.unpack('<4s5H3I2H',blob[:30]);assert values[0]==b'PK\x03\x04' and not values[2]&1
            method=values[3];name_length,extra_length=values[-2:];local_name=blob[30:30+name_length].decode('utf-8' if values[2]&2048 else 'cp437');assert local_name==name
            offset=30+name_length+extra_length;raw=blob[offset:offset+info.compress_size];assert len(raw)==info.compress_size
            assert method==info.compress_type and method in (0,8)
            content=raw if method==0 else zlib.decompress(raw,-15)
            assert len(content)==info.file_size and zlib.crc32(content)&0xffffffff==info.CRC
            with Image.open(io.BytesIO(content)) as image:image.load();assert image.size==(images[item['image_id']]['width'],images[item['image_id']]['height'])
            temp=dest.with_suffix('.jpg.part');temp.write_bytes(content);temp.replace(dest)
        assert len(content)==info.file_size and zlib.crc32(content)&0xffffffff==info.CRC
        im=images[item['image_id']];labels=[]
        for ann in anns[im['id']]:
            x,y,w,h=ann['bbox'];assert not ann['iscrowd'] and w>0 and h>0 and min(x,y)>=0 and x+w<=im['width'] and y+h<=im['height']
            labels.append(f"0 {(x+w/2)/im['width']:.8f} {(y+h/2)/im['height']:.8f} {w/im['width']:.8f} {h/im['height']:.8f}")
        payload=('\n'.join(labels)+'\n').encode();lp=OUT/'labels'/Path(name).with_suffix('.txt').name;lp.parent.mkdir(exist_ok=True)
        if lp.exists():assert lp.read_bytes()==payload
        else:lp.write_bytes(payload)
        entries.append({**item,'source':'roe','image_path':str(dest),'label_path':str(lp),'image_sha256':hashlib.sha256(content).hexdigest(),'label_sha256':hashlib.sha256(payload).hexdigest(),'source_member_crc32':info.CRC,'reference_boxes':len(labels),'dimensions':[im['width'],im['height']]})
        if number%8==0 or number==len(selection):
            waid.save_json(OUT/'partial_index.json',{'entries':entries,'completed':number,'total':len(selection)})
            waid.save_json(OUT/'status.json',{'state':'downloading','completed':number,'total':len(selection),'range_body_MiB':(prior+reader.bytes_received)/2**20,'elapsed_seconds':time.perf_counter()-started})
            print(f'{number}/512 verified; body {(prior+reader.bytes_received)/2**20:.1f} MiB',flush=True)
    account();assert prior+reader.bytes_received<=BUDGET
    result={'created_utc':datetime.now(timezone.utc).isoformat(),'plan':plan,'entries':entries,'candidate_images':len(entries),'candidate_boxes':sum(e['reference_boxes'] for e in entries),'image_bytes':sum(Path(e['image_path']).stat().st_size for e in entries),'range_body_bytes':prior+reader.bytes_received,'protected_names_not_downloaded':True}
    waid.save_json(OUT/'index.json',result);waid.save_json(OUT/'status.json',{'state':'completed','completed':512,'total':512,'image_MiB':result['image_bytes']/2**20,'range_body_MiB':result['range_body_bytes']/2**20,'boxes':result['candidate_boxes']})
    print('Larger candidate pool complete',flush=True)

if __name__=='__main__':
    for resume_attempt in range(12):
        try:
            main();break
        except (urllib.error.URLError,TimeoutError,OSError) as error:
            if resume_attempt==11:
                waid.save_json(OUT/'status.json',{'state':'failed','error':str(error)});raise
            delay=min(120,30*(resume_attempt+1))
            waid.save_json(OUT/'status.json',{'state':'retrying_network','attempt':resume_attempt+1,'retry_seconds':delay,'error':str(error)})
            print(f'Network interruption; resuming cached files after {delay}s: {error}',flush=True)
            remaining=delay
            while remaining>0:
                pause=min(30,remaining);time.sleep(pause);remaining-=pause
        except Exception as error:
            waid.save_json(OUT/'status.json',{'state':'failed','error':str(error)});raise
