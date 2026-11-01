"""Download a bounded candidate pool for a mixed deer/roe training pilot; no training."""
from collections import defaultdict,Counter
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
from datetime import datetime,timezone
import hashlib,io,json,random,zipfile,zlib
from PIL import Image
from remote_zip import RemoteZipReader
import waid
ROOT=Path(__file__).resolve().parent;DATA=ROOT/'data/roe';OUT=DATA/'mixed_candidates'


def main():
    assert ROOT.drive.lower()=='d:'
    OUT.mkdir(parents=True,exist_ok=True)
    source_record=json.loads((DATA/'audit/source_record.json').read_text(encoding='utf-8'))
    source=next(f for f in source_record['files'] if f['key']=='Dataset_young_deer_RGB.zip')
    cached_members=json.loads((DATA/'members.json').read_text(encoding='utf-8'));members={m['name']:m for m in cached_members}
    coco=json.loads((DATA/'source_metadata/annotation_coco.json').read_text(encoding='utf-8'))
    images={im['id']:{k:im[k] for k in ('id','file_name','width','height')} for im in coco['images']};anns=defaultdict(list)
    for a in coco['annotations']:anns[a['image_id']].append({k:a[k] for k in ('id','category_id','bbox','iscrowd')})
    del coco
    protected=json.loads((DATA/'audit/summary.json').read_text(encoding='utf-8'))['sample'];protected_names={i['name'] for i in protected}
    rng=random.Random(20261102);bins=defaultdict(list)
    for iid,im in images.items():
        if im['file_name'] in protected_names:continue
        areas=[a['bbox'][2]*a['bbox'][3]*(min(1024/im['width'],1024/im['height'])**2) for a in anns[iid] if a['category_id']==1]
        area=max(areas) if areas else None
        group='no_target' if area is None else 'below_16_squared' if area<16**2 else '16_to_32_squared' if area<32**2 else '32_to_64_squared' if area<64**2 else 'at_least_64_squared'
        bins[group].append(iid)
    chosen=[]
    for group,count in [('below_16_squared',23),('16_to_32_squared',23),('32_to_64_squared',23),('at_least_64_squared',23),('no_target',4)]:
        for iid in rng.sample(sorted(bins[group]),count):chosen.append({'image_id':iid,'name':images[iid]['file_name'],'size_group':group})
    chosen.sort(key=lambda i:i['image_id']);assert len(chosen)==96 and len({i['name'] for i in chosen})==96
    plan={'source_record':'https://zenodo.org/records/18206908','source_archive_size':source['size'],'source_checksum':source['checksum'],'protected_sample_names':sorted(protected_names),'candidate_images':96,'selection_seed':20261102,'source_size_bins':dict(Counter(i['size_group'] for i in chosen)),'network_budget_bytes':128*1024**2,'scope':'Candidate preparation only. No training or independent-flight test claim.','selection':chosen}
    frozen=OUT/'selection.json'
    if frozen.exists():assert json.loads(frozen.read_text(encoding='utf-8'))==plan
    else:waid.save_json(frozen,plan)
    ledger_path=OUT/'network_ledger.json'
    if ledger_path.exists():ledger=json.loads(ledger_path.read_text(encoding='utf-8'))
    else:
        existing=[item for item in chosen if (OUT/'images'/item['name']).is_file()]
        upper=sum(members[item['name']]['compressed']+65536 for item in existing)+4*4*1024**2+4*65536
        ledger={'prior_attempt_completed_images':len(existing),'prior_successful_range_bytes_conservative_upper':upper,'basis':'Four central-directory reads bounded by 4 MiB each, completed member compressed bytes plus header prefetches, and four possible final orphan header reads. Source archive immutable; central-directory bound is verified again below.'}
        waid.save_json(ledger_path,ledger)
    resume_budget=plan['network_budget_bytes']-ledger['prior_successful_range_bytes_conservative_upper']
    buckets=[[]];estimates=[4*1024**2]

    for item in sorted(chosen,key=lambda i:members[i['name']]['compressed'],reverse=True):
        worker=0;buckets[worker].append(item)
        if not (OUT/'images'/item['name']).is_file():estimates[worker]+=members[item['name']]['compressed']+2*65536
    assert max(estimates)<resume_budget,'Estimated cumulative body budget would be exceeded'
    def worker(number,items):
        reader=RemoteZipReader(source['links']['self'],source['size'],budget=resume_budget,minimum_interval=2.1);rows=[]
        with zipfile.ZipFile(reader) as archive:
            infos={i.filename:i for i in archive.infolist()}
            assert reader.bytes_received<=4*1024**2,'Central directory exceeded accounting bound'
            for n,item in enumerate(items,1):
                name=item['name'];assert Path(name).name==name and '\\' not in name
                dest=OUT/'images'/name;dest.parent.mkdir(parents=True,exist_ok=True)
                if dest.exists():content=dest.read_bytes()
                else:
                    content=archive.read(name);temp=dest.with_suffix('.jpg.part');temp.write_bytes(content);temp.replace(dest)
                assert len(content)==infos[name].file_size and zlib.crc32(content)&0xffffffff==infos[name].CRC
                with Image.open(io.BytesIO(content)) as img:
                    img.load();assert img.size==(images[item['image_id']]['width'],images[item['image_id']]['height'])
                im=images[item['image_id']];labels=[]
                for ann in anns[im['id']]:
                    if ann['category_id']!=1:continue
                    x,y,w,h=ann['bbox'];assert not ann['iscrowd'] and w>0 and h>0 and x>=0 and y>=0 and x+w<=im['width'] and y+h<=im['height']
                    labels.append(f"0 {(x+w/2)/im['width']:.8f} {(y+h/2)/im['height']:.8f} {w/im['width']:.8f} {h/im['height']:.8f}")
                payload=('\n'.join(labels)+ ('\n' if labels else '')).encode('utf-8');lp=OUT/'labels'/Path(name).with_suffix('.txt').name;lp.parent.mkdir(exist_ok=True)
                if lp.exists():assert lp.read_bytes()==payload
                else:lp.write_bytes(payload)
                rows.append({**item,'image_path':str(dest),'label_path':str(lp),'image_sha256':hashlib.sha256(content).hexdigest(),'label_sha256':hashlib.sha256(payload).hexdigest(),'source_member_crc32':infos[name].CRC,'reference_boxes':len(labels),'dimensions':[im['width'],im['height']]})
                if n%4==0 or n==len(items):print(f'Worker {number+1}: {n}/{len(items)} candidates verified',flush=True)
        return rows,{'worker':number,'bytes_received':reader.bytes_received,'requests':reader.requests,'rate_limit_waits':reader.rate_limit_waits}
    entries=[];transfers=[]
    with ThreadPoolExecutor(max_workers=1) as pool:
        for future in as_completed([pool.submit(worker,i,items) for i,items in enumerate(buckets)]):
            rows,transfer=future.result();entries.extend(rows);transfers.append(transfer)
    entries.sort(key=lambda i:i['name']);total=sum(t['bytes_received'] for t in transfers)
    assert total+ledger['prior_successful_range_bytes_conservative_upper']<=plan['network_budget_bytes'] and not protected_names.intersection(i['name'] for i in entries)
    for item in protected:
        assert hashlib.sha256((DATA/'audit/images'/item['name']).read_bytes()).hexdigest()==item['sha256']
        assert hashlib.sha256((DATA/'audit/labels'/Path(item['name']).with_suffix('.txt').name).read_bytes()).hexdigest()==item['label_sha256']
    result={'created_utc':datetime.now(timezone.utc).isoformat(),'plan':plan,'entries':entries,'transfers':transfers,'network_accounting':{'previous_attempt_upper':ledger['prior_successful_range_bytes_conservative_upper'],'resume_measured':total,'cumulative_conservative_upper':ledger['prior_successful_range_bytes_conservative_upper']+total,'budget':plan['network_budget_bytes']},'downloaded_range_bytes_this_run':total,'candidate_image_bytes':sum(Path(i['image_path']).stat().st_size for i in entries),'candidate_target_boxes':sum(i['reference_boxes'] for i in entries),'protected_sample_unchanged':True}
    waid.save_json(OUT/'index.json',result)
    print(json.dumps({'candidate_images':len(entries),'target_boxes':result['candidate_target_boxes'],'range_MiB':total/2**20,'image_MiB':result['candidate_image_bytes']/2**20,'protected_sample_unchanged':True}),flush=True)

if __name__=='__main__':main()
