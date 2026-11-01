"""Download a small reviewed RGB red-deer preview from BAMBI, using HTTP ranges."""
import csv
from collections import defaultdict
import hashlib
import io
import json
from pathlib import Path
import zipfile
from PIL import Image
import waid
from remote_zip import RemoteZipReader

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data'/'bambi'
TRAIN_FLIGHTS={70,74,104,155}
VALID_FLIGHTS={13,236,240}


def eligible_rows(folder, flight):
    tracks=json.loads((folder/f'{flight}_labels.json').read_text())['tracks']
    members=json.loads((folder/'members.json').read_text())
    frame_members={int(Path(m['name']).stem):m['name'] for m in members if m['name'].startswith('rgb/') and m['name'].endswith('.jpg')}
    rows=defaultdict(list)
    for row in csv.reader((folder/f'{flight}_rgb_mot.txt').read_text().splitlines()):
        if row and int(row[0]) in frame_members:
            rows[int(row[0])].append(row)
    eligible={}
    for frame, boxes in rows.items():
        all_clear=True
        for row in boxes:
            meta=tracks.get(str(int(row[1])),{})
            segments=(meta.get('occlusion') or {}).get('rgb',[])
            if not(meta.get('class_name')=='red deer' and meta.get('gender') is not None and meta.get('label_source')=='agreement' and 'rgb' in meta.get('accepted_modalities',[]) and any(seg.get('occluded')=='clear' and seg['start_frame']<=frame<=seg['end_frame'] for seg in segments)):
                all_clear=False
        if all_clear:
            eligible[frame]=boxes
    spaced=[]
    for frame in sorted(eligible):
        if not spaced or frame-spaced[-1]>=30:
            spaced.append(frame)
    if len(spaced)>8:
        spaced=[spaced[round(i*(len(spaced)-1)/7)] for i in range(8)]
    return [(frame,frame_members[frame],eligible[frame]) for frame in spaced]


def main():
    scouts=json.loads((DATA/'scout.json').read_text())
    pairs=[]; transfer_bytes=0; provenance=[]
    for scout in scouts:
        flight=scout['flight']
        if flight not in TRAIN_FLIGHTS|VALID_FLIGHTS:
            continue
        folder=DATA/'scout'/str(flight)
        selected=eligible_rows(folder,flight)
        split='train' if flight in TRAIN_FLIGHTS else 'valid'
        archive_entry=scout['archive']
        reader=RemoteZipReader(archive_entry['links']['self'],archive_entry['size'],budget=32*1024**2)
        with zipfile.ZipFile(reader) as archive:
            for frame,member,rows in selected:
                name=f'{flight:03d}_{frame:06d}.jpg'
                image_path=DATA/'images'/split/name
                label_path=DATA/'labels'/split/(Path(name).stem+'.txt')
                image_path.parent.mkdir(parents=True,exist_ok=True)
                label_path.parent.mkdir(parents=True,exist_ok=True)
                expected_crc=archive.getinfo(member).CRC
                import zlib
                if image_path.exists() and zlib.crc32(image_path.read_bytes())==expected_crc:
                    content=image_path.read_bytes()
                else:
                    content=archive.read(member)  # zipfile verifies CRC32 before returning.
                    image_path.write_bytes(content)
                with Image.open(io.BytesIO(content)) as image:
                    image.verify()
                with Image.open(io.BytesIO(content)) as image:
                    width,height=image.size
                labels=[]
                for row in rows:
                    x,y,w,h=map(float,row[2:6])
                    x1,y1,x2,y2=max(0,x),max(0,y),min(width,x+w),min(height,y+h)
                    if x2<=x1 or y2<=y1:
                        raise ValueError('Invalid source box')
                    labels.append(f'0 {(x1+x2)/2/width:.8f} {(y1+y2)/2/height:.8f} {(x2-x1)/width:.8f} {(y2-y1)/height:.8f}')
                label_content=('\n'.join(labels)+'\n').encode('utf-8')
                label_path.write_bytes(label_content)
                def entry(kind, filename, payload):
                    return {'path':f'BAMBI/{kind}/{split}/{filename}','size':len(payload),'sha':waid.blob_hash(payload),'sha256':hashlib.sha256(payload).hexdigest()}
                pairs.append({'split':split,'name':name,'species':'red_deer','flight_id':flight,'frame_index':frame,'image':entry('images',name,content),'label':entry('labels',Path(name).stem+'.txt',label_content),'original_size':[width,height],'source_archive':archive_entry['links']['self'],'source_member':member,'source_crc32':expected_crc,'review':'All RGB MOT boxes in this selected frame have agreement, red deer, known gender and clear RGB visibility.'})
                print(f'flight {flight}, frame {frame}, {len(rows)} deer, {width}x{height}',flush=True)
        transfer_bytes+=reader.bytes_received
        provenance.append({'flight':flight,'split':split,'frames':len(selected),'range_bytes':reader.bytes_received,'archive_checksum':archive_entry.get('checksum'),'archive_url':archive_entry['links']['self']})
    if not pairs:
        raise ValueError('No eligible RGB frames')
    revision=hashlib.sha256(json.dumps([(p['flight_id'],p['frame_index'],p['image']['sha256'],p['label']['sha256']) for p in pairs],sort_keys=True).encode()).hexdigest()
    index={'dataset':'bambi','prefix':'BAMBI','title':'BAMBI RGB — благородные олени','revision':revision,'classes':['animal'],'category_names':{'red_deer':'Благородный олень'},'source_records':[21291168,21298044],'source_urls':['https://zenodo.org/records/21291168','https://zenodo.org/records/21298044'],'location':'Austria; selected species occurs in Eastern Europe. This is not imagery captured in Eastern Europe.','pairs':pairs,'download':{'image_range_bytes':transfer_bytes,'scouting_range_bytes':sum(s['range_bytes'] for s in scouts),'flights':provenance},'selection':['Small pilot selected before observing model predictions.','Only human-accepted, agreement-labelled red-deer RGB frames where every RGB MOT row is marked clear.','At most 8 frames per flight, minimum separation of 30 source-frame indices. Do not treat these as independent animals.','Split by flight ID; spatial/site independence across flights is not audited.','These labels cover reviewed tracks, not necessarily every animal in the scene. Unmatched predictions need visual review.','Clear-visibility positive frames only: not representative of all occlusion, empty backgrounds, or future drone flights.','Per-file sha values use Git blob SHA1 calculated locally; source ZIP member CRC32 and extracted-image SHA256 are retained.']}
    waid.save_json(DATA/'index.json',index)
    print(json.dumps({'images':len(pairs),'flights':len(provenance),'train_images':sum(p['split']=='train' for p in pairs),'valid_images':sum(p['split']=='valid' for p in pairs),'image_transfer_MiB':round(transfer_bytes/1024**2,2),'total_source_archives_downloaded':0}),flush=True)


if __name__=='__main__':
    main()