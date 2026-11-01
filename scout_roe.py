import json,pathlib,zipfile
from remote_zip import RemoteZipReader
root=pathlib.Path('data/roe');root.mkdir(exist_ok=True)
meta=json.loads(pathlib.Path('data/dataset_candidates/roe_rgb.json').read_text(encoding='utf-8'));entry=meta['files'][0]
reader=RemoteZipReader(entry['links']['self'],entry['size'],budget=64*1024**2)
with zipfile.ZipFile(reader) as archive:
    members=[{'name':i.filename,'size':i.file_size,'compressed':i.compress_size,'crc32':i.CRC} for i in archive.infolist()]
    (root/'members.json').write_text(json.dumps(members),encoding='utf-8')
    annotations=[m for m in members if m['name'].lower().endswith(('.json','.yaml','.txt','.md'))]
    print(json.dumps({'members':len(members),'metadata_files':annotations,'first_image_names':[m['name'] for m in members if m['name'].lower().endswith(('.jpg','.png','.jpeg'))][:12]}),flush=True)
    for m in annotations:
        if m['size']>150*1024**2 or m['compressed']>15*1024**2:
            print('SKIP_LARGE',m['name'],flush=True);continue
        content=archive.read(m['name'])
        path=root/'source_metadata'/pathlib.PurePosixPath(m['name']);path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(content)
        if m['name'].lower().endswith('.json'):
            data=json.loads(content)
            if isinstance(data,dict) and 'images' in data:
                print(json.dumps({'annotation_file':m['name'],'images':len(data['images']),'annotations':len(data.get('annotations',[])),'categories':data.get('categories'),'first_image':data['images'][:2],'first_annotation':data.get('annotations',[])[:1]},ensure_ascii=False)[:2800],flush=True)
(root/'scout-transfer.json').write_text(json.dumps({'bytes_received':reader.bytes_received,'requests':reader.requests,'source_entry':entry}),encoding='utf-8')