import json,pathlib,zipfile
from remote_zip import RemoteZipReader
root=pathlib.Path('data/bambi')
files=[f for record in (21291168,21298044) for f in json.loads((root/f'record-{record}.json').read_text(encoding='utf-8'))['files']]
summary=[]
for entry in sorted(files,key=lambda f:f['size'])[:12]:
    flight=entry['key'].removeprefix('flight_').removesuffix('.zip')
    reader=RemoteZipReader(entry['links']['self'],entry['size'],budget=12*1024**2)
    with zipfile.ZipFile(reader) as archive:
        entries=archive.namelist()
        labels=[n for n in entries if n.endswith(('_rgb_mot.txt','_labels.json','manifest.json'))]
        folder=root/'scout'/flight;folder.mkdir(parents=True,exist_ok=True)
        for name in labels:
            content=archive.read(name)
            (folder/pathlib.PurePosixPath(name).name).write_bytes(content)
        row={'flight':int(flight),'archive':entry,'members':len(entries),'rgb_images':sum('/rgb/' in '/'+n and n.endswith('.jpg') for n in entries),'labels':labels,'range_bytes':reader.bytes_received}
        (folder/'members.json').write_text(json.dumps([{'name':i.filename,'size':i.file_size,'compressed':i.compress_size,'crc32':i.CRC} for i in archive.infolist()]),encoding='utf-8')
        summary.append(row)
        print(json.dumps({k:v for k,v in row.items() if k!='archive'}),flush=True)
(root/'scout.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')