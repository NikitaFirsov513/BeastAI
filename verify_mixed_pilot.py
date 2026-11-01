"""Check the frozen mixed manifests with the real training loader; never train."""
import hashlib,json,os,urllib.request
from pathlib import Path
import waid
ROOT=Path(__file__).resolve().parent; OUT=ROOT/'data/mixed_pilot'
for key,folder in [('YOLO_CONFIG_DIR','ultralytics-config'),('MPLCONFIGDIR','matplotlib-cache'),('TORCH_HOME','torch-cache')]:
    os.environ[key]=str(ROOT/'data'/folder)
os.environ['YOLO_AUTOINSTALL']='false'

def main():
    import torch
    from ultralytics import settings
    from ultralytics.data.dataset import YOLODataset
    from ultralytics.data.utils import check_det_dataset
    settings.update({'sync':False});torch.set_num_threads(4)
    index=json.loads((OUT/'index.json').read_text(encoding='utf-8'))
    data=check_det_dataset(str(OUT/'dataset.yaml'),autodownload=False)
    checks={}
    for split,key in [('train','training'),('val','validation')]:
        entries=index[key]
        dataset=YOLODataset(img_path=data[split],data=data,imgsz=1024,augment=False,cache=False,rect=False,batch_size=2)
        expected={str(Path(e['image_path']).resolve()):e for e in entries}
        assert {str(Path(p).resolve()) for p in dataset.im_files}==set(expected)
        count=0
        for label in dataset.labels:
            entry=expected[str(Path(label['im_file']).resolve())]
            assert len(label['cls'])==entry['reference_boxes']
            assert (label['cls']==0).all()
            count+=len(label['cls'])
            assert hashlib.sha256(Path(entry['image_path']).read_bytes()).hexdigest()==entry['image_sha256']
            assert hashlib.sha256(Path(entry['label_path']).read_bytes()).hexdigest()==entry['label_sha256']
        tensor_checks=[]
        for source in ('sherbrooke','roe'):
            number=next(i for i,p in enumerate(dataset.im_files) if expected[str(Path(p).resolve())]['source']==source)
            sample=dataset[number]
            assert list(sample['img'].shape)==[3,1024,1024]
            assert len(sample['cls'])==expected[str(Path(dataset.im_files[number]).resolve())]['reference_boxes']
            tensor_checks.append({'source':source,'shape':list(sample['img'].shape),'labels':len(sample['cls'])})
        checks[split]={'images':len(dataset),'boxes':count,'tensor_checks':tensor_checks}
    weights=ROOT/'data/sherbrooke/training/train512_20261101T165631Z/weights/last.pt'
    assert hashlib.sha256(weights.read_bytes()).hexdigest()=='ac1229f41c338a8c3913e792857c27fa40b5b7f52a17bd2772e4d4355b6beead'
    waid.save_json(OUT/'verification.json',{'passed':True,'revision':index['revision'],'loader':checks,'current_weights_unchanged':True,'training_started':False})
    print(json.dumps(checks))

if __name__=='__main__':main()
