"""Audit actual positive-assignment eligibility, no training or inference."""
from pathlib import Path
import os,json,hashlib
import waid
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'data/approach_review'
for k,v in [('YOLO_CONFIG_DIR','ultralytics-config'),('MPLCONFIGDIR','matplotlib-cache'),('TORCH_HOME','torch-cache')]:os.environ[k]=str(ROOT/'data'/v)
os.environ['YOLO_AUTOINSTALL']='false'
def main():
    import torch,cv2
    from ultralytics import YOLO,settings
    from ultralytics.data.dataset import YOLODataset
    from ultralytics.utils.tal import make_anchors,TaskAlignedAssigner
    from ultralytics.utils.ops import xywh2xyxy
    settings.update({'sync':False});torch.set_num_threads(4);cv2.setNumThreads(4)
    folder=ROOT/'data/crop_pilot';index=json.loads((folder/'index.json').read_text(encoding='utf-8'));run=json.loads((folder/'latest_run.json').read_text(encoding='utf-8'));sources={e['name']:e for e in index['roe_training_sources']};targets={}
    for name,e in sources.items():
        boxes=waid.parse_labels(Path(e['label_path']),['animal'])['boxes']
        for i,b in enumerate(boxes):
            area=b['width']*b['height']*1024**2
            if area<16**2:targets[(name,i)]={'source_name':name,'source_box':i,'full_input_wh':[b['width']*1024,b['height']*1024],'eligible_counts_in_fully_containing_crops':[],'crop_names':[]}
    assert len(targets)==30
    model=YOLO(run['weights']);strides=model.model.stride.cpu();del model
    anchors,scales=make_anchors([torch.zeros(1,1,1024//int(s),1024//int(s)) for s in strides],strides);anchors*=scales
    ds=YOLODataset(img_path=str(folder/'train.txt'),data={'names':{0:'animal'},'channels':3},imgsz=1024,augment=False,rect=False,cache=False,batch_size=2,pad=0)
    tiles={str(Path(e['image_path']).resolve()):e for e in index['training'] if e['source']=='roe'};checked=0
    for i,path in enumerate(ds.im_files):
        e=tiles.get(str(Path(path).resolve()))
        if not e:continue
        needed=[(j,m,targets[(e['source_name'],m['source_box'])]) for j,m in enumerate(e['source_box_mapping']) if (e['source_name'],m['source_box']) in targets and not m['truncated']]
        if not needed:continue
        sample=ds[i];assert len(sample['cls'])==len(e['source_box_mapping'])
        counts=TaskAlignedAssigner.select_candidates_in_gts(anchors,(xywh2xyxy(sample['bboxes'])*1024).unsqueeze(0))[0].sum(1).to(torch.int64).tolist()
        for j,m,target in needed:target['eligible_counts_in_fully_containing_crops'].append(counts[j]);target['crop_names'].append(e['name'])
        checked+=1
    rows=list(targets.values());assert all(r['eligible_counts_in_fully_containing_crops'] for r in rows)
    for r in rows:r['maximum_candidates']=max(r['eligible_counts_in_fully_containing_crops']);r['blocked_in_all_complete_crops']=r['maximum_candidates']==0
    result={'crop_dataset_revision':index['revision'],'training_tiny_original_references':30,'actual_crop_inputs_checked':checked,'blocked_in_all_fully_containing_crops':sum(r['blocked_in_all_complete_crops'] for r in rows),'at_least_one_eligible_complete_crop':sum(not r['blocked_in_all_complete_crops'] for r in rows),'native_model_strides':strides.tolist(),'random_geometric_augmentation_enabled':False,'rows':rows,'limits':'Counts concern this fixed crop-training recipe. Eligible centers are necessary, not sufficient for positive assignment. A blocked direct training target can still produce an incidental prediction. This is not a general limit of YOLO or RGB animal detection.'}
    OUT.mkdir(exist_ok=True);waid.save_json(OUT/'crop_assignment_audit.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
if __name__=='__main__':main()
