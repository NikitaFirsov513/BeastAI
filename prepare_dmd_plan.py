from pathlib import Path
import os,json,hashlib
import waid
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'data/dronemegadetector'
for key,value in [('YOLO_CONFIG_DIR','data/ultralytics-config'),('MPLCONFIGDIR','data/dronemegadetector/matplotlib-cache'),('TORCH_HOME','data/dronemegadetector/torch-cache'),('HF_HOME','data/dronemegadetector/hf-cache')]:os.environ[key]=str(ROOT/value)
import torch
weights=OUT/'weights/DroneMegaDetector_RFDETRMedium_2026.10.04.pth';payload=torch.load(weights,map_location='cpu',weights_only=False)
config=payload.get('model_config',{});assert isinstance(config,dict)
meta={'top_level_keys':list(payload),'model_config':config,'class_names':payload.get('class_names'),'args_type':str(type(payload.get('args'))),'model_tensor_count':len(payload['model']),'state_parameter_elements':sum(t.numel() for t in payload['model'].values() if isinstance(t,torch.Tensor))}
waid.save_json(OUT/'checkpoint_metadata.json',meta)
print(json.dumps(meta,default=str),flush=True)
index=json.loads((ROOT/'data/roe_expanded/index.json').read_text(encoding='utf-8'));run=json.loads((ROOT/'data/roe_expanded/latest_run.json').read_text(encoding='utf-8'))
protected={str(ROOT/'data/roe_expanded'/p):hashlib.sha256((ROOT/'data/roe_expanded'/p).read_bytes()).hexdigest() for p in ('index.json','latest_run.json','reserved_fresh_groups.json','evaluation/after.json','evaluation/fresh_after.json','evaluation/published.json')}
for p in [Path(run['weights']),weights]:protected[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
for e in index['validation']+index['fresh_validation']:
    for k,h in [('image_path','image_sha256'),('label_path','label_sha256')]:assert hashlib.sha256(Path(e[k]).read_bytes()).hexdigest()==e[h];protected[e[k]]=e[h]
waid.save_json(OUT/'protected_artifacts.json',protected)
waid.save_json(OUT/'index.json',{'revision':index['revision'],'validation':index['validation'],'fresh_validation':index['fresh_validation']})
waid.save_json(OUT/'experiment_plan.json',{'dataset_revision':index['revision'],'models':{'baseline':{'path':run['weights'],'sha256':run['weights_sha256'],'input':1024},'reference':{'path':str(weights),'sha256':protected[str(weights)],'release':'v1.1','input':config.get('resolution',1280)}},'confidence_threshold':.25,'match_iou':.5,'additional_duplicate_nms_iou':.5,'frozen_small_reference_scale':1024,'repeats':3,'batch_size':1,'precision':'FP32; AMP and TF32 disabled','training':False,'threshold_selection':'Fixed before any new prediction. Equal numerical thresholds do not imply equally calibrated confidence. No selection on the64+16 controls.','scope':'Specialist diagnostic comparator; no promotion. Training overlap unresolved pending source audit.','primary_search_mode':'Whole original frame, one network call. No reference-centered crops or tiling in primary comparison.','timing_plan':'Alternate model order per frame and repeat after warmup with both models resident; also measure isolated GPU allocations on representative Roe/deer frames separately. No external training concurrently.'})
