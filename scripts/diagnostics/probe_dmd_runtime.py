from project_paths import PROJECT_ROOT
from pathlib import Path
import os,json,time
import waid
ROOT=PROJECT_ROOT;OUT=ROOT/'data/dronemegadetector'
for key,folder in [('HF_HOME','hf-cache'),('TORCH_HOME','torch-cache'),('MPLCONFIGDIR','matplotlib-cache')]:os.environ[key]=str(OUT/folder)
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';os.environ['YOLO_CONFIG_DIR']=str(ROOT/'data/ultralytics-config');os.environ['YOLO_AUTOINSTALL']='false'
import torch,cv2,numpy as np,rfdetr
from ultralytics import settings
settings.update({'sync':False});torch.set_num_threads(4);cv2.setNumThreads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
start=time.perf_counter();model=rfdetr.from_checkpoint(str(OUT/'weights/DroneMegaDetector_RFDETRMedium_2026.10.04.pth'),device='cuda',amp=False)
load_seconds=time.perf_counter()-start
print('class_names',model.class_names,flush=True);print('resolution',model.model_config.resolution,flush=True)
module=model.model.model;assert module is not None;module.float().eval();seen={}
def before(m,args):
    t=args[0];seen['input_shape']=list(t.shape);seen['input_dtype']=str(t.dtype);seen['input_device']=str(t.device);seen['autocast_enabled']=torch.is_autocast_enabled('cuda')
def after(m,args,out):seen['output_dtypes']={k:str(v.dtype) for k,v in out.items() if isinstance(v,torch.Tensor)}
a=module.register_forward_pre_hook(before);b=module.register_forward_hook(after)
index=json.loads((ROOT/'data/roe_expanded/index.json').read_text(encoding='utf-8'));e=index['roe_training'][0];image=cv2.cvtColor(cv2.imdecode(np.fromfile(e['image_path'],np.uint8),cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)
with torch.inference_mode(),torch.autocast('cuda',enabled=False):pred=model.predict(image,threshold=.25,include_source_image=False)
torch.cuda.synchronize();a.remove();b.remove()
assert seen['input_shape']==[1,3,1280,1280] and seen['input_dtype']=='torch.float32' and not seen['autocast_enabled']
assert all(v=='torch.float32' for v in seen['output_dtypes'].values())
assert all(p.dtype==torch.float32 for p in module.parameters())
result={'passed':True,'model_type':type(model).__name__,'classes':model.class_names,'source_training_image':e['name'],'resolution':model.model_config.resolution,'load_seconds':load_seconds,'forward':seen,'parameter_elements':sum(p.numel() for p in module.parameters()),'parameter_dtype':'FP32','prediction_classes':sorted(set(int(c) for c in pred.class_id)),'predictions':len(pred),'cuda':torch.cuda.is_available()}
waid.save_json(OUT/'runtime_probe.json',result);print(json.dumps(result,default=str),flush=True)
