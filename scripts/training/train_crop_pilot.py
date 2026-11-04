"""Bounded crop-learning pilot on existing training groups; no downloads."""
from project_paths import PROJECT_ROOT
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import hashlib,json,os,time
import waid
from prepare_crop_lesson import clipped_labels,POSITIONS,SIZE
ROOT=PROJECT_ROOT;OUT=ROOT/'data/crop_pilot'
for key,folder in [('YOLO_CONFIG_DIR','ultralytics-config'),('MPLCONFIGDIR','matplotlib-cache'),('TORCH_HOME','torch-cache')]:os.environ[key]=str(ROOT/'data'/folder)
os.environ['YOLO_AUTOINSTALL']='false'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    import cv2,numpy as np,torch,ultralytics
    from ultralytics import YOLO,settings
    from ultralytics.data.dataset import YOLODataset
    from ultralytics.data.augment import LetterBox
    from ultralytics.models.yolo.detect import DetectionTrainer
    assert ROOT.drive.lower()=='d:';OUT.mkdir(exist_ok=True)
    if (OUT/'latest_run.json').exists() or (OUT/'experiment_plan.json').exists():raise RuntimeError('Preserve existing pilot; use a new experiment directory')
    settings.update({'sync':False});torch.set_num_threads(4);cv2.setNumThreads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    assert torch.cuda.is_available()
    source=ROOT/'data/roe_expanded';index=json.loads((source/'index.json').read_text(encoding='utf-8'));run=json.loads((source/'latest_run.json').read_text(encoding='utf-8'));base=Path(run['weights']);assert sha(base)==run['weights_sha256']
    protected={str(source/n):sha(source/n) for n in ('index.json','latest_run.json','reserved_fresh_groups.json','evaluation/after.json','evaluation/fresh_after.json','evaluation/published.json')};protected[str(base)]=sha(base)
    for e in index['training']+index['validation']+index['fresh_validation']:
        for k,h in [('image_path','image_sha256'),('label_path','label_sha256')]:assert sha(e[k])==e[h];protected[e[k]]=e[h]
    waid.save_json(OUT/'protected_artifacts.json',protected)
    excluded={e['image_sha256'] for e in index['validation']+index['fresh_validation']};tiles=[];uncontained=[]
    for folder in ('images','labels'):(OUT/folder).mkdir(exist_ok=True)
    for number,e in enumerate(index['roe_training'],1):
        assert e['image_sha256'] not in excluded
        image=cv2.imdecode(np.fromfile(e['image_path'],dtype=np.uint8),cv2.IMREAD_COLOR);assert image.shape[:2]==(1280,1280)
        truth=waid.parse_labels(Path(e['label_path']),['animal']);assert not truth['errors'];contained=set()
        for y in POSITIONS:
            for x in POSITIONS:
                name=f'{Path(e["name"]).stem}__x{x}_y{y}.png';ip=OUT/'images'/name;lp=OUT/'labels'/(Path(name).stem+'.txt');rows,mapping=clipped_labels(truth['boxes'],x,y)
                ok,encoded=cv2.imencode('.png',image[y:y+SIZE,x:x+SIZE]);assert ok;ip.write_bytes(encoded.tobytes())
                assert np.array_equal(cv2.imdecode(np.fromfile(ip,dtype=np.uint8),cv2.IMREAD_COLOR),image[y:y+SIZE,x:x+SIZE])
                lp.write_text(''.join('0 '+' '.join(f'{v:.12f}' for v in r[1:])+'\n' for r in rows),encoding='utf-8')
                parsed=waid.parse_labels(lp,['animal']);assert not parsed['errors'] and len(parsed['boxes'])==len(rows)
                tiles.append({'source':'roe','name':name,'source_name':e['name'],'source_image_sha256':e['image_sha256'],'source_label_sha256':e['label_sha256'],'visual_group':e['visual_group'],'image_path':str(ip),'label_path':str(lp),'image_sha256':sha(ip),'label_sha256':sha(lp),'reference_boxes':len(rows),'dimensions':[512,512],'origin_xy':[x,y],'source_box_mapping':mapping})
                contained.update(m['source_box'] for m in mapping if not m['truncated'])
        uncontained.extend([{'source_name':e['name'],'source_box':i} for i in range(len(truth['boxes'])) if i not in contained])
        if number%8==0:waid.save_json(OUT/'status.json',{'state':'preparing','source_frames_completed':number,'source_frames':58})
    deer=[e for e in index['training'] if e['source']=='sherbrooke'];assert len(deer)==512 and len(tiles)==522
    entries=deer+tiles
    revision=hashlib.sha256(json.dumps([(e['image_path'],e['image_sha256'],e['label_sha256']) for e in entries]).encode()).hexdigest()
    manifest=OUT/'train.txt';manifest.write_text(''.join(Path(e['image_path']).as_posix()+'\n' for e in entries),encoding='utf-8')
    config=OUT/'dataset.yaml';config.write_text(f'path: {ROOT.as_posix()}\ntrain: {manifest.as_posix()}\nval: {manifest.as_posix()}\nnames:\n  0: animal\n',encoding='utf-8')
    waid.save_json(OUT/'index.json',{'revision':revision,'source_dataset_revision':index['revision'],'training':entries,'roe_training_sources':index['roe_training'],'validation':index['validation'],'fresh_validation':index['fresh_validation'],'tile_size':512,'stride':384,'scope':'Train source groups unchanged; crop grid independent of annotations. Every target intersection is labeled, including clipped edge appearances. No validation images used by trainer; its last-epoch diagnostics are training-only.'})
    class FixedTrainer(DetectionTrainer):
        def build_dataset(self,img_path,mode='train',batch=None):
            return YOLODataset(img_path=img_path,data=self.data,imgsz=1024,augment=False,rect=False,cache=False,batch_size=batch,stride=int(self.model.stride.max()),hyp=self.args,pad=0,task='detect')
        def optimizer_step(self):
            super().optimizer_step()
            self.actual_optimizer_steps=getattr(self,"actual_optimizer_steps",0)+1
        def final_eval(self):
            # External evaluation below is the only held-out scoring; no auto best-epoch selection.
            pass
    ds=YOLODataset(img_path=str(manifest),data={'names':{0:'animal'},'channels':3},imgsz=1024,augment=False,rect=False,cache=False,batch_size=2,pad=0)
    expected={str(Path(e['image_path']).resolve()):e for e in entries};checked=[]
    assert len(ds)==len(entries) and Counter(str(Path(p).resolve()) for p in ds.im_files)==Counter(expected.keys())
    for i,path in enumerate(ds.im_files):
        e=expected[str(Path(path).resolve())];a=ds[i];assert len(a['cls'])==e['reference_boxes'] and (a['cls']==0).all()
        # All 522 crop inputs must equal real inference's interpolation exactly.
        if e['source']=='roe' or len(checked)<3:
            raw=cv2.imdecode(np.fromfile(path,dtype=np.uint8),cv2.IMREAD_COLOR);im=LetterBox(new_shape=(1024,1024),auto=False,scaleup=True,stride=32)(image=raw)
            tensor=torch.from_numpy(im[:,:,::-1].transpose(2,0,1).copy());assert torch.equal(a['img'],tensor),path
            assert torch.equal(a['img'],ds[i]['img'])
            checked.append({'name':e['name'],'input_sha256':hashlib.sha256(tensor.numpy().tobytes()).hexdigest()})
    waid.save_json(OUT/'input_verification.json',{'passed':True,'entries':len(entries),'roe_tiles':522,'roe_source_frames':58,'roe_source_boxes':89,'tile_label_appearances':sum(e['reference_boxes'] for e in tiles),'positive_tiles':sum(e['reference_boxes']>0 for e in tiles),'background_tiles':sum(e['reference_boxes']==0 for e in tiles),'clipped_appearances':sum(m['truncated'] for e in tiles for m in e['source_box_mapping']),'source_boxes_without_any_full_tile':uncontained,'all_labels_loaded':True,'all_crop_pixels_match_inference':True,'transforms':[type(t).__name__ for t in ds.transforms.transforms],'checked':checked});del ds
    options=dict(run['training_options']);options.update(data=str(config),epochs=3,patience=4,batch=2,nbs=8,imgsz=1024,lr0=.0001,lrf=1.,warmup_epochs=0.,warmup_bias_lr=.0001,save=False,val=False,save_period=-1,project=str(OUT/'training'),name=datetime.now(timezone.utc).strftime('crop_%Y%m%dT%H%M%SZ'),hsv_h=0.,hsv_s=0.,hsv_v=0.,scale=0.,translate=0.,fliplr=0.,flipud=0.,degrees=0.,shear=0.,perspective=0.,mosaic=0.,mixup=0.,cutmix=0.,copy_paste=0.,bgr=0.,workers=0,cache=False,rect=False,amp=False,plots=False)
    plan={'created_utc':datetime.now(timezone.utc).isoformat(),'dataset_revision':revision,'base_weights':str(base),'base_weights_sha256':sha(base),'training_options':options,'epochs':3,'unique_source_frames':570,'roe_source_frames':58,'training_entries':len(entries),'checkpoint_policy':'Fixed last epoch3, no control-based selection. Custom final_eval disabled.','hypothesis':'Crop learning may help tiny-target transfer when inference searches the whole frame with the same source tile size and input resolution.','evaluation_plan':{'controls':64,'additional_diagnostic':16,'modes':['before_full','before_tiles','after_full','after_tiles'],'tile_size':512,'overlap':.25,'input':1024,'batch':1,'precision':'FP32; TF32 off','confidence':.25,'local_nms_iou':.7,'global_merge_iou':.5,'match_iou':.5,'repeats':3,'fixed_small_reference_scale':1024,'selection':'Report all four modes. Preserve main candidate; no deployment promotion.','search':'Predictor accepts only original image path and mode. Fixed grid covers full frame; annotations used only after predictions for scoring.'},'limits':['Short pilot changes training presentations and training settings; not a one-factor causal comparison.','No verified independent flight IDs for roe; old and fresh sets are development diagnostics.','Crop copies are not new scenes.','Interpolation cannot recover absent source details.']}
    waid.save_json(OUT/'experiment_plan.json',plan)
    torch.cuda.reset_peak_memory_stats();model=YOLO(str(base));started=time.perf_counter()
    def epoch_done(trainer):waid.save_json(OUT/'status.json',{'state':'training','epoch_completed':trainer.epoch+1,'epochs':3,'training_seconds':time.perf_counter()-started,'loss':trainer.tloss.detach().cpu().tolist(),'optimizer_steps':int(getattr(model.trainer,'actual_optimizer_steps',0))})
    model.add_callback('on_fit_epoch_end',epoch_done)
    waid.save_json(OUT/'status.json',{'state':'training','epoch_completed':0,'epochs':3,'training_entries':len(entries)})
    model.train(trainer=FixedTrainer,**options)
    last=Path(model.trainer.save_dir)/'weights/last.pt';assert last.exists()
    result={'dataset_revision':revision,'weights':str(last),'weights_sha256':sha(last),'base_weights':str(base),'base_weights_sha256':sha(base),'training_options':options,'epochs':3,'optimizer_steps':int(getattr(model.trainer,'actual_optimizer_steps',0)),'training_seconds':time.perf_counter()-started,'gpu_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'gpu_peak_reserved_bytes':torch.cuda.max_memory_reserved(),'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,'ultralytics':ultralytics.__version__,'checkpoint_policy':plan['checkpoint_policy'],'main_candidate_unchanged':True}
    assert all(sha(p)==h for p,h in protected.items());waid.save_json(OUT/'latest_run.json',result);waid.save_json(OUT/'status.json',{'state':'trained_pending_external_evaluation','epochs':3,'training_seconds':result['training_seconds'],'optimizer_steps':int(getattr(model.trainer,'actual_optimizer_steps',0))});print(json.dumps(result),flush=True)
if __name__=='__main__':
    try:main()
    except Exception as error:waid.save_json(OUT/'status.json',{'state':'failed','error':str(error)});raise
