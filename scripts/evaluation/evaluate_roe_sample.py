"""Evaluate the frozen 32-frame roe sample with the unchanged active full-frame detector."""
from project_paths import PROJECT_ROOT
from datetime import datetime,timezone
from pathlib import Path
from collections import Counter
import hashlib,json,os,platform,statistics,subprocess,time
import baseline,waid
from tiled_inference import map_box,merge_boxes
ROOT=PROJECT_ROOT
DATA=ROOT/'data/roe/audit';OUT=ROOT/'data/roe/evaluation'


def main():
    assert ROOT.drive.lower()=='d:','Project downloads/results must stay on D'
    for key,folder in [('YOLO_CONFIG_DIR','ultralytics-config'),('MPLCONFIGDIR','matplotlib-cache'),('TORCH_HOME','torch-cache')]:
        path=ROOT/'data'/folder;path.mkdir(parents=True,exist_ok=True);os.environ[key]=str(path)
    os.environ['YOLO_AUTOINSTALL']='false'
    import cv2,numpy as np,psutil,torch,torchvision,ultralytics
    from ultralytics import YOLO,settings
    settings.update({'sync':False});torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no silent CPU fallback')
    active_path=ROOT/'data/sherbrooke/baseline/latest.json';active_bytes=active_path.read_bytes();active=json.loads(active_bytes)
    weights=ROOT/active['model_path'];weight_sha=hashlib.sha256(weights.read_bytes()).hexdigest()
    assert weight_sha==active['model_sha256']=='ac1229f41c338a8c3913e792857c27fa40b5b7f52a17bd2772e4d4355b6beead'
    c=active['conditions'];assert c['input_shape']==[1,3,1024,1024] and c['confidence_threshold']==.25 and c['nms_iou']==.7 and c['global_merge_iou']==.5 and c['network_calls_per_frame']==1
    audit=json.loads((DATA/'summary.json').read_text(encoding='utf-8'));selection=json.loads((DATA/'selection.json').read_text(encoding='utf-8'));sample=audit['sample']
    assert len(sample)==32 and {i['name'] for i in sample}=={i['name'] for i in selection}
    references={};fingerprints=[]
    for item in sample:
        image_path=DATA/'images'/item['name'];label_path=DATA/'labels'/Path(item['name']).with_suffix('.txt').name
        assert hashlib.sha256(image_path.read_bytes()).hexdigest()==item['sha256']
        assert hashlib.sha256(label_path.read_bytes()).hexdigest()==item['label_sha256']
        truth=waid.parse_labels(label_path,['animal']);assert truth['status'] in ('ok','empty')
        assert len(truth['boxes'])==item['target_labels'];references[item['name']]=truth['boxes']
        fingerprints.append((item['name'],item['sha256'],item['label_sha256']))
    assert sum(map(len,references.values()))==44
    revision=hashlib.sha256(json.dumps(sorted(fingerprints)).encode()).hexdigest()
    run=OUT/'runs'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');run.mkdir(parents=True,exist_ok=False)
    plan={'sample_revision':revision,'sample_images':32,'reference_boxes':44,'model_path':str(weights),'model_sha256':weight_sha,'source_active_model':active['model'],'input_shape':[1,3,1024,1024],'confidence_threshold':.25,'detector_nms_iou':.7,'additional_merge_iou':.5,'match_iou':.5,'repeats':3,'warmup_frames':5,'precision':'FP32; TF32 disabled','batch_size':1,'scope':'Fixed stratified diagnostic sample; not independent final test or video event benchmark.'}
    waid.save_json(run/'plan.json',plan)
    load_start=time.perf_counter();model=YOLO(str(weights));assert model.names=={0:'animal'}
    config=dict(imgsz=1024,rect=False,device=0,half=False,batch=1,conf=.25,iou=.7,agnostic_nms=True,classes=baseline.animal_class_ids(model.names),max_det=300,verbose=False,save=False)

    def predict(path):
        torch.cuda.synchronize();start=time.perf_counter()
        image=cv2.imdecode(np.fromfile(path,dtype=np.uint8),cv2.IMREAD_COLOR)
        if image is None:raise ValueError('Image decode failed: '+str(path))
        height,width=image.shape[:2];read_ms=(time.perf_counter()-start)*1000
        result=model.predict(source=image,**config)[0]
        assert result.orig_shape==(height,width)
        boxes=[]
        for row in result.boxes.data.detach().cpu().tolist():
            box=map_box(row,(0,0,width,height),(width,height),model.names)
            if box:boxes.append(box)
        candidates=len(boxes);merge_start=time.perf_counter();boxes=merge_boxes(boxes,.5,300);merge_ms=(time.perf_counter()-merge_start)*1000
        torch.cuda.synchronize();total_ms=(time.perf_counter()-start)*1000
        return boxes,{'total_ms':total_ms,'read_decode_ms':read_ms,**{k+'_ms':float(v) for k,v in result.speed.items()},'additional_merge_ms':merge_ms},[width,height],candidates

    first=DATA/'images'/sample[0]['name'];_,cold,_,_=predict(first);load_first_ms=(time.perf_counter()-load_start)*1000
    for _ in range(5):predict(first)
    # Positive control: the exact predictor must reproduce the active report on the original frozen validation frames.
    waid.configure_dataset('sherbrooke');control_index=waid.get_index();control_counts=Counter();control_items=[]
    for pair in control_index['pairs']:
        if pair['split']!='valid':continue
        for kind in ('image','label'):
            assert hashlib.sha256(waid.local_file(pair[kind]).read_bytes()).hexdigest()==pair[kind]['sha256']
        control_truth=waid.parse_labels(waid.local_file(pair['label']),control_index['classes'])['boxes']
        control_boxes,_,_,_=predict(waid.local_file(pair['image']))
        control_match=baseline.match_boxes(control_boxes,control_truth,.5)
        control_counts.update({k:control_match[k] for k in ('tp','fp','fn')})
        control_items.append({'name':pair['name'],**{k:control_match[k] for k in ('tp','fp','fn')}})
    expected_control={k:active['validation_preview'][k] for k in ('tp','fp','fn')}
    assert dict(control_counts)==expected_control,('Positive control failed',dict(control_counts),expected_control)
    positive_control={'dataset':'Sherbrooke original frozen validation','images':len(control_items),'counts':dict(control_counts),'expected_counts':expected_control,'same_predictor_same_settings':True,'passed':True,'items':control_items}
    print('Positive control passed: '+json.dumps(dict(control_counts)),flush=True)
    torch.cuda.reset_peak_memory_stats();all_timings=[];items={};aggregate=Counter();size_groups={name:Counter() for name in ('area_below_16_squared','area_16_to_32_squared','area_32_to_64_squared','area_at_least_64_squared')};small_ref=small_tp=0;stable=True
    for number,item in enumerate(sample,1):
        truth=references[item['name']];timings=[];outputs=[];repeat_counts=[];max_candidates=0
        for _ in range(3):
            boxes,timing,shape,candidates=predict(DATA/'images'/item['name']);timings.append(timing);all_timings.append(timing);outputs.append(boxes);max_candidates=max(max_candidates,candidates)
            match=baseline.match_boxes(boxes,truth,.5);repeat_counts.append({k:match[k] for k in ('tp','fp','fn')})
        assert shape==item['dimensions']
        boxes=outputs[0];match=baseline.match_boxes(boxes,truth,.5);stable&=all(row==repeat_counts[0] for row in repeat_counts)
        aggregate.update({k:match[k] for k in ('tp','fp','fn')})
        w,h=shape;scale=min(1024/w,1024/h);small=set();matched=set(match['matched_truth'])
        for i,box in enumerate(truth):
            area=box['width']*w*box['height']*h*scale**2
            group='area_below_16_squared' if area<16**2 else 'area_16_to_32_squared' if area<32**2 else 'area_32_to_64_squared' if area<64**2 else 'area_at_least_64_squared'
            size_groups[group]['reference']+=1;size_groups[group]['found']+=i in matched
            if area<32**2:small.add(i)
        small_ref+=len(small);small_tp+=len(small.intersection(matched))
        misses=[{'reference_index':i,'box':box,'best_emitted_prediction_iou':max((baseline.overlap(box,p) for p in boxes),default=0)} for i,box in enumerate(truth) if i not in matched]
        items[item['name']]={'image_path':str(DATA/'images'/item['name']),'label_path':str(DATA/'labels'/Path(item['name']).with_suffix('.txt').name),'image_sha256':item['sha256'],'label_sha256':item['label_sha256'],'original_size':shape,'reference':truth,'predictions':boxes,'counts':repeat_counts[0],'repeat_counts':repeat_counts,'timings':timings,'small_reference':len(small),'small_matched':len(small.intersection(matched)),'max_candidates_before_extra_merge':max_candidates,'missed_references':misses}
        print(f"[{number}/32] {item['name']}: found={match['tp']}/{len(truth)} false={match['fp']}",flush=True)
    assert stable,'Quality counts changed between repeated predictions; inspect before publishing'
    assert aggregate['tp']+aggregate['fn']==44 and all(item['max_candidates_before_extra_merge']<300 for item in items.values())
    assert hashlib.sha256(weights.read_bytes()).hexdigest()==weight_sha and active_path.read_bytes()==active_bytes
    for item in sample:
        assert hashlib.sha256((DATA/'images'/item['name']).read_bytes()).hexdigest()==item['sha256']
        assert hashlib.sha256((DATA/'labels'/Path(item['name']).with_suffix('.txt').name).read_bytes()).hexdigest()==item['label_sha256']
    tp,fp,fn=(aggregate[k] for k in ('tp','fp','fn'))
    quality={'images':32,'reference_boxes':44,'positive_images':sum(bool(t) for t in references.values()),'images_without_target':sum(not t for t in references.values()),'matched':tp,'missed':fn,'false_boxes':fp,'precision':tp/(tp+fp) if tp+fp else None,'reference_recall':tp/44,'small_reference':small_ref,'small_found':small_tp,'small_reference_recall':small_tp/small_ref if small_ref else None,'false_boxes_on_images_without_target':sum(i['counts']['fp'] for i in items.values() if not i['reference']),'size_groups':{k:dict(v) for k,v in size_groups.items()},'quality_counts_identical_across_repeats':stable}
    try:driver=subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,memory.total,power.draw','--format=csv,noheader'],capture_output=True,text=True,timeout=10).stdout.strip()
    except (OSError,subprocess.TimeoutExpired):driver='unavailable'
    report={'created_utc':datetime.now(timezone.utc).isoformat(),'dataset':'roe_rgb_frozen_audit_sample','sample_revision':revision,'plan':plan,'quality':quality,'positive_control':positive_control,'conditions':{**plan,'gpu':torch.cuda.get_device_name(0),'driver_snapshot':driver,'cpu':platform.processor(),'os':platform.platform(),'ram_total_bytes':psutil.virtual_memory().total,'python':platform.python_version(),'torch':torch.__version__,'torchvision':torchvision.__version__,'ultralytics':ultralytics.__version__,'cuda':torch.version.cuda,'cpu_threads':4,'actual_model_dtype':str(next(model.model.parameters()).dtype)},'memory':{'gpu_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'gpu_peak_reserved_bytes':torch.cuda.max_memory_reserved(),'process_rss_bytes':psutil.Process().memory_info().rss},'cold_first_frame':cold,'load_and_first_frame_ms':load_first_ms,'timing_summary':{k:{'median':statistics.median(t[k] for t in all_timings),'p95':baseline.percentile([t[k] for t in all_timings],.95)} for k in all_timings[0]},'source_urls':['https://zenodo.org/records/18206908','https://fly4future.com/TQ16000038-v7'],'limitations':['32 deliberately stratified still images, not representative frequencies or independent final test.','44 are annotated appearances in images, not identified individual animals in video.','R1 event recall and R2 false alerts per minute cannot be evaluated from this sample.','Annotation completeness assumed from source detection protocol; semantic completeness is not independently certified.','The 10 source person boxes are excluded from the animal class; predictions on persons count as unmatched boxes for this target.','Fixed threshold, weights and full-frame preprocessing; no training, confidence sweep or geographic coordinate processing.','Timing includes reading and JPEG decoding, input preparation, inference, output transfer, normalized boxes and extra duplicate filtering.','Model load, warmup, reference parsing/scoring, JSON output and browser are excluded from warm latency. Repeated reads may use OS cache.','JPEG sample and older PNG set differ in dimensions/encoding; do not attribute between-dataset latency differences to model acceleration.','GPU allocation excludes driver and other processes. Per-frame energy not measured; laptop speed is not a future 1 TOPS prediction.'],'items':items}
    waid.save_json(run/'report.json',report);waid.save_json(OUT/'latest.json',report)
    p=quality['precision'];precision='не определена: нет выданных рамок' if p is None else f'{p:.1%}'
    lines=['# Перенос текущего детектора на RGB-кадры косуль','', 'Проверка выполнена автоматически по неизменной разметке. Обучение на этих снимках не проводилось. Это исходная диагностика нового источника, не итоговая оценка проекта.',
        '', '| Изображений | Рамок животных | Найдено | Пропущено | Ложных рамок |', '|---:|---:|---:|---:|---:|',f'| 32 | 44 | {tp} | {fn} | {fp} |',
        '',f'Доля найденных размеченных появлений: {tp/44:.1%}. Точность выдачи: {precision}. Малые животные: {small_tp} найдено из {small_ref}. Малые определены площадью рамки меньше 32² пикселей при входе 1024.',
        'Совпадение один к одному при IoU ≥ 0,5. IoU — площадь пересечения двух рамок, делённая на площадь их объединения. Отсутствующая рамка не сопоставляется. Люди не входят в целевой класс; они сохранены в исходном COCO.',
        '',f"Положительный контроль: тот же код и те же веса на прежних 32 проверочных кадрах воспроизвели {control_counts['tp']}/115 найденных, {control_counts['fn']} пропусков и {control_counts['fp']} ложные рамки. Это соответствует сохранённому рабочему отчёту.", '', '## Размеры животных', '| Площадь рамки на входе 1024 | Рамок | Найдено |','|---|---:|---:|']
    for group,n in quality['size_groups'].items():lines.append(f"| {group} | {n.get('reference',0)} | {n.get('found',0)} |")
    timings=report['timing_summary'];lines+=['','## Условия и ресурсы',f"YOLO11n после 20 эпох на 512 кадрах Sherbrooke; веса SHA256 {weight_sha}. Вход 1024×1024; один кадр; FP32, TF32 отключён; порог 0,25, внутренний фильтр IoU 0,7 и дополнительный IoU 0,5. {report['conditions']['gpu']}; CPU Intel Core i5-12450H; PyTorch {torch.__version__}, Ultralytics {ultralytics.__version__}, CUDA {torch.version.cuda}.",
        f"Вся цепочка на кадр: медиана {timings['total_ms']['median']:.2f} мс, p95 {timings['total_ms']['p95']:.2f} мс. Три повтора на каждом из 32 JPEG после пяти прогревочных запусков; подсчёты качества совпали во всех повторах.",
        f"Пик выделенных тензоров GPU {report['memory']['gpu_peak_allocated_bytes']/2**20:.1f} МиБ; RSS процесса {report['memory']['process_rss_bytes']/2**20:.0f} МиБ; файл весов {weights.stat().st_size/2**20:.2f} МиБ.",
        '', '| Этап | Медиана, мс | p95, мс |', '|---|---:|---:|']
    for key in ('read_decode_ms','preprocess_ms','inference_ms','postprocess_ms','additional_merge_ms'):lines.append(f"| {key} | {timings[key]['median']:.2f} | {timings[key]['p95']:.2f} |")
    lines+=['', 'Полное время включает чтение/декодирование JPEG, подготовку, сеть, обработку выхода, перенос рамок на CPU и дополнительный фильтр. Исключены загрузка модели, прогрев, чтение разметки, расчёт качества, запись JSON и браузер. Медианы этапов не обязаны складываться в медиану всей цепочки. Энергия кадра не измерялась.',
        '', '## Границы вывода',*['- '+note for note in report['limitations']],
        '', '## Артефакты',f"Неизменные веса: `{weights}`",f"Полный отчёт по кадрам и повторам: `{run/'report.json'}`",f"Версия образца: `{revision}`",'Повторный запуск из PowerShell в папке проекта: `.\\.venv\\Scripts\\python.exe evaluate_roe_sample.py`. Он только использует сохранённую сеть и сравнивает рамки, не обучает её. Всё сохраняется на D.',
        '', '## Источники',*report['source_urls']]
    (OUT/'TRANSFER_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'quality':quality,'latency':report['timing_summary']['total_ms'],'report':str(OUT/'TRANSFER_REPORT.md'),'weights_unchanged':True,'existing_viewer_report_unchanged':True}),flush=True)

if __name__=='__main__':main()
