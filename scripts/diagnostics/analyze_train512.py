"""Compare frozen validation predictions after the 512-frame training experiment."""
from project_paths import PROJECT_ROOT
from collections import Counter
import hashlib,json
from pathlib import Path
import baseline,waid
ROOT=PROJECT_ROOT
DATA=ROOT/'data/sherbrooke';OUT=DATA/'train512'


def summarize(label,report,pilot):
    valid=[p for p in pilot['pairs'] if p['split']=='valid']
    counts=Counter();small_total=small_tp=background_fp=background_frames=0
    frames=[]
    for pair in valid:
        item=report['items']['valid/'+pair['name']]
        assert item['image_git_sha']==pair['image']['sha'] and item['label_git_sha']==pair['label']['sha']
        truth=waid.parse_labels(waid.local_file(pair['label']),pilot['classes'])
        assert truth['status'] in ('ok','empty')
        truth=truth['boxes'];match=baseline.match_boxes(item['boxes'],truth,.5)
        counts.update({key:match[key] for key in ('tp','fp','fn')})
        w,h=item['original_size'];scale=min(1024/w,1024/h)
        small={i for i,b in enumerate(truth) if b['width']*w*scale*b['height']*h*scale<32**2}
        small_total+=len(small);small_tp+=len(small.intersection(match['matched_truth']))
        if not truth:
            background_frames+=bool(item['boxes']);background_fp+=len(item['boxes'])
        frames.append({'label':label,'name':pair['name'],**{k:match[k] for k in ('tp','fp','fn')}})
    tp,fp,fn=(counts[k] for k in ('tp','fp','fn'))
    c=report['conditions'];timing=report['timing_summary']
    result={'label':label,'resolution':1024,'confidence_threshold':.25,'images':len(valid),'reference_boxes':tp+fn,'predicted_boxes':tp+fp,'matched':tp,'unmatched_reference':fn,'unmatched_predictions':fp,'precision':tp/(tp+fp) if tp+fp else None,'recall':tp/(tp+fn),'reference_box_recall':tp/(tp+fn),'f1':2*tp/(2*tp+fp+fn),'small_fixed_reference_at_1024':small_total,'small_fixed_matched':small_tp,'small_recall':small_tp/small_total,'background_images':sum(p['reference_count']==0 for p in valid),'background_images_with_false_detections':background_frames,'false_detections_on_background':background_fp,'median_total_ms':timing['total_ms']['median'],'p95_total_ms':timing['total_ms']['p95'],'timing_images':len(report['items']),'gpu_peak_allocated_MiB':report['memory']['gpu_peak_allocated_bytes']/2**20,'process_rss_MiB':report['memory']['process_rss_bytes']/2**20,'model_file_MiB':report['model_file_bytes']/2**20,'parameters':report['parameters'],'model_sha256':report['model_sha256'],'model_path':report['model_path'],'stage_timing_ms':timing,'conditions':c}
    assert report['validation_preview']['tp']==tp and report['validation_preview']['fp']==fp and report['validation_preview']['fn']==fn
    return result,frames



def diagnose(report,pilot):
    counts=Counter();items=[]
    for pair in pilot['pairs']:
        if pair['split']!='valid':continue
        item=report['items']['valid/'+pair['name']]
        truth=waid.parse_labels(waid.local_file(pair['label']),pilot['classes'])['boxes']
        unmatched=set(range(len(truth)));false=[]
        for prediction in sorted(item['boxes'],key=lambda p:p['score'],reverse=True):
            best=max(unmatched,key=lambda i:baseline.overlap(prediction,truth[i]),default=None)
            if best is not None and baseline.overlap(prediction,truth[best])>=.5:unmatched.remove(best)
            else:false.append(prediction)
        for prediction in false:
            overlap=max((baseline.overlap(prediction,b) for b in truth),default=0)
            kind='overlaps_already_matched_reference_at_0_5' if overlap>=.5 else 'partial_overlap_0_1_to_0_5' if overlap>=.1 else 'no_reference_overlap_at_0_1'
            counts['false_'+kind]+=1
            items.append({'name':pair['name'],'kind':'false_prediction','max_iou_to_any_reference':overlap,'score':prediction['score']})
        w,h=item['original_size'];scale=min(1024/w,1024/h)
        for i in unmatched:
            box=truth[i];small=box['width']*w*scale*box['height']*h*scale<32**2
            overlap=max((baseline.overlap(box,p) for p in item['boxes']),default=0)
            counts['missed_small' if small else 'missed_larger']+=1
            counts['missed_with_any_prediction_iou_0_1_to_0_5' if .1<=overlap<.5 else 'missed_with_any_prediction_iou_at_least_0_5' if overlap>=.5 else 'missed_without_prediction_overlap_0_1']+=1
            items.append({'name':pair['name'],'kind':'missed_reference','small':small,'max_iou_to_any_prediction':overlap})
    return {'counts':dict(counts),'model_sha256':report['model_sha256'],'confidence_threshold':.25,'match_iou':.5,'notes':['Geometric diagnosis of emitted boxes only; not independent visual certification of predictions.'],'items':items}


def main():
    waid.configure_dataset('sherbrooke');pilot=waid.get_index()
    expanded=json.loads((OUT/'index.json').read_text(encoding='utf-8'))
    for pair in pilot['pairs']:
        for kind in ('image','label'):
            assert hashlib.sha256(waid.local_file(pair[kind]).read_bytes()).hexdigest()==pair[kind]['sha256'],pair['name']
    assert [p for p in pilot['pairs'] if p['split']=='valid']==[p for p in expanded['pairs'] if p['split']=='valid']
    training=json.loads((OUT/'latest_run.json').read_text(encoding='utf-8'))
    old_training=json.loads((DATA/'training/latest_run.json').read_text(encoding='utf-8'))
    settings=[('pretrained_1024','До обучения · 1024'),('pilot64_1024','64 снимка · 20 эпох · 1024'),('train512_1024','512 снимков · 20 эпох · 1024')]
    reports=[json.loads((OUT/'baseline'/(name+'.json')).read_text(encoding='utf-8')) for name,_ in settings]
    first=reports[0];experiments=[];frames=[]
    exact_conditions=('gpu','cpu','torch','ultralytics','cuda','input_shape','model_dtype','batch_size','cpu_threads','confidence_threshold','nms_iou','class_agnostic_nms','match_iou','repeats','warmup_frames')
    for (name,label),report in zip(settings,reports):
        assert report['dataset_revision']==pilot['revision'] and set(report['items'])==set(first['items'])
        assert all(report['conditions'][key]==first['conditions'][key] for key in exact_conditions)
        assert report['conditions']['batch_size']==1 and report['conditions']['model_dtype']=='torch.float32'
        assert report['conditions']['confidence_threshold']==.25 and report['conditions']['match_iou']==.5
        assert len(report['items'])==96
        row,frame=summarize(label,report,pilot);row['report_file']='train512/baseline/'+name+'.json';experiments.append(row);frames.extend(frame)
    assert reports[0]['model_sha256']==training['base_weights_sha256']==old_training['base_weights_sha256']
    assert reports[1]['model_sha256']==old_training['weights_sha256']
    assert reports[2]['model_sha256']==training['weights_sha256']
    assert training['training_images']==512 and training['training_options']['epochs']==20
    assert all(e['reference_boxes']==115 and e['small_fixed_reference_at_1024']==81 for e in experiments)
    assert all(hashlib.sha256((ROOT/report['model_path']).read_bytes()).hexdigest()==report['model_sha256'] for report in reports)
    limitations=expanded['selection']+[
        'Quality uses only the same 32 development frames and 115 annotated appearances; these are not unique individual deer.',
        'All latency measurements were rerun sequentially in this session on the same 96 frames, three repeats, batch 1, FP32, TF32 disabled.',
        'Train512 and train64 both use 20 epochs, but train512 has eight times as many minibatches per epoch and a different positive/background mixture. This is not an equal-compute ablation.',
        'More frames from the same farm do not establish generalization to other sites, species or Eastern European forests.',
        'No per-frame energy measurement; speed on the laptop cannot be inferred for a future 1 TOPS accelerator.',
        'Ground truth completeness is assumed according to the author detection protocol; it has not been independently certified.',
        'Fixed last epoch 20, not validation-selected best.pt. Validation was reused for development; a separate final test is still needed.'
    ]
    analysis={'dataset':pilot['title'],'dataset_revision':pilot['revision'],'training_dataset_revision':expanded['revision'],'source_urls':pilot['source_urls'],'validation_images':32,'validation_flights':[],'split_provenance':pilot['split_provenance'],'precision_available':True,'annotation_policy':pilot['annotation_coverage'],'target_scope':'One animal class for labelled red deer; other species have not been trained or evaluated.','conditions':first['conditions'],'training':training,'reference_training':old_training,'experiments':experiments,'frame_results':frames,'limitations':limitations}
    diagnosis=diagnose(reports[2],pilot)
    analysis['error_breakdown']=diagnosis['counts']
    waid.save_json(OUT/'error_breakdown.json',diagnosis)
    waid.save_json(OUT/'analysis.json',analysis)
    c=first['conditions'];lines=['# Обучение YOLO11n на 512 снимках: автоматическое сравнение','',
        '512 учебных изображений: 256 с оленями и 256 пустых; 2342 рамки. Все первые 64 учебных снимка сохранены. Проверка: те же 32 кадра (8 с животными и 24 пустых), 115 появлений оленей. Исходное разделение авторов по полётам сохранено; конкретные ID полётов в COCO не раскрыты.',
        '', '## Обучение',
        'Оба опыта начаты с одних исходных COCO-весов YOLO11n. 20 фиксированных эпох; вход 1024, batch 2, эффективный пакет 8, FP32, AdamW, seed 42. Для сравнения используются last.pt после эпохи 20. В новом опыте больше шагов изменения весов и иное соотношение кадров с животными и пустого фона; это не сравнение при одинаковом объёме вычислений.',
        f"Длительность обучения на 512: {training['training_seconds']/60:.1f} мин; пик выделенных тензоров GPU: {training['gpu_peak_allocated_bytes']/2**30:.2f} ГиБ.",
        '', '## Качество на неизменной проверочной части',
        'Рамки сопоставляются программно один к одному при IoU ≥ 0,5. Порог оценки сети 0,25. Все животные объединены в один класс. Визуальный осмотр для подсчётов не используется.',
        '', '| Модель | Найдено | Пропущено | Ложных рамок | Точность выдачи | Доля найденных | Малые животные |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for e in experiments:
        p='—' if e['precision'] is None else f"{e['precision']:.1%}"
        lines.append(f"| {e['label']} | {e['matched']} / 115 | {e['unmatched_reference']} | {e['unmatched_predictions']} | {p} | {e['recall']:.1%} | {e['small_fixed_matched']} / 81 |")
    old,new=experiments[1:];delta=new['matched']-old['matched'];fpdelta=new['unmatched_predictions']-old['unmatched_predictions']
    lines+=['',f"Изменение относительно 64 снимков: найденных {delta:+d}, ложных рамок {fpdelta:+d}. Малые объекты — неизменная группа из 81 рамки, площадь которых при входе 1024 меньше 32² пикселей.",'', '## Фон без животных']
    for e in experiments:
        lines.append(f"- {e['label']}: {e['false_detections_on_background']} ложных рамок на {e['background_images_with_false_detections']} из 24 пустых кадров.")
    lines+=['', '## Время одного кадра, память, размер модели',
        f"Условия: {c['gpu']}; {c['cpu']}; PyTorch {c['torch']}; Ultralytics {c['ultralytics']}; CUDA {c['cuda']}. Вход 1024×1024, batch 1, FP32, TF32 отключён. Все три модели заново измерены в одной сессии последовательно, на одних 96 кадрах, по три повтора после пяти прогревочных запусков.",
        '', '| Модель | Вся цепочка, медиана / p95, мс | Пик тензоров GPU, МиБ | Процесс RAM, МиБ | Файл весов, МиБ |',
        '|---|---:|---:|---:|---:|']
    for e in experiments:
        lines.append(f"| {e['label']} | {e['median_total_ms']:.2f} / {e['p95_total_ms']:.2f} | {e['gpu_peak_allocated_MiB']:.1f} | {e['process_rss_MiB']:.1f} | {e['model_file_MiB']:.2f} |")
    lines+=['', 'Вся цепочка включает чтение и декодирование файла, подготовку входа, выполнение сети, удаление дублирующих рамок и перенос результатов на CPU. Загрузка модели, прогрев, чтение разметки, расчёт метрик, запись JSON и браузер исключены. Повторное чтение может использовать кэш ОС. GPU-память — выделенные тензоры, без драйвера и других приложений. Небольшие различия времени могут быть шумом измерения; аппаратные режимы мощности не фиксировались. Энергия кадра не измерялась.',
        '', '## Этапы обработки, медианы в миллисекундах',
        '| Модель | Чтение и декодирование | Подготовка | Нейросеть | Обработка выхода |',
        '|---|---:|---:|---:|---:|']
    for e in experiments:
        t=e['stage_timing_ms'];values=[t[k]['median'] for k in ('read_decode_ms','preprocess_ms','inference_ms','postprocess_ms')]
        lines.append('| '+e['label']+' | '+' | '.join(f'{v:.2f}' for v in values)+' |')
    lines+=['', 'Медианы этапов не обязаны складываться в медиану всей цепочки. Дополнительные накладные расходы и преобразование рамок учитываются общим таймером.', '', '## Разбор оставшихся ошибок и следующий опыт',
        f"Из {new['unmatched_reference']} пропусков {diagnosis['counts'].get('missed_small',0)} относятся к малым животным, {diagnosis['counts'].get('missed_larger',0)} — к более крупным. Для {diagnosis['counts'].get('missed_without_prediction_overlap_0_1',0)} пропусков ни одна выданная при пороге 0,25 рамка не перекрывает разметку на IoU ≥ 0,1. Здесь анализируются только выданные ответы.",
        f"Из лишних рамок {diagnosis['counts'].get('false_overlaps_already_matched_reference_at_0_5',0)} перекрывают уже найденную разметку на IoU ≥ 0,5: вероятны дубли. Ещё {diagnosis['counts'].get('false_partial_overlap_0_1_to_0_5',0)} имеют частичное перекрытие 0,1–0,5. Это геометрическая диагностика, а не визуальное подтверждение содержания рамок.",
        'Следующий опыт по качеству: обработка изображения фрагментами на тех же весах и проверочных кадрах. Животное займёт больше пикселей на входе сети, но понадобится несколько запусков вместо одного. Нужны замеры полной задержки и проверка рамок на границах. Рост качества заранее не гарантируется.',
        f"В текущей файловой цепочке чтение и декодирование PNG занимают медианно {new['stage_timing_ms']['read_decode_ms']['median']:.2f} мс, сама сеть — {new['stage_timing_ms']['inference_ms']['median']:.2f} мс. Подготовка данных существенно влияет на задержку. Для камеры без PNG-файлов чтение будет другим.",
        '', '## Артефакты',
        f"Новые веса: `{training['weights']}`",f"SHA256: `{training['weights_sha256']}`",f"Исходные веса SHA256: `{training['base_weights_sha256']}`",f"Версия учебной подборки: `{expanded['revision']}`",
        'Подробные отчёты по моделям: baseline/*.json; значения по кадрам: analysis.json; журнал и параметры обучения: latest_run.json и training.log.',
        '', '## Ограничения',*['- '+item for item in limitations], '', '## Источники',*pilot['source_urls'],
        'Лицензия данных: CC BY-NC 4.0; исследовательское и учебное применение. Съёмка оленьей фермы в Квебеке, Канада.']
    (OUT/'TRAINING_COMPARISON.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'training_minutes':training['training_seconds']/60,'experiments':experiments},ensure_ascii=True),flush=True)

if __name__=='__main__':main()
