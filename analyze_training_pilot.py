"""Automatically compare pretrained and fine-tuned detectors on a frozen labelled split."""
from collections import Counter
import json
from pathlib import Path
import waid,baseline


def main():
    waid.configure_dataset('sherbrooke');index=waid.get_index();folder=waid.DATA/'baseline'
    training=json.loads((waid.DATA/'training/latest_run.json').read_text(encoding='utf-8'))
    settings=[('pretrained_1024.json','До обучения · 1024'),('finetuned_1024.json','После 20 эпох · 1024'),('finetuned_640.json','После 20 эпох · 640')]
    reports=[json.loads((folder/name).read_text(encoding='utf-8')) for name,_ in settings]
    keys=set(reports[0]['items']);valid=[p for p in index['pairs'] if p['split']=='valid']
    experiments=[];frame_results=[]
    for (filename,label),report in zip(settings,reports):
        assert report['dataset_revision']==index['revision'] and set(report['items'])==keys
        assert report['conditions']['batch_size']==1 and report['conditions']['model_dtype']=='torch.float32'
        assert report['conditions']['confidence_threshold']==.25 and report['conditions']['match_iou']==.5
        assert report['conditions']['gpu']==reports[0]['conditions']['gpu']
        counts=Counter();small_total=small_tp=0;class_counts=Counter();background_detection_frames=0;negative_fp=0
        for pair in valid:
            key='valid/'+pair['name'];item=report['items'][key]
            assert item['image_git_sha']==pair['image']['sha'] and item['label_git_sha']==pair['label']['sha']
            reference=waid.parse_labels(waid.local_file(pair['label']),index['classes'])['boxes'];predictions=item['boxes']
            match=baseline.match_boxes(predictions,reference,.5)
            width,height=item['original_size'];scale=min(1024/width,1024/height)
            small={i for i,b in enumerate(reference) if b['width']*width*scale*b['height']*height*scale<32**2}
            small_total+=len(small);small_tp+=len(small.intersection(match['matched_truth']))
            row={'images':1,'reference_boxes':len(reference),'predicted_boxes':len(predictions),'matched':match['tp'],'unmatched_reference':match['fn'],'unmatched_predictions':match['fp']}
            counts.update(row);class_counts.update(p['class_name'] for p in predictions)
            if not reference:
                background_detection_frames+=bool(predictions);negative_fp+=len(predictions)
            frame_results.append({'label':label,'name':pair['name'],**row})
        tp,fp,fn=counts['matched'],counts['unmatched_predictions'],counts['unmatched_reference']
        precision=tp/(tp+fp) if tp+fp else None;recall=tp/(tp+fn) if tp+fn else None
        f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None
        t=report['timing_summary']['total_ms']
        experiments.append({'label':label,'resolution':report['conditions']['input_shape'][2],'confidence_threshold':.25,**dict(counts),'precision':precision,'recall':recall,'f1':f1,'reference_box_recall':recall,'small_fixed_reference_at_1024':small_total,'small_fixed_matched':small_tp,'small_recall':small_tp/small_total if small_total else None,'background_images':sum(p['reference_count']==0 for p in valid),'background_images_with_false_detections':background_detection_frames,'false_detections_on_background':negative_fp,'median_total_ms':t['median'],'p95_total_ms':t['p95'],'timing_images':len(report['items']),'gpu_peak_allocated_MiB':report['memory']['gpu_peak_allocated_bytes']/1024**2,'model_file_MiB':report['model_file_bytes']/1024**2,'model_sha256':report['model_sha256'],'predicted_class_counts':dict(class_counts),'report_file':filename})
    assert experiments[1]['model_sha256']==experiments[2]['model_sha256']==training['weights_sha256']
    assert experiments[0]['model_sha256']==training['base_weights_sha256']
    analysis={'dataset':index['title'],'dataset_revision':index['revision'],'source_urls':index['source_urls'],'validation_images':len(valid),'validation_flights':[],'split_provenance':index['split_provenance'],'precision_available':True,'annotation_policy':index['annotation_coverage'],'target_scope':'One animal class represents labelled red deer in this pilot. Other wildlife species have not been trained or evaluated.','conditions':reports[0]['conditions'],'training':training,'experiments':experiments,'frame_results':frame_results,'limitations':index['selection']+['False detections and misses are calculated automatically against author COCO detection labels, assumed complete for the deer target.','Author train/validation split is retained; a separate final test set is not provided.','No checkpoint selection using these reported before/after counts: fixed last epoch 20 was specified in advance.','The same 32 held-out images and 115 reference boxes are used for all reports.','Small-object membership is fixed by the 1024-input baseline for every comparison.','Timing uses all 96 pilot frames, three repeats, batch 1, FP32. Energy per frame is not measured.']}
    waid.save_json(waid.DATA/'analysis.json',analysis)
    lines=['# Первое дообучение YOLO11n: автоматический результат','', '## Данные и протокол','Источник: Sherbrooke Red deer CARTEL, летняя RGB-часть, версия 2.0. Это съёмка с дрона над оленьей фермой в Квебеке, Канада. Вид встречается в Восточной Европе, но сцены и содержание животных не представляют естественные местные полёты.','64 учебных снимка: 48 с оленями, 16 с пустым фоном; 401 рамка. 32 проверочных: 8 с оленями, 24 с пустым фоном; 115 рамок. Это появления животных на кадрах, не 115 уникальных особей.','Авторы разделяли снимки по полётам до формирования train/validation; сохранены их исходные части. ID полётов в COCO-именах не указан, поэтому независимая проверка конкретных ID невозможна. Пустые кадры участвуют в расчёте ложных обнаружений. Число положительных кадров в пилоте намеренно повышено относительно полного набора; итоговую точность на будущих полётах отсюда выводить нельзя.','Изменения координат выполнены по исходным размерам 7952 × 5304; загружены официальные полнокадровые копии 1024 × 683. Кадры не обрезались. Оригинальные URL, ID и контрольные суммы сохранены.','', '## Обучение',f"20 заранее заданных эпох, вход 1024, пакет 2, эффективный пакет 8, AdamW, FP32, без AMP, seed 42. Для сравнения взят last.pt после фиксированной 20-й эпохи, а не best.pt, выбранный по проверочным метрикам. Длительность: {training['training_seconds']/60:.1f} мин. Пиковая выделенная память тензоров GPU: {training['gpu_peak_allocated_bytes']/1024**3:.2f} ГиБ.",'Эпоха — один проход по всем учебным изображениям. Проверочные изображения не участвуют в изменении весов. Их встроенные метрики могли вычисляться для наблюдения за обучением, но для выбора эпохи или настройки порога не использовались.','', '## Автоматическое сравнение','Порог оценки модели 0,25; сопоставление один к одному при IoU ≥ 0,5, без требования различать виды. В COCO-предобученной модели принимаются категории животных, в дообученной — один выход animal. В этом пилоте animal означает размеченного оленя.','', '| Модель / вход | Найдено | Пропущено | Ложных рамок | Точность выдачи | Доля найденных | Вся цепочка, медиана / p95, мс |','|---|---:|---:|---:|---:|---:|---:|']
    for e in experiments:
        precision='—' if e['precision'] is None else f"{e['precision']:.1%}"
        lines.append(f"| {e['label']} | {e['matched']} / {e['reference_boxes']} | {e['unmatched_reference']} | {e['unmatched_predictions']} | {precision} | {e['recall']:.1%} | {e['median_total_ms']:.1f} / {e['p95_total_ms']:.1f} |")
    lines+=['', 'Точность выдачи = верные рамки / все выданные рамки. Доля найденных = верные рамки / все рамки разметки. Все значения получены программно по неизменной разметке. Источник предназначен для детекции; полнота принимается по протоколу авторов, не объявляется независимо гарантированной.','', '## Малые животные и пустой фон']
    for e in experiments:
        lines.append(f"- {e['label']}: найдено малых {e['small_fixed_matched']} / {e['small_fixed_reference_at_1024']}; ложные обнаружения на пустом фоне: {e['false_detections_on_background']} рамок на {e['background_images_with_false_detections']} из {e['background_images']} пустых кадров.")
    lines+=['','Группа малых объектов фиксирована по исходному входу 1024: площадь рамки после этого масштабирования меньше 32² пикселей. При переходе к 640 эта группа не пересчитывается, чтобы сравнивались те же животные.','', '## Скорость, память и размер']
    for e in experiments:
        lines.append(f"- {e['label']}: веса {e['model_file_MiB']:.2f} МиБ; пик выделенных тензоров GPU {e['gpu_peak_allocated_MiB']:.1f} МиБ.")
    c=analysis['conditions'];lines += ['',f"Условия: {c['gpu']}; {c['cpu']}; PyTorch {c['torch']}, Ultralytics {c['ultralytics']}, CUDA {c['cuda']}. По одному кадру, FP32, TF32 отключён. Измерена вся цепочка: чтение/декодирование, подготовка, модель, обработка выхода и рамки на CPU. Загрузка модели, прогрев и запись JSON исключены. Чтение может использовать файловый кэш. Память драйвера и других процессов не включена. Энергия кадра не измерялась. Скорость ноутбука нельзя переносить на ещё не выбранную бортовую плату.",'', '## Артефакты',f"Дообученные веса: `{training['weights']}`",f"SHA256 исходных весов: `{training['base_weights_sha256']}`",f"SHA256 дообученных весов: `{training['weights_sha256']}`",f"Версия подготовленной выборки: `{index['revision']}`",'Подробные значения по кадрам: analysis.json; протокол и ход обучения: training/latest_run.json и results.csv в каталоге опыта. Исходные веса не перезаписывались.','', '## Ограничения','Первый короткий учебный опыт на 96 изображениях одного хозяйства. Выборка не охватывает другие виды, леса Восточной Европы, разные сезоны и высоты полёта. Отдельного итогового test здесь нет. Это не результат для эксплуатации.','', '## Источники',*index['source_urls'],'Лицензия источника: CC BY-NC 4.0; README авторов ограничивает использование академическими и исследовательскими целями.']
    (waid.DATA/'TRAINING_COMPARISON.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'training_minutes':training['training_seconds']/60,'experiments':experiments,'weights':training['weights']},ensure_ascii=True),flush=True)


if __name__=='__main__':main()