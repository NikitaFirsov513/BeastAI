"""Automatic box-to-box comparison for a fixed BAMBI RGB preview. No visual review required."""
from collections import Counter,defaultdict
import json
from pathlib import Path
import statistics
import waid,baseline


def main():
    waid.configure_dataset('bambi');index=waid.get_index();folder=waid.DATA/'baseline'
    latest=json.loads((folder/'latest.json').read_text(encoding='utf-8'))
    report_by_setting={}
    for path in sorted(folder.glob('*-yolo11n-*.json')):
        report=json.loads(path.read_text(encoding='utf-8'))
        if report['dataset_revision']!=index['revision'] or report['model_sha256']!=latest['model_sha256']:
            continue
        c=report['conditions'];report_by_setting[(c['input_shape'][2],c['confidence_threshold'])]=(path,report)
    experiments=[];frame_rows=[]
    valid=[p for p in index['pairs'] if p['split']=='valid']
    for setting in ((640,.25),(1024,.25),(640,.1)):
        path,report=report_by_setting[setting]
        assert set(report['items'])==set(latest['items']), 'Different image sets cannot be compared as one optimisation'
        assert report['conditions']['gpu']==latest['conditions']['gpu']
        assert report['conditions']['precision']==latest['conditions']['precision']
        assert report['conditions']['batch_size']==1
        total=Counter();by_flight=defaultdict(Counter);best_ious=[];labels=Counter();small_reference=small_matched=0
        for pair in valid:
            key='valid/'+pair['name'];item=report['items'][key]
            assert item['image_git_sha']==pair['image']['sha'] and item['label_git_sha']==pair['label']['sha']
            truth=waid.parse_labels(waid.local_file(pair['label']),index['classes'])['boxes'];preds=item['boxes']
            match=baseline.match_boxes(preds,truth,.5)
            width,height=item['original_size'];scale=min(640/width,640/height)
            small={i for i,b in enumerate(truth) if b['width']*width*scale*b['height']*height*scale<32**2}
            small_reference+=len(small);small_matched+=len(small.intersection(match['matched_truth']))
            ious=[max((baseline.overlap(p,b) for p in preds),default=0) for b in truth]
            best_ious.extend(ious);labels.update(p['class_name'] for p in preds)
            row={'images':1,'reference_boxes':len(truth),'predicted_boxes':len(preds),'matched':match['tp'],'unmatched_reference':match['fn'],'unmatched_predictions':match['fp']}
            total.update(row);by_flight[str(pair['flight_id'])].update(row)
            frame_rows.append({'experiment':f'{setting[0]}-conf{setting[1]}','name':pair['name'],'flight_id':pair['flight_id'],**row,'best_iou_per_reference':ious})
        t=report['timing_summary']['total_ms'];c=report['conditions']
        experiments.append({'resolution':setting[0],'confidence_threshold':setting[1],**dict(total),'reference_box_recall':total['matched']/total['reference_boxes'],'precision':None,'precision_reason':'Published labels are not exhaustive per frame. Unmatched predictions cannot automatically be labelled false positives.','small_fixed_reference_at_640':small_reference,'small_fixed_matched':small_matched,'references_with_any_overlap':sum(i>0 for i in best_ious),'maximum_iou':max(best_ious),'predicted_class_counts':dict(labels),'median_total_ms':t['median'],'p95_total_ms':t['p95'],'timing_images':len(report['items']),'gpu_peak_allocated_MiB':report['memory']['gpu_peak_allocated_bytes']/1024**2,'by_flight':{k:dict(v) for k,v in by_flight.items()},'report_file':path.name})
    analysis={'dataset':index['title'],'dataset_revision':index['revision'],'source_urls':index['source_urls'],'validation_images':len(valid),'validation_flights':sorted({p['flight_id'] for p in valid}),'training_flights':sorted({p['flight_id'] for p in index['pairs'] if p['split']=='train'}),'match_rule':'Greedy one-to-one matching in descending prediction score, category-agnostic animal, IoU >= 0.5.','reference_definition':'Annotated appearances in frames, not unique animals.','conditions':latest['conditions'],'model_sha256':latest['model_sha256'],'model_file_MiB':latest['model_file_bytes']/1024**2,'experiments':experiments,'frame_results':frame_rows,'limitations':['All numerical comparisons here are automated against unchanged source labels.','Precision is withheld because source annotations cover reviewed trajectories, not all visible animals.','Reference-box recall describes only supplied boxes; it is not full-scene animal recall.','The small-object group is fixed by the 640-input baseline for all three runs.','The same 47 frames are timed; only the same 19 valid frames contribute to quality.','No training or change of model weights was performed.','This is a small positive-frame pilot from 7 flights. Negative frames, occlusion and independent sites need a separate evaluation set.','End-to-end timings include read/decode, preparation, inference and CPU box extraction. Driver and other-process memory is excluded; energy per frame is not measured.']}
    waid.save_json(waid.DATA/'analysis.json',analysis)
    lines=['# Автоматическое сравнение: олени BAMBI RGB','',f"Рабочая выборка: {len(index['pairs'])} кадров из 7 полётов. Обучающая часть: 28 кадров; проверочная: 19 кадров, 24 опубликованные рамки.",f"Полёты для проверки: {analysis['validation_flights']}; остальные полёты: {analysis['training_flights']}.",'','## Как считается результат','Для каждой рамки модели программа ищет ещё не занятую рамку разметки. Совпадение засчитывается при IoU ≥ 0,5. Вид животного при сопоставлении не требуется: целевой класс один — animal. Число 24 означает появления размеченных животных в выбранных кадрах, а не 24 разных особи.','', '| Вход | Порог модели | Совпадений / рамок | Несопоставленных предсказаний | Вся цепочка: медиана / p95, мс | Пик тензоров GPU, МиБ |','|---|---:|---:|---:|---:|---:|']
    for e in experiments:
        lines.append(f"| {e['resolution']} × {e['resolution']} | {e['confidence_threshold']} | {e['matched']} / {e['reference_boxes']} | {e['unmatched_predictions']} | {e['median_total_ms']:.1f} / {e['p95_total_ms']:.1f} | {e['gpu_peak_allocated_MiB']:.1f} |")
    lines+=['','Увеличение разрешения и снижение порога в этой пробе не увеличили число совпадений по выбранному правилу. Это не доказывает, что они бесполезны после дообучения. Снижение порога вывело больше рамок; оно не является обучением модели.','', '## Геометрия рамок, автоматически',f"В исходном режиме {experiments[0]['references_with_any_overlap']} из 24 рамок разметки имеют хотя бы какое-то пересечение с предсказанием. Максимальный IoU: {experiments[0]['maximum_iou']:.3f}. Поэтому 0 совпадений при пороге 0,5 не означает отсутствия любой локализации.",f"Фиксированная группа малых рамок: {experiments[0]['small_fixed_reference_at_640']} появления, определена по размеру после уменьшения до 640 и не меняется между опытами. Во всех трёх режимах число совпадений для неё равно нулю.",'','## Почему не публикуется точность по ложным обнаружениям','Опубликованная разметка этого выпуска содержит проверенные траектории, а не обязательно все животные каждого кадра. Поэтому предсказание без соответствующей рамки считается «несопоставленным», а не автоматически ошибочным. Число ложных обнаружений и precision по этой разметке не определяются надёжно. Дополнительный аудит хранится отдельно; основная таблица не требует визуального осмотра.','', '## Условия измерений',f"GPU: {analysis['conditions']['gpu']}. CPU: {analysis['conditions']['cpu']}. Веса: YOLO11n, {analysis['model_file_MiB']:.2f} МиБ. FP32, TF32 отключён; по одному кадру; 4 потока CPU. PyTorch {analysis['conditions']['torch']}, Ultralytics {analysis['conditions']['ultralytics']}, CUDA {analysis['conditions']['cuda']}.",'Время измерялось последовательно на тех же 47 кадрах, по 3 раза после прогрева. Условия и значения по каждому кадру сохранены в JSON. Уменьшение медианы в опыте с низким порогом нельзя считать ускорением: разброс велик, а чтение использует файловый кэш. Энергия кадра не измерялась.','', '## Предложение следующего шага','1. Для честной автоматической оценки использовать небольшую фиксированную проверочную часть с полной покадровой разметкой всех видимых животных, пустыми кадрами и отдельными полётами. Отбор/полнота разметки — разовая подготовка; дальнейшее сравнение моделей выполняется автоматически.','2. Дообучить ту же YOLO11n на одном классе animal и сравнить с сохранённым исходным запуском на неизменной проверочной части. Пока не увеличивать архитектуру и не вводить INT8: сначала требуется полезное качество.','3. После дообучения повторить сравнение 640/1024 и оценку малых животных, затем выбирать ускорение по задержке всей цепочки и памяти. Формат экспорта и INT8 выбирать после появления конкретной платы.','', 'Источники: '+', '.join(index['source_urls']),f"Версия выборки: {index['revision']}",f"SHA256 весов: {analysis['model_sha256']}"]
    (waid.DATA/'AUTOMATIC_COMPARISON.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'experiments':experiments},ensure_ascii=False))


if __name__=='__main__': main()