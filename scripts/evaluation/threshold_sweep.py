"""Evaluate confidence thresholds from one saved low-threshold detector run."""
from collections import Counter
import hashlib,json
from pathlib import Path
import waid,baseline

THRESHOLDS=(.01,.03,.05,.075,.1,.15,.2,.25,.3,.4,.5)


def score_frame(predictions,reference,threshold):
    accepted=[p for p in predictions if p['score']>threshold]
    return accepted,baseline.match_boxes(accepted,reference,.5)


def main():
    waid.configure_dataset('sherbrooke');index=waid.get_index();folder=waid.DATA/'baseline'
    candidates=json.loads((folder/'threshold_candidates.json').read_text(encoding='utf-8'))
    current=json.loads((folder/'finetuned_1024.json').read_text(encoding='utf-8'))
    assert candidates['dataset_revision']==current['dataset_revision']==index['revision']
    assert candidates['model_sha256']==current['model_sha256']
    assert candidates['conditions']['input_shape']==current['conditions']['input_shape']==[1,3,1024,1024]
    for key in ('model_dtype','precision','batch_size','nms_iou','class_agnostic_nms','match_iou'):
        assert candidates['conditions'][key]==current['conditions'][key]
    valid=[p for p in index['pairs'] if p['split']=='valid']
    assert set(candidates['items'])=={'valid/'+p['name'] for p in valid}
    assert all(len(i['boxes'])<300 for i in candidates['items'].values()),'Candidate output cap reached'
    points=[];frame_rows=[]
    for threshold in THRESHOLDS:
        totals=Counter();small_total=small_tp=0;negative_frames=negative_fp_frames=negative_fp=0
        for pair in valid:
            key='valid/'+pair['name'];item=candidates['items'][key]
            assert item['image_git_sha']==pair['image']['sha'] and item['label_git_sha']==pair['label']['sha']
            reference=waid.parse_labels(waid.local_file(pair['label']),index['classes'])['boxes']
            predictions,match=score_frame(item['boxes'],reference,threshold)
            width,height=item['original_size'];scale=min(1024/width,1024/height)
            small={i for i,b in enumerate(reference) if b['width']*width*scale*b['height']*height*scale<1024}
            small_total+=len(small);small_tp+=len(small.intersection(match['matched_truth']))
            row={'images':1,'reference_boxes':len(reference),'predicted_boxes':len(predictions),'matched':match['tp'],'missed':match['fn'],'false_detections':match['fp']}
            totals.update(row);frame_rows.append({'threshold':threshold,'name':pair['name'],**row})
            if not reference:
                negative_frames+=1;negative_fp_frames+=bool(predictions);negative_fp+=len(predictions)
        tp,fp,fn=totals['matched'],totals['false_detections'],totals['missed']
        points.append({'threshold':threshold,**dict(totals),'precision':tp/(tp+fp) if tp+fp else None,'recall':tp/(tp+fn),'f1':2*tp/(2*tp+fp+fn),'small_reference':small_total,'small_matched':small_tp,'small_recall':small_tp/small_total,'background_images':negative_frames,'background_images_with_false_detections':negative_fp_frames,'false_detections_on_background':negative_fp})
    baseline_point=next(p for p in points if p['threshold']==.25)
    assert (baseline_point['matched'],baseline_point['false_detections'],baseline_point['missed'])==(current['validation_preview']['tp'],current['validation_preview']['fp'],current['validation_preview']['fn']),'Saved low-threshold candidates do not reproduce original operating point'
    for lower,higher in zip(points,points[1:]):
        assert lower['matched']>=higher['matched'] and lower['false_detections']>=higher['false_detections']
    nonworse=[p for p in points if p['false_detections']<=baseline_point['false_detections']]
    candidate=max(nonworse,key=lambda p:(p['matched'],-p['false_detections'],p['threshold']))
    best_f1=max(points,key=lambda p:p['f1'])
    displayed={.05,.1,.15,.2,.25,candidate['threshold']}
    result={'dataset':'sherbrooke','dataset_revision':index['revision'],'model_sha256':current['model_sha256'],'conditions':candidates['conditions'],'validation_images':32,'reference_boxes':115,'small_reference':81,'background_images':24,'current_threshold':.25,'working_threshold_changed':False,'candidate_floor':.01,'candidate_boxes':sum(len(i['boxes']) for i in candidates['items'].values()),'candidate_cap_hit':False,'points':points,'display_thresholds':sorted(displayed,reverse=True),'best_without_increasing_current_false_count':candidate,'maximum_f1_on_this_validation_only':best_f1,'frame_results':frame_rows,'notes':['One inference pass per validation image at confidence 0.01; then saved post-NMS boxes are filtered using score > threshold.','At 0.25 the filtered candidates reproduce the original detector counts exactly.','Same 32 images, same 115 source labels, unchanged weights and 1024 input.','The 81 small-object instances are fixed at the 1024-input baseline.','Model score is not a calibrated probability of correctness.','No speed or energy is inferred for each threshold from this table. The candidate run timing describes only confidence 0.01.','This is threshold selection on development validation, not an independent final test.','The current working threshold stays 0.25; maximum F1 is descriptive, not a user-approved error budget.']}
    waid.save_json(waid.DATA/'threshold-analysis.json',result)
    lines=['# Порог обнаружения: автоматическое сравнение','', 'Модель после 20 эпох не менялась. Проверка: те же 32 кадра, 115 рамок оленей, 24 пустых кадра. Вход 1024 × 1024, по одному кадру, FP32. Совпадение при IoU ≥ 0,5.','Один запуск на каждый кадр при пороге 0,01 сохранил 220 рамок-кандидатов. Далее разные пороги проверяются фильтрацией этих рамок без повторного обучения и без нового запуска нейросети для каждой строки. При пороге 0,25 результат точно воспроизводит прежний опыт. Ограничение 300 рамок на кадр нигде не достигнуто.','', '| Порог | Найдено / 115 | Пропуски | Ложные рамки | Найдено малых / 81 | Ложные рамки на пустом фоне |','|---:|---:|---:|---:|---:|---:|']
    for p in reversed(points):lines.append(f"| {p['threshold']} | {p['matched']} | {p['missed']} | {p['false_detections']} | {p['small_matched']} | {p['false_detections_on_background']} |")
    lines+=['',f"При условии «не больше ложных рамок, чем исходные {baseline_point['false_detections']} на этой выборке» лучший из проверенных порогов — {candidate['threshold']}: найдено {candidate['matched']}, ложных {candidate['false_detections']}.",f"Максимум F1 среди проверенных строк: порог {best_f1['threshold']}, F1={best_f1['f1']:.3f}. Это лишь математический баланс двух ошибок, а не установленное требование проекта.",'','## Что изменяет порог','Порог — минимальная оценка модели для сохранения рамки. Меньший порог может вернуть пропущенных животных и добавить ложные рамки. Оценка 0,1 не означает вероятность 10%; оценки не калиброваны как вероятности.','Рабочая выдача пока сохранена при 0,25. Перед выбором рабочего порога нужно определить допустимый рост ложных обнаружений. Сама таблица уже рассчитана автоматически; просмотр картинок для расчёта не нужен.','', '## Ограничения','Эта выборка используется для настройки. После выбора порога потребуется отдельный итоговый test, не использованный при его выборе. Нельзя переносить эти числа на естественные полёты или другие виды. Времена отдельных порогов не измерялись этой фильтрацией: сохранённый замер относится только к исходному запуску 0,01.','', f"Версия данных: {index['revision']}",f"SHA256 модели: {current['model_sha256']}"]
    (waid.DATA/'THRESHOLD_COMPARISON.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'points':points,'best_without_extra_false':candidate,'best_f1':best_f1},ensure_ascii=True))


if __name__=='__main__':main()