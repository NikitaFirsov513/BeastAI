"""Validate and publish the balanced-source experiment without changing historical reports."""
from project_paths import PROJECT_ROOT
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,math
import baseline,publish_mixed,waid
ROOT=PROJECT_ROOT;OUT=ROOT/'data/roe_focus';EVAL=OUT/'evaluation'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    # Reuse the original before/after checks against the new experiment directory.
    publish_mixed.OUT=OUT;publish_mixed.EVAL=EVAL;publish_mixed.main()
    comparison=json.loads((EVAL/'comparison.json').read_text(encoding='utf-8'));training=comparison['training']
    reports={stage:json.loads((EVAL/(stage+'.json')).read_text(encoding='utf-8')) for stage in ('before','previous','after')}
    previous=reports['previous'];assert previous['model_sha256']==training['previous_mixed_weights_sha256']==sha(training['previous_mixed_weights'])
    assert previous['dataset_revision']==comparison['dataset_revision']
    fields=('input_shape','batch_size','precision','actual_model_dtype','confidence_threshold','nms_iou','global_merge_iou','match_iou','class_agnostic_nms','network_calls_per_frame','repeats','warmup_frames','gpu','cpu_threads','opencv_threads')
    assert all(previous['conditions'][k]==reports['after']['conditions'][k] for k in fields)
    assert len(previous['items'])==64 and len({r['parameters'] for r in reports.values()})==1
    index=json.loads((OUT/'index.json').read_text(encoding='utf-8'));changes=[]
    for e in index['validation']:
        key=e['source']+'/'+e['name'];row=previous['items'][key];truth=reports['after']['items'][key]['reference']
        assert row['reference']==truth and row['image_sha256']==e['image_sha256'] and row['label_sha256']==e['label_sha256']
        match=baseline.match_boxes(row['boxes'],truth,.5);assert all(match[k]==row['counts'][k] for k in ('tp','fp','fn'))
        assert all(c==row['counts'] for c in row['repeat_counts']) and all(t['total_ms']>0 and all(math.isfinite(v) for v in t.values()) for t in row['timings'])
        found_after=set(baseline.match_boxes(reports['after']['items'][key]['boxes'],truth,.5)['matched_truth']);found_previous=set(match['matched_truth'])
        changes.append({'key':key,'source':e['source'],'gained_references':sorted(found_after-found_previous),'lost_references':sorted(found_previous-found_after),'previous':row['counts'],'after':reports['after']['items'][key]['counts']})
    for source in ('sherbrooke','roe'):
        rows=[r for r in previous['items'].values() if r['source']==source];q=previous['sources'][source]['quality']
        assert all(sum(r['counts'][k] for r in rows)==q[k] for k in ('tp','fp','fn'))
        assert q['small_matched']==sum(r['small_matched'] for r in rows)
    frozen=json.loads((OUT/'protected_artifacts.json').read_text(encoding='utf-8'));assert all(sha(path)==value for path,value in frozen.items())
    fit=json.loads((EVAL/'training_fit_diagnostic.json').read_text(encoding='utf-8'));assert fit['dataset_revision']==index['revision']
    previous_fit=json.loads((ROOT/'data/mixed_pilot/evaluation/training_fit_diagnostic.json').read_text(encoding='utf-8'))
    assert all(fit['models']['after']['items'][e['name']]['image_sha256']==previous_fit['models']['after']['items'][e['name']]['image_sha256'] for e in index['roe_training'])
    comparison.update(previous={k:v for k,v in previous.items() if k!='items'},changes_from_previous_mixed=changes,training_fit={'current':fit['models']['after']['counts'],'previous':previous_fit['models']['after']['counts'],'scope':'Same27 training images, not validation'},source_exposure={'previous':{'sherbrooke':512,'roe':108},'current':{'sherbrooke':310,'roe':310},'entries_per_epoch':620,'epochs':10,'initial_weights_same':True,'other_training_options_equal_except_paths':True},model_policy='Original and mixed models retained. Focused weights are a separately published experimental candidate, not a verified deployment model.')
    waid.save_json(EVAL/'comparison.json',comparison)
    labels={'before':'Исходные веса','previous':'Прошлый опыт: 17% косуль','after':'Новый опыт: 50% косуль'}
    lines=['# Равная доля косуль и оленей: сравнение','', '2 ноября 2026, Москва. Фиксированные последние веса 10-й эпохи; best.pt не выбирался. Все три модели заново проверены одной программой на неизменённых 32+32 кадрах.', '', '| Источник | Модель | Найдено | Пропуски | Ложные рамки | Малые найдены | Медиана / p95, мс |','|---|---|---:|---:|---:|---:|---:|']
    for source,title in [('roe','Косули'),('sherbrooke','Олени')]:
        for stage in ('before','previous','after'):
            r=reports[stage]['sources'][source];q=r['quality'];t=r['timing_summary']['total_ms'];lines.append(f"| {title} | {labels[stage]} | {q['tp']}/{q['reference_boxes']} | {q['fn']} | {q['fp']} | {q['small_matched']}/{q['small_reference']} | {t['median']:.2f} / {t['p95']:.2f} |")
    lines+=['','## Что изменено в обучении','', 'Начальные веса, YOLO11n, вход 1024, batch 2, FP32, 10 эпох, AdamW lr 0,0003 и остальные настройки совпадают с первым смешанным опытом. Число предъявлений сохранено: 620 за эпоху, 310 пакетов по 2 кадра. Вместо 512 кадров оленей и 108 предъявлений косуль теперь по 310 предъявлений каждого источника.','', 'Для оленей заранее выбраны 155 положительных и 155 отрицательных учебных кадров, seed 20261102. Для косуль — прежние 27 кадров, каждый повторён 11 или 12 раз. Всего 337 уникальных изображений; повторение не создаёт независимых примеров. Доля источника и разнообразие используемого поднабора оленей меняются вместе; это не изолированное доказательство единственной причины результата. Новых изображений не скачано.',f"Время обучения {training['training_seconds']/60:.1f} мин; пиковая выделенная GPU-память {training['gpu_peak_allocated_bytes']/2**30:.2f} ГиБ. Исходные и прошлые веса, контрольные файлы и отчёты проверены по SHA256 и сохранены.",'', '## Усвоение учебных примеров','',f"На тех же 27 учебных кадрах косуль (40 рамок): прошлый опыт {previous_fit['models']['after']['counts']}, новый {fit['models']['after']['counts']}. tp — найденные, fp — ложные рамки, fn — пропуски. Это диагностика обучения; она не оценивает качество на незнакомых данных.",'', '## Условия измерения','']
    c=reports['after']['conditions'];lines.append(f"{c['gpu']}; CPU {c['cpu']}; RAM {c['ram_total_bytes']/2**30:.1f} ГиБ. Python {c['python']}, torch {c['torch']}, Ultralytics {c['ultralytics']}, CUDA {c['cuda']}. Вход 1×3×1024×1024, batch 1, FP32, TF32 выключен, CPU/OpenCV 4 потока. Порог уверенности 0,25; внутренний фильтр повторов IoU 0,7; дополнительный фильтр IoU 0,5; совпадение с разметкой один к одному при IoU ≥ 0,5. IoU — площадь пересечения двух рамок, делённая на площадь их объединения. Малые животные: площадь рамки <32² при полном входе 1024.")
    for stage,r in reports.items():
        m=r['memory'];lines.append(f"{labels[stage]}: файл {r['model_file_bytes']/2**20:.2f} МиБ, {r['parameters']} параметров; GPU allocated {m['gpu_peak_allocated_bytes']/2**20:.1f} МиБ, GPU reserved {m['gpu_peak_reserved_bytes']/2**20:.1f} МиБ, RAM процесса {m['process_rss_bytes']/2**20:.1f} МиБ. SHA256 `{r['model_sha256']}`.")
    lines+=['', 'Полная задержка включает чтение/декодирование, подготовку, сеть, внутреннюю фильтрацию, перенос/преобразование рамок и дополнительный фильтр повторов. Загрузка модели, первый холодный вызов, пять прогревов, чтение/сопоставление разметки, JSON и браузер исключены. Три повтора на кадр. Модели измерялись последовательно в отдельных процессах. Файловый кэш Windows, частоты и питание не фиксировались: малую разницу времени не трактуем как ускорение. Энергия кадра не измерялась; FPS будущего ускорителя 1 TOPS не прогнозируется.', '', '## Ограничения','', 'Контроль используется повторно для разработки. Независимость по полётам не подтверждена; алгоритм исключения известных связанных копий не гарантирует отсутствия других связей. Подсчитываются размеченные появления животных на отдельных снимках. Цель 90% наблюдений в видео и ложные события в минуту по этим данным не проверяются. Полнота авторской разметки принимается как рабочее допущение. Люди не входят в класс animal. Ноль ложных рамок при отсутствии всех ответов не означает хорошего результата.','', 'Просмотр: http://127.0.0.1:8765/?dataset=roe_focus&split=valid. Зелёные рамки — разметка, синие — новые веса. Старые подборки сохранены. Машинные отчёты: before.json, previous.json, after.json, comparison.json, training_fit_diagnostic.json. План до запуска: ../experiment_plan.json.']
    (EVAL/'COMPARISON.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    publication={'dataset_revision':index['revision'],'created_utc':datetime.now(timezone.utc).isoformat(),'stages':['before','previous','after'],'model_policy':comparison['model_policy']}
    for stage in publication['stages']:
        path=EVAL/(stage+'.json');publication[stage+'_path']=str(path.relative_to(ROOT));publication[stage+'_sha256']=sha(path)
    waid.save_json(EVAL/'published.json',publication)
    waid.save_json(EVAL/'verification.json',{'passed':True,'models':3,'same_control_files_and_inference_conditions':True,'quality_recomputed':True,'previous_models_reports_and_controls_unchanged':True,'training_entries_and_epochs_equal_previous':True,'training_checkpoint':'Fixed last epoch10','dataset_revision':index['revision']})
    print('Focused comparison validated and published',flush=True)

if __name__=='__main__':main()
