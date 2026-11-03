"""Validate expanded data experiment, choose resolution on old controls, publish all results."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,math
import baseline,waid
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'data/roe_expanded';EVAL=OUT/'evaluation'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def choose_resolution():
    a=json.loads((EVAL/'after.json').read_text(encoding='utf-8'));b=json.loads((EVAL/'after_1280.json').read_text(encoding='utf-8'))
    nonworse=all(b['sources'][s]['quality']['tp']>=a['sources'][s]['quality']['tp'] and b['sources'][s]['quality']['fp']<=a['sources'][s]['quality']['fp'] for s in a['sources'])
    strict=any(b['sources'][s]['quality']['tp']>a['sources'][s]['quality']['tp'] or b['sources'][s]['quality']['fp']<a['sources'][s]['quality']['fp'] for s in a['sources'])
    latency=all(v['timing_summary']['total_ms']['p95']<=200 for v in b['sources'].values())
    return 'after_1280' if nonworse and strict and latency else 'after'

def main():
    from analyse_roe_expanded_errors import main as analyse_errors
    analyse_errors()
    index=json.loads((OUT/'index.json').read_text(encoding='utf-8'));run=json.loads((OUT/'latest_run.json').read_text(encoding='utf-8'));plan=json.loads((OUT/'experiment_plan.json').read_text(encoding='utf-8'))
    assert run['dataset_revision']==index['revision'] and run['training_options']['epochs']==20 and Path(run['weights']).name=='last.pt'
    reports={stage:json.loads((EVAL/(stage+'.json')).read_text(encoding='utf-8')) for stage in ('before','after','after_1280')}
    fresh={stage:json.loads((EVAL/('fresh_'+stage+'.json')).read_text(encoding='utf-8')) for stage in ('before','after') if (EVAL/('fresh_'+stage+'.json')).exists()}
    assert (set(fresh)=={'before','after'}) if index['fresh_validation'] else not fresh
    active=choose_resolution()
    if active=='after_1280' and index['fresh_validation']:
        fresh[active]=json.loads((EVAL/'fresh_after_1280.json').read_text(encoding='utf-8'))
    fields=('batch_size','precision','actual_model_dtype','confidence_threshold','nms_iou','global_merge_iou','match_iou','class_agnostic_nms','network_calls_per_frame','repeats','warmup_frames','gpu','cpu_threads','opencv_threads')
    expected_controls={e['source']+'/'+e['name']:e for e in index['validation']};expected_fresh={e['source']+'/'+e['name']:e for e in index['fresh_validation']}
    for collection,expected in ((reports,expected_controls),(fresh,expected_fresh)):
        for stage,r in collection.items():
            assert r['dataset_revision']==index['revision'] and set(r['items'])==set(expected)
            side=1280 if stage=='after_1280' else 1024;assert r['conditions']['input_shape']==[1,3,side,side]
            assert all(r['conditions'][k]==reports['after']['conditions'][k] for k in fields)
            expected_model=run.get('comparison_weights_sha256',run['base_weights_sha256']) if stage=='before' else run['weights_sha256'];assert r['model_sha256']==expected_model==sha(ROOT/r['model_path'])
            for key,e in expected.items():
                assert sha(e['image_path'])==e['image_sha256'] and sha(e['label_path'])==e['label_sha256'];row=r['items'][key];truth=waid.parse_labels(Path(e['label_path']),['animal'])['boxes']
                assert row['reference']==truth and row['image_sha256']==e['image_sha256'] and row['label_sha256']==e['label_sha256']
                match=baseline.match_boxes(row['boxes'],truth,.5);assert all(match[k]==row['counts'][k] for k in ('tp','fp','fn'))
                assert all(c==row['counts'] for c in row['repeat_counts']) and len(row['timings'])==3
                assert all(t['total_ms']>0 and all(math.isfinite(v) for v in t.values()) for t in row['timings'])
            for source,s in r['sources'].items():
                rows=[row for row in r['items'].values() if row['source']==source];q=s['quality'];assert all(sum(row['counts'][k] for row in rows)==q[k] for k in ('tp','fp','fn'))
                assert q['small_reference']==sum(row['small_reference'] for row in rows) and q['small_matched']==sum(row['small_matched'] for row in rows)
    for source in reports['after']['sources']:assert reports['after']['sources'][source]['quality']['small_reference']==reports['after_1280']['sources'][source]['quality']['small_reference']
    assert len({r['parameters'] for r in reports.values()})==1
    assert {e['visual_group'] for e in index['roe_training']}.isdisjoint(e['visual_group'] for e in index['fresh_validation'])
    frozen=json.loads((OUT/'protected_artifacts.json').read_text(encoding='utf-8'));assert all(sha(p)==value for p,value in frozen.items())
    fit=json.loads((EVAL/'training_fit_diagnostic.json').read_text(encoding='utf-8'));assert fit['dataset_revision']==index['revision']
    resolution_policy={'selected_stage':active,'selection_used':'Only old64 developmental controls, never fresh groups.','rule':'Choose1280 only if TP does not decrease and FP does not increase on either source, at least one strict improvement, and p95 full latency <=200ms on this laptop. Otherwise retain1024.','scope':'Laptop latency bound is not accelerator validation; power/energy not measured.'}
    error_analysis=json.loads((EVAL/'error_analysis.json').read_text(encoding='utf-8'))
    data_profile=json.loads((OUT/'data_profile.json').read_text(encoding='utf-8')) if (OUT/'data_profile.json').exists() else None
    comparison={'created_utc':datetime.now(timezone.utc).isoformat(),'dataset_revision':index['revision'],'training':run,'plan':plan,'data_profile':data_profile,'error_summary':{name:r['sources'] for name,r in error_analysis['reports'].items()},'controls':{k:{field:value for field,value in r.items() if field!='items'} for k,r in reports.items()},'fresh':{k:{field:value for field,value in r.items() if field!='items'} for k,r in fresh.items()},'training_fit':{stage:{k:v for k,v in r.items() if k!='items'} for stage,r in fit['models'].items()},'resolution_policy':resolution_policy,'validation_scope':index['validation_scope'],'new_fresh_scope':index['reserved_fresh_policy'],'previous_artifacts_and_control_files_unchanged':True,'quality_recomputed':True,'model_policy':'New weights are a separately published candidate; all older versions retained.'}
    waid.save_json(EVAL/'comparison.json',comparison);waid.save_json(EVAL/'resolution_selection.json',resolution_policy)
    names={'before':'Прежние веса, 27 кадров косуль','after':'Больше данных, вход1024','after_1280':'Больше данных, вход1280'}
    lines=['# Расширенный набор: обучение и измерения','', 'Дата пользователя: 2 ноября 2026, Москва. Все файлы находятся в D:\\PROJECTS\\PYTHON\\BeastAI.','', '## Прежние контрольные кадры','', '| Источник | Вариант | Найдено | Пропущено | Ложные рамки | Малые найдены | Медиана / p95, мс |','|---|---|---:|---:|---:|---:|---:|']
    def table(collection):
        result=[]
        for source,title in [('roe','Косули'),('sherbrooke','Олени')]:
            for stage,r in collection.items():
                if source not in r['sources']:continue
                s=r['sources'][source];q=s['quality'];t=s['timing_summary']['total_ms'];result.append(f"| {title} | {names[stage]} | {q['tp']}/{q['reference_boxes']} | {q['fn']} | {q['fp']} | {q['small_matched']}/{q['small_reference']} | {t['median']:.2f} / {t['p95']:.2f} |")
        return result
    lines+=table(reports)
    lines+=['','## Новые отложенные группы','',f"До обучения выделено {len(index['fresh_validation'])} новых кадров из отдельных найденных визуальных компонентов. Они не входят в учебный список; выбор не использует предсказания модели. Это дополнительная диагностика, а не подтверждённая независимость по полётам и не репрезентативная частота животных в полёте.",'', '| Источник | Вариант | Найдено | Пропущено | Ложные рамки | Малые найдены | Медиана / p95, мс |','|---|---|---:|---:|---:|---:|---:|']+table(fresh)
    lines+=['','## Данные и обучение','',f"Извлечено512 новых кандидатов ограниченными диапазонами ZIP, без скачивания полного архива25ГБ. Геометрическое сопоставление исключает известные связанные с контролем компоненты; в обучении по одному представителю компонента. После очистки и откладывания новых групп: {run['unique_roe_images']} уникальных кадров косуль, 512 кадров оленей. За эпоху: {run['entries_per_epoch']} предъявлений, по {run['source_entries']['roe']} от каждого источника. Повторы не являются новыми независимыми примерами.",f"Начальные веса зафиксированы в плане; при новых связях прежних учебных кадров с контролем использованы веса только с оленями. SHA256 `{run['base_weights_sha256']}`. 20 фиксированных эпох, последние веса last.pt, вход1024, batch2, FP32, AdamW lr0,0003; короткий прогрев1эпоха, начальная скорость bias0,0003. Время обучения {run['training_seconds']/60:.1f} мин, пиковая GPU allocated {run['gpu_peak_allocated_bytes']/2**30:.2f} ГиБ. Изменились данные, начальные веса, длительность и прогрев; результат нельзя приписать исключительно количеству изображений."]
    lines+=['', 'В сравнении контроля прежняя модель — кандидат с27 учебными кадрами косуль. В диагностике усвоения `before` обозначает начальные веса нового обучения, обученные только на оленях; пути и SHA256 сохранены в `training_fit_diagnostic.json`.']
    for stage,r in fit['models'].items():lines.append(f"Усвоение учебных косуль ({stage}): {r['images']} кадров, {r['reference_boxes']} рамок, {r['counts']}. tp — найденные, fp — ложные рамки, fn — пропуски. Это не качество на новых данных.")
    if index.get('legacy_selected_now_connected_to_control',0):
        lines+=['',f"Расширенный поиск связал {index['legacy_selected_now_connected_to_control']} прежних учебных представителей с контрольными компонентами. Из новых учебных списков они исключены. Из-за этих новых связей для нового обучения выбраны веса, обученные только на оленях. Прежний контроль уже многократно использовался при разработке и остаётся диагностическим; новые отложенные компоненты отбирались отдельно."]
    lines+=['','## Условия измерений','']
    c=reports['after']['conditions'];lines.append(f"{c['gpu']}, CPU {c['cpu']}, RAM {c['ram_total_bytes']/2**30:.1f} ГиБ; Python {c['python']}, torch {c['torch']}, Ultralytics {c['ultralytics']}, CUDA {c['cuda']}. По1 кадру, FP32, TF32 выключен, CPU/OpenCV4 потока. Порог0,25, внутренний NMS IoU0,7, дополнительный фильтр IoU0,5. Совпадение с эталоном один к одному при IoU≥0,5. IoU — площадь пересечения рамок, делённая на площадь объединения. Малые объекты закреплены по площади<32² при входе1024, группа не меняется при входе1280.")
    for stage,r in reports.items():
        m=r['memory'];lines.append(f"{names[stage]}: {r['model_file_bytes']/2**20:.2f} МиБ, {r['parameters']} параметров; GPU allocated {m['gpu_peak_allocated_bytes']/2**20:.1f} МиБ, GPU reserved {m['gpu_peak_reserved_bytes']/2**20:.1f} МиБ, RAM процесса {m['process_rss_bytes']/2**20:.1f} МиБ. SHA256 `{r['model_sha256']}`.")
    lines+=['', 'Полная задержка: чтение/декодирование, подготовка, сеть, внутренняя фильтрация, перенос/преобразование рамок, дополнительная фильтрация. Три повтора после5 прогревов. Загрузка модели, холодный вызов, прогрев, чтение/сопоставление разметки, JSON и браузер исключены. Модели измерялись в отдельных последовательных процессах; частоты/питание и файловый кэш Windows не фиксировались. Небольшая разница времени не доказывает ускорение. Энергия кадра не измерялась; производительность будущего ускорителя1TOPS не прогнозируется.','', '## Выбор разрешения','', f"В просмотрщике выбран `{active}`. Правило использовало только прежние64 контрольных кадра: вход1280 выбирается при отсутствии ухудшений TP/FP на обоих источниках, хотя бы одном строгом улучшении и p95≤200мс на ноутбуке; иначе остаётся1024. Новые отложенные группы не использовались для выбора. Более высокое разрешение не объявляется безусловно лучшим.",'', '## Ограничения','', 'Визуальные компоненты не равны ID полётов. Отсутствие найденной связи не доказывает независимость. Старый контроль повторно используется для разработки; свежая диагностика того же источника тоже не является проверенным независимым итоговым тестом. Рамки авторов приняты за рабочий эталон, их смысловая полнота отдельно не сертифицирована. Подсчитываются появления животных на снимках. Цель90% наблюдений в видео, ложные события в минуту, трекинг и карта здесь не проверяются.','', 'Лицензии и авторство: косули — Zenodo18206908, CC BY4.0; олени Sherbrooke — условия CC BY-NC4.0. Исходные метаданные сохранены в соответствующих папках.','', 'Просмотр прежнего контроля: http://127.0.0.1:8765/?dataset=roe_expanded&split=valid ; новых отложенных кадров: http://127.0.0.1:8765/?dataset=roe_expanded_fresh&split=diagnostic ; учебных косуль: http://127.0.0.1:8765/?dataset=roe_expanded_train&split=train . Все прежние подборки сохранены.']
    if data_profile and data_profile['partitions']['fresh_diagnostic'].get('roe'):
        p=data_profile['partitions']['fresh_diagnostic']['roe'];lines+=['','## Размеры животных в данных','',f"Новые отложенные кадры: медианная рамка при входе1024 — {p['median_box_width_at1024']:.1f}×{p['median_box_height_at1024']:.1f} пикселя; распределение площадей {p['size_groups_at1024']}. Это намеренно непростая диагностика, а не типичная частота объектов в полёте. Детальный профиль обучения и контроля: `../data_profile.json`. Увеличение уже уменьшенного изображения само по себе не восстанавливает отсутствующие детали."]
    lines+=['','## Дополнительный разбор ошибок','', 'В `error_analysis.json` выданные ложные рамки разделены по геометрическим признакам: повтор к уже сопоставленной рамке, совпадение с размеченным человеком (только источник косуль), частичное пересечение животного при IoU<0,5, отсутствие пересечения целевых рамок. Последнее не доказывает фон: полнота разметки не сертифицирована. Пропуски разделены на отсутствие выданной рамки с пересечением, недостаточное пересечение и конфликт сопоставления один к одному. Анализ не объясняет причины подавления низкоуверенных ответов.']
    (EVAL/'COMPARISON.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    publication={'dataset_revision':index['revision'],'active_stage':active,'reports':{'control':{},'fresh':{}},'model_policy':comparison['model_policy']}
    for mode,collection,prefix in [('control',reports,''),('fresh',fresh,'fresh_')]:
        for stage in collection:
            path=EVAL/(prefix+stage+'.json');publication['reports'][mode][stage]={'path':str(path.relative_to(ROOT)),'sha256':sha(path)}
    waid.save_json(EVAL/'published.json',publication);waid.save_json(EVAL/'verification.json',{'passed':True,'dataset_revision':index['revision'],'control_images':64,'fresh_images':len(index['fresh_validation']),'quality_recomputed':True,'same_control_files':True,'small_cohort_fixed_at1024':True,'previous_artifacts_unchanged':True,'fresh_not_used_for_resolution_selection':True,'active_stage':active})
    print('Expanded experiment published: '+active,flush=True)

if __name__=='__main__':main()
