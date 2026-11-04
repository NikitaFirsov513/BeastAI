"""Validate and publish the fixed mixed experiment as a separate viewer collection."""
from project_paths import PROJECT_ROOT
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,math
import baseline,waid
ROOT=PROJECT_ROOT;OUT=ROOT/'data/mixed_pilot';EVAL=OUT/'evaluation'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    index=json.loads((OUT/'index.json').read_text(encoding='utf-8'))
    training=json.loads((OUT/'latest_run.json').read_text(encoding='utf-8'))
    plan=json.loads((OUT/'experiment_plan.json').read_text(encoding='utf-8'))
    reports={stage:json.loads((EVAL/(stage+'.json')).read_text(encoding='utf-8')) for stage in ('before','after')}
    assert training['training_options']['epochs']==10 and training['checkpoint_policy']==plan['checkpoint_policy']
    assert training['dataset_revision']==index['revision']==reports['before']['dataset_revision']==reports['after']['dataset_revision']
    assert reports['before']['model_sha256']==training['base_weights_sha256']
    assert reports['after']['model_sha256']==training['weights_sha256']
    assert Path(training['weights']).name=='last.pt'
    fields=('input_shape','batch_size','precision','actual_model_dtype','confidence_threshold','nms_iou','global_merge_iou','match_iou','class_agnostic_nms','network_calls_per_frame','repeats','warmup_frames','gpu','cpu_threads','opencv_threads')
    assert all(reports['before']['conditions'][f]==reports['after']['conditions'][f] for f in fields)
    assert reports['after']['conditions']['actual_model_dtype']=='torch.float32'
    frame_changes=[]
    for e in index['validation']:
        key=e['source']+'/'+e['name'];found={}
        assert sha(e['image_path'])==e['image_sha256'] and sha(e['label_path'])==e['label_sha256']
        truth=waid.parse_labels(Path(e['label_path']),['animal'])['boxes']
        for stage,r in reports.items():
            row=r['items'][key];assert row['reference']==truth and row['image_sha256']==e['image_sha256'] and row['label_sha256']==e['label_sha256']
            match=baseline.match_boxes(row['boxes'],truth,.5)
            assert all(match[k]==row['counts'][k] for k in ('tp','fp','fn'))
            assert all(c==row['counts'] for c in row['repeat_counts'])
            assert len(row['timings'])==3 and all(t['total_ms']>0 and all(math.isfinite(v) for v in t.values()) for t in row['timings'])
            found[stage]=set(match['matched_truth'])
        frame_changes.append({'key':key,'source':e['source'],'gained_references':sorted(found['after']-found['before']),'lost_references':sorted(found['before']-found['after']),'before':reports['before']['items'][key]['counts'],'after':reports['after']['items'][key]['counts']})
    for stage,r in reports.items():
        assert sha(ROOT/r['model_path'])==r['model_sha256']
        for source in ('sherbrooke','roe'):
            rows=[v for v in r['items'].values() if v['source']==source];q=r['sources'][source]['quality']
            assert all(sum(v['counts'][k] for v in rows)==q[k] for k in ('tp','fp','fn'))
            assert sum(v['small_matched'] for v in rows)==q['small_matched']
    comparison={'created_utc':datetime.now(timezone.utc).isoformat(),'dataset_revision':index['revision'],'training':training,'conditions':reports['after']['conditions'],'before':{k:v for k,v in reports['before'].items() if k!='items'},'after':{k:v for k,v in reports['after'].items() if k!='items'},'frame_changes':frame_changes,'quality_recomputed_from_predictions':True,'same_controls_and_inference_conditions':True,'limitations':reports['after']['limitations'],'model_policy':'Old active model remains available; mixed weights are a separately published candidate. No claim of final flight-independent or video-event validation.'}
    waid.save_json(EVAL/'comparison.json',comparison)
    lines=['# Смешанное дообучение: автоматическое сравнение','', '2 ноября 2026, время пользователя — Москва. Контрольные изображения и разметка сохранены. Использованы последние веса заранее заданной 10-й эпохи; best.pt не выбирался.', '', '| Источник | Веса | Найдено | Пропуски | Ложные рамки | Малые найдены | Медиана / p95, мс |','|---|---|---:|---:|---:|---:|---:|']
    for source,title in [('roe','Косули'),('sherbrooke','Олени')]:
        for stage,label in [('before','Прежние'),('after','После дообучения')]:
            r=reports[stage]['sources'][source];q=r['quality'];t=r['timing_summary']['total_ms']
            lines.append(f"| {title} | {label} | {q['tp']}/{q['reference_boxes']} | {q['fn']} | {q['fp']} | {q['small_matched']}/{q['small_reference']} | {t['median']:.2f} / {t['p95']:.2f} |")
    lines+=['','IoU — площадь пересечения рамок, делённая на площадь их объединения. Совпадение один к одному при IoU ≥ 0,5; уверенность ≥ 0,25. Малое животное: площадь рамки менее 32² пикселей при полнокадровом входе 1024. Люди не входят в целевой класс. Ноль ложных рамок у прежней модели на косулях связан с отсутствием ответов; precision там не определена.','', '## Обучение','',f"YOLO11n, начальные веса прежнего опыта с оленями. {training['training_seconds']/60:.1f} мин, FP32, batch 2, вход 1024, 10 эпох, AdamW lr 0,0003. 539 уникальных кадров: 512 прежних и 27 косуль. Каждый кадр косуль предъявляется четырежды за эпоху: 620 записей в учебном списке. Повторы не увеличивают число независимых примеров.",f"Пиковая выделенная GPU-память обучения: {training['gpu_peak_allocated_bytes']/2**30:.2f} ГиБ. Все исходные веса сохранены. Контроль: прежние 32 кадра оленей со 115 рамками и 32 кадра косуль с 44 рамками.", '', '## Условия использования и измерений','']
    c=reports['after']['conditions'];lines.append(f"{c['gpu']}; CPU {c['cpu']}; RAM {c['ram_total_bytes']/2**30:.1f} ГиБ; Python {c['python']}, torch {c['torch']}, Ultralytics {c['ultralytics']}, CUDA {c['cuda']}. Вход 1×3×1024×1024, batch 1, FP32, TF32 выключен, CPU/OpenCV 4 потока. Внутренний NMS IoU 0,7, дополнительный фильтр повторов IoU 0,5, одна сеть на кадр. Три повтора после пяти прогревочных запусков; две модели измерялись последовательно в отдельных процессах.")
    for stage,label in [('before','Прежние'),('after','Новые')]:
        r=reports[stage];m=r['memory'];lines.append(f"{label} веса: {r['model_file_bytes']/2**20:.2f} МиБ, {r['parameters']} параметров после объединения слоёв; пиковая GPU allocated {m['gpu_peak_allocated_bytes']/2**20:.1f} МиБ, GPU reserved {m['gpu_peak_reserved_bytes']/2**20:.1f} МиБ, RAM процесса {m['process_rss_bytes']/2**20:.1f} МиБ. SHA256: `{r['model_sha256']}`.")
    lines+=['','Полная задержка включает чтение, JPEG/PNG-декодирование, подготовку, сеть, внутреннюю фильтрацию, перенос и преобразование рамок на CPU, дополнительный фильтр повторов. Загрузка модели, холодный запуск, прогрев, чтение/сопоставление разметки, запись отчёта и браузер не входят. Возможен файловый кэш Windows. Питание и частоты GPU не фиксировались: небольшую разницу времени нельзя считать доказанным ускорением. Энергия кадра не измерялась; скорость ноутбука не переносится на будущий ускоритель 1 TOPS.','', '## Границы результата','', 'Это повторно используемый контроль разработки. Косули представлены небольшой выборкой; группы исходных полётов неизвестны. Найденные связанные копии исключены, но отсутствие связи не доказывает независимость. 90% обнаруженных наблюдений животных и ложные события в минуту по этим отдельным снимкам проверить нельзя. Разметка источников принята за эталон, её смысловая полнота не сертифицирована отдельно.','', 'В просмотрщике отдельная подборка «Смешанное дообучение — до/после»: http://127.0.0.1:8765/?dataset=mixed&split=valid. Зелёные рамки — эталон, синие — новые веса. Старые подборки и прежняя активная модель сохранены.','', 'Машинные результаты: before.json, after.json, comparison.json. Состав и причины исключения учебных кадров: ../index.json. Настройки до запуска: ../experiment_plan.json.']
    (EVAL/'COMPARISON.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    publication={'dataset_revision':index['revision'],'created_utc':datetime.now(timezone.utc).isoformat(),'model_policy':comparison['model_policy']}
    for stage in ('before','after'):
        path=EVAL/(stage+'.json');publication[stage+'_path']=str(path.relative_to(ROOT));publication[stage+'_sha256']=sha(path)
    waid.save_json(EVAL/'published.json',publication)
    waid.save_json(EVAL/'verification.json',{'passed':True,'same_controls_and_inference_conditions':True,'quality_recomputed':True,'weights_unchanged':True,'frame_changes':len(frame_changes),'training_epochs':10,'checkpoint':'last.pt','dataset_revision':index['revision']})
    print(json.dumps({stage:r['sources'] for stage,r in reports.items()}),flush=True)

if __name__=='__main__':main()
