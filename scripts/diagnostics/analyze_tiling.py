"""Analyze the frozen tiling experiment; optionally publish the inexpensive full-frame control."""
from project_paths import PROJECT_ROOT
from datetime import datetime,timezone
from pathlib import Path
import argparse,hashlib,json,shutil
import waid
from analyze_train512 import summarize
from tiled_inference import MODES,LABELS
ROOT=PROJECT_ROOT;DATA=ROOT/'data/sherbrooke';OUT=DATA/'tiling'


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--publish',action='store_true');args=parser.parse_args()
    waid.configure_dataset('sherbrooke');index=waid.get_index()
    run=json.loads((OUT/'latest_run.json').read_text(encoding='utf-8'))
    reports={mode:json.loads(Path(run['reports'][mode]).read_text(encoding='utf-8')) for mode in MODES}
    training=json.loads((DATA/'train512/latest_run.json').read_text(encoding='utf-8'))
    for pair in index['pairs']:
        for kind in ('image','label'):
            assert hashlib.sha256(waid.local_file(pair[kind]).read_bytes()).hexdigest()==pair[kind]['sha256']
    experiments=[];frames=[];fixed=('gpu','cpu','torch','ultralytics','cuda','model_dtype','batch_size','cpu_threads','confidence_threshold','nms_iou','class_agnostic_nms','match_iou','repeats','warmup_frames')
    for mode in MODES:
        report=reports[mode];c=report['conditions']
        assert report['model_sha256']==training['weights_sha256'] and report['dataset_revision']==index['revision']
        assert hashlib.sha256((ROOT/report['model_path']).read_bytes()).hexdigest()==report['model_sha256']
        assert set(report['items'])==set(reports['full']['items']) and len(report['items'])==96
        assert all(c[k]==reports['full']['conditions'][k] for k in fixed)
        assert c['model_dtype']=='torch.float32' and c['batch_size']==1 and c['confidence_threshold']==.25
        assert all(not v['tiling']['per_crop_cap_reached'] and not v['tiling']['global_cap_reached'] for v in report['items'].values())
        row,details=summarize(LABELS[mode],report,index)
        row.update(mode=mode,resolution=c['input_shape'][2],network_calls_per_frame=c['network_calls_per_frame'],source_crop_size=c['crop_size'],report_file=run['reports'][mode])
        assert row['reference_boxes']==115 and row['small_fixed_reference_at_1024']==81
        experiments.append(row);frames.extend(details)
    previous=json.loads((DATA/'train512/baseline/train512_1024.json').read_text(encoding='utf-8'))
    for k in ('tp','fp','fn','small_matched'):assert reports['full']['validation_preview'][k]==previous['validation_preview'][k]
    full,filtered,zoomed,native=experiments
    assert filtered['matched']==full['matched'] and filtered['unmatched_predictions']<full['unmatched_predictions']
    active='full_merge';diagnosis=json.loads((OUT/'scale_diagnosis.json').read_text(encoding='utf-8'))
    assert diagnosis['model_sha256']==training['weights_sha256'] and diagnosis['dataset_revision']==index['revision']
    limitations=index['selection']+[
        'The original 32 development frames and 115 reference appearances are used; no independent final test.',
        'All timings in this table are from one fresh run on the same 96 images, 3 repeats, rotating mode order. Do not mix with historical timings.',
        'Sequential batch 1. Tiled 640 runs four 640x640 inputs; enlarged tiles run four 1024x1024 inputs. Crops are 640x448 source pixels.',
        'Tile seams overlap by 256 horizontal and 213 vertical source pixels for 1024x683 frames. No source detail above the downloaded 1024px derivative is recovered.',
        'Tile detector NMS remains 0.7; an additional class-agnostic greedy overlap filter at 0.5 merges results. Full_merge applies that same additional filter after the normal detector.',
        'Small-object group remains the same 81 boxes at full-frame scale 1024, including when crop network input is 640.',
        'GPU peak allocated is reset per prediction; allocator reserved memory is shared across modes. CPU RSS is sampled outside timed prediction.',
        'Warm timing excludes model load, warmup, annotation parsing, metric calculation, JSON and browser; repeated reads may use OS cache.',
        'Energy per frame was not measured; laptop latency cannot be inferred for a future 1 TOPS board. Power and clock modes were not fixed.'
    ]
    selection='Рабочий режим: целый кадр с дополнительным фильтром повторов. На проверке сохранены все 83 найденных животных, ложные рамки уменьшились с 6 до 3, дополнительных запусков сети нет. Фрагменты без увеличения — кандидат для уменьшения пропусков с большей задержкой.'
    analysis={'experiment_kind':'tiling','dataset':index['title'],'dataset_revision':index['revision'],'source_urls':index['source_urls'],'validation_images':32,'validation_flights':[],'split_provenance':index['split_provenance'],'precision_available':True,'annotation_policy':index['annotation_coverage'],'conditions':reports['full']['conditions'],'training':training,'experiments':experiments,'frame_results':frames,'active_mode':active,'active_mode_label':LABELS[active],'active_model_sha256':training['weights_sha256'],'selection':selection,'geometry_verification':diagnosis['geometry_verification'],'scale_diagnosis':{k:{'input_size':v['input_size'],'counts':v['counts']} for k,v in diagnosis['modes'].items()},'limitations':limitations}
    waid.save_json(OUT/'analysis.json',analysis)
    lines=['# Обработка фрагментов: автоматический результат','',
        'Во всех режимах используются одни и те же веса YOLO11n после обучения на 512 снимках. Порог 0,25; совпадение с разметкой при IoU ≥ 0,5. Проверка: прежние 32 кадра и 115 появлений оленей; малые животные — прежняя группа из 81 рамки. Разметка и файлы не менялись.',
        '', '## Режимы',
        '- Целый кадр: вход сети 1024×1024, один запуск, прежний внутренний фильтр рамок IoU 0,7.',
        '- Целый кадр + фильтр повторов: та же выдача, затем дополнительное удаление перекрывающихся рамок с IoU > 0,5.',
        '- 4 увеличенных фрагмента: исходный кадр 1024×683 режется на четыре окна 640×448, каждое подаётся в сеть с входом 1024×1024. Увеличение животных примерно в 1,6 раза. Перекрытие окон 256 пикселей по горизонтали и 213 по вертикали.',
        '- 4 фрагмента без увеличения: те же окна, вход сети 640×640; размер животных в пикселях на входе сохраняется. Фрагменты обрабатываются последовательно, batch 1.',
        'В обоих режимах фрагментов рамки переводятся в координаты исходного кадра и объединяются тем же дополнительным фильтром IoU 0,5. Новые детали, которых нет в исходной копии 1024, не восстанавливаются.',
        '', '## Качество и полная задержка',
        '| Режим | Найдено / 115 | Пропущено | Ложных рамок | Малых / 81 | Вся цепочка, медиана / p95, мс | Пик тензоров GPU, МиБ |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for e in experiments:lines.append(f"| {e['label']} | {e['matched']} | {e['unmatched_reference']} | {e['unmatched_predictions']} | {e['small_fixed_matched']} | {e['median_total_ms']:.2f} / {e['p95_total_ms']:.2f} | {e['gpu_peak_allocated_MiB']:.1f} |")
    ratio=native['median_total_ms']/filtered['median_total_ms']
    lines+=['',selection,
        f"Фрагменты без увеличения добавили {native['matched']-filtered['matched']} найденных животных относительно быстрого режима, ложных рамок стало на {native['unmatched_predictions']-filtered['unmatched_predictions']} больше. Доля найденных: {filtered['recall']:.1%} → {native['recall']:.1%}; задержка выросла примерно в {ratio:.2f} раза. Малые животные: {filtered['small_fixed_matched']} → {native['small_fixed_matched']} из 81.",
        'Разница времени двух вариантов целого кадра мала относительно разброса измерений; ускорение от фильтра не утверждается. Выигрыш фильтра здесь — уменьшение ложных рамок.',
        '', '## Почему увеличение не помогло',
        'Проверена геометрия на реальных результатах: размеры исходных фрагментов и пиксели совпали; 445 сравнений IoU до и после переноса координат дали максимальную разницу около 1,9×10⁻¹⁵. Исходный плохой результат 14/115 воспроизведён.',
        'В дополнительном диагностическом прогоне: увеличенный целый кадр с входом 1632 дал 12 найденных, фрагменты с входом 640 — 89. Это подтверждает сильную чувствительность текущих весов к масштабу. Возможная причина — ограниченное разнообразие размеров животных в обучении; это гипотеза, а не установленная внутренняя причина поведения сети.',
        'Диагностика выполнялась одним проходом; время из неё не использовалось. Таблица выше содержит отдельные полноценные замеры всех четырёх режимов.',
        '', '## Этапы обработки, медианы в миллисекундах',
        '| Режим | Чтение/декодирование | Выделение фрагментов | Подготовка | Сеть, сумма запусков | Обработка выхода, сумма | Объединение рамок |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for e in experiments:
        t=e['stage_timing_ms'];keys=('read_decode_ms','slicing_ms','preprocess_ms','inference_ms','postprocess_ms','global_merge_ms')
        lines.append('| '+e['label']+' | '+' | '.join(f"{t[k]['median']:.2f}" for k in keys)+' |')
    c=reports['full']['conditions']
    lines+=['', '## Условия и границы измерения',
        f"NVIDIA GeForce RTX 4060 Laptop GPU; Intel Core i5-12450H; 16 ГБ RAM; Windows. PyTorch {c['torch']}, Ultralytics {c['ultralytics']}, CUDA {c['cuda']}. FP32, TF32 отключён, 4 потока CPU. Один вход сети за запуск, обработка фрагментов последовательная. Веса {full['model_file_MiB']:.2f} МиБ во всех режимах.",
        'Время измерено заново в одной сессии на прежних 96 кадрах, по три повтора. Порядок режимов меняется для каждого кадра и повтора. На прогрев каждого режима — пять полных вызовов. Качество считается только по 32 проверочным кадрам.',
        'В полное время входят чтение и декодирование, выделение фрагментов, подготовка, все вызовы сети, перенос и перевод рамок на CPU и их объединение. Загрузка модели, прогрев, чтение разметки, подсчёт качества, запись JSON и браузер исключены. Медианы этапов не обязаны складываться в медиану всей цепочки. Файловый кэш, тепловой режим и управление частотами влияют на абсолютные значения.',
        f"RSS всего Python-процесса: {min(e['process_rss_MiB'] for e in experiments):.0f}–{max(e['process_rss_MiB'] for e in experiments):.0f} МиБ. Пик выделенных GPU-тензоров измеряется отдельно для каждого вызова; память драйвера и других программ не включена. Зарезервированная память аллокатора общая между режимами и не считается независимой оценкой каждого режима.",
        'Лимиты 300 рамок не достигнуты. Энергия кадра не измерялась. Время ноутбука не переносится на ещё не выбранную бортовую плату по одной цифре TOPS.',
        '', '## Ограничения и следующий шаг',
        'Небольшая разработческая выборка одной оленьей фермы в Канаде; она не подтверждает качество над лесами Восточной Европы или на других видах. Это не отдельный итоговый test.',
        'Следующий выбор — допустимая задержка и пропуски. Для дальнейшего обучения полезно проверить более разнообразные масштабы животных, сохранив прежние отложенные данные. Новый большой детектор и квантование в этом опыте не вводились.',
        '', '## Артефакты',f"Веса: `{training['weights']}`",f"SHA256: `{training['weights_sha256']}`",f"Версия данных: `{index['revision']}`",f"Полные отчёты текущего прогона: `{run['run']}`",'Автоматический анализ: analysis.json. Диагностика масштаба и геометрии: scale_diagnosis.json. Исходный опыт обучения: ../train512/TRAINING_COMPARISON.md.',
        '', '## Источники',*index['source_urls'],'Лицензия: CC BY-NC 4.0; исследовательское и учебное применение.']
    (OUT/'TILING_COMPARISON.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    if args.publish:
        before=OUT/'before';before.mkdir(exist_ok=True)
        for source,name in [(DATA/'analysis.json','analysis.json'),(DATA/'baseline/latest.json','baseline.json')]:
            if not (before/name).exists():shutil.copy2(source,before/name)
        selected=dict(reports[active]);selected['model']='YOLO11n: 512 снимков, целый кадр с фильтром повторов'
        selected['training_dataset_revision']=training['dataset_revision']
        waid.save_json(DATA/'baseline/latest.json',selected);waid.save_json(DATA/'analysis.json',analysis)
        waid.save_json(OUT/'status.json',{'state':'completed','active_mode':active,'run':run['run'],'completed_utc':datetime.now(timezone.utc).isoformat()})
        readme=ROOT/"docs/EXPERIMENT_HISTORY.md";text=readme.read_text(encoding='utf-8')
        marker='<!-- tiling-result -->';end='<!-- end tiling-result -->'
        if marker in text:text=text[:text.index(marker)]+text[text.index(end)+len(end):].lstrip('\n')
        section=f'''{marker}
## Текущий режим: целый кадр с фильтром повторов

Проверка фрагментов завершена. Рабочий режим сохраняет 83/115 найденных животных и уменьшает ложные рамки с 6 до 3; новые веса не обучались. Обработка в текущем сравнении — {filtered['median_total_ms']:.1f} мс на полный кадр. Фрагменты без увеличения дают 89/115 и 4 ложные рамки, {native['median_total_ms']:.1f} мс; это отдельный вариант с большей задержкой. Увеличенные фрагменты ухудшили результат до 14/115.

Полный отчёт: `data/sherbrooke/tiling/TILING_COMPARISON.md`. Повторный замер из PowerShell в папке проекта: `.\\.venv\\Scripts\\python.exe tiled_inference.py`, затем `.\\.venv\\Scripts\\python.exe analyze_tiling.py --publish`. Первая команда сравнивает режимы на сохранённых весах, вторая создаёт отчёт и обновляет местный просмотрщик. Энергия кадра не измерялась. Ниже сохранена история обучения.
{end}

'''
        anchor='<!-- train512-result -->';assert anchor in text;text=text.replace(anchor,section+anchor,1);text=text.replace('## Текущий результат: обучение на 512 снимках','## Предыдущий этап: обучение на 512 снимках',1)
        readme.write_text(text,encoding='utf-8')
    print(json.dumps({'active_mode':active,'modes':[{k:e[k] for k in ('mode','matched','unmatched_reference','unmatched_predictions','small_fixed_matched','median_total_ms','p95_total_ms','gpu_peak_allocated_MiB')} for e in experiments]}),flush=True)

if __name__=='__main__':main()
