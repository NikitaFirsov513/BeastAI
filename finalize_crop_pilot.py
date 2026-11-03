"""Publish all prespecified pilot modes, without promoting a new main model."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
import waid,baseline
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'data/crop_pilot';EVAL=OUT/'evaluation'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    reports={stage:json.loads((EVAL/(stage+'.json')).read_text(encoding='utf-8')) for stage in ('before','after')}
    index=json.loads((OUT/'index.json').read_text(encoding='utf-8'));run=json.loads((OUT/'latest_run.json').read_text(encoding='utf-8'));plan=json.loads((OUT/'experiment_plan.json').read_text(encoding='utf-8'));inputs=json.loads((OUT/'input_verification.json').read_text(encoding='utf-8'))
    for stage,r in reports.items():
        assert r['dataset_revision']==index['revision']==run['dataset_revision']==plan['dataset_revision']
        key='base_weights' if stage=='before' else 'weights';assert r['model_sha256']==run[key+'_sha256']==sha(run[key]);assert r['quality_repeat_counts_stable']
        assert r['conditions']['batch_size']==1 and r['conditions']['repeats']==3
        for mode,data in r['modes'].items():
            assert len(data['items'])==80
            for e in index['validation']+index['fresh_validation']:
                row=data['items'][e['source']+'/'+e['name']];assert row['reference']==waid.parse_labels(Path(e['label_path']),['animal'])['boxes']
                assert row['counts']['tp']+row['counts']['fn']==e['reference_boxes']
                matched=baseline.match_boxes(row['boxes'],row['reference'],.5)
                assert row['counts']=={k:matched[k] for k in ('tp','fp','fn')}
                assert row['matched_reference_indices']==matched['matched_truth']
                assert row['search']['network_calls']==(1 if mode=='full' else 9 if e['source']=='roe' else 6)
    for split in ('control','fresh'):
        for source in reports['before']['modes']['full']['partitions'][split]:
            smalls=[r['modes'][mode]['partitions'][split][source]['quality']['small_reference'] for r in reports.values() for mode in ('full','tiles')];assert len(set(smalls))==1
    assert all(sha(p)==h for p,h in json.loads((OUT/'protected_artifacts.json').read_text(encoding='utf-8')).items())
    rows=[]
    for split,source,label in [('control','roe','Прежние косули'),('control','sherbrooke','Прежние олени'),('fresh','roe','Новые отложенные косули')]:
        for stage in ('before','after'):
            for mode in ('full','tiles'):
                r=reports[stage]['modes'][mode]['partitions'][split][source];q=r['quality'];t=r['timing_summary']['total_ms'];rows.append({'split':split,'source':source,'label':label,'stage':stage,'mode':mode,'quality':q,'total_ms':t})
    summary={'dataset_revision':index['revision'],'created_utc':datetime.now(timezone.utc).isoformat(),'rows':rows,'training_seconds':run['training_seconds'],'epochs':3,'main_candidate_promoted':False,'interpretation':'Short crop-learning pilot; all four prespecified modes are shown. Fresh16 is an additional development diagnostic, not an independent flight-level final test. Changes are not a one-factor causal estimate.'}
    waid.save_json(EVAL/'comparison.json',summary)
    lines=['# Короткое обучение на фрагментах','',f"Три эпохи, {run['training_seconds']/60:.1f} мин, {run['optimizer_steps']} обновлений весов. Начальные веса — основной расширенный кандидат, не игрушечные веса на пяти кадрах.",'','58 исходных учебных кадров косуль →522 фрагмента512×512 по сетке с шагом384. Добавлены прежние512 кадров оленей целиком:1034 входа за эпоху. Случайные преобразования отключены, AdamW lr0=0,0001. Вход1024, batch2, накопление до номинального batch8, FP32. Последняя фиксированная эпоха3, без выбора по контролю.','',f"Фрагментов с целевой разметкой: {inputs['positive_tiles']}, без неё: {inputs['background_tiles']}; {inputs['tile_label_appearances']} предъявлений рамок (повторы одного животного между фрагментами). Обрезанных рамок: {inputs['clipped_appearances']}. У {len(inputs['source_boxes_without_any_full_tile'])} исходных рамок нет полностью вмещающего их фрагмента; это ограничение крупного масштаба. Все пересечения сохранены с обрезанной разметкой.",'','| Проверка | Веса | Поиск | Найдено | Лишние рамки | Малые | Медиана, мс | p95, мс |','|---|---|---|---:|---:|---:|---:|---:|']
    for r in rows:
        q=r['quality'];t=r['total_ms'];lines.append(f"| {r['label']} | {'Прежние' if r['stage']=='before' else 'Новые'} | {'Целый кадр' if r['mode']=='full' else 'Фрагменты'} | {q['tp']}/{q['reference_boxes']} | {q['fp']} | {q['small_matched']}/{q['small_reference']} | {t['median']:.1f} | {t['p95']:.1f} |")
    lines+=['','Поиск по фрагментам обрабатывает весь исходный кадр по фиксированной сетке, без доступа к правильным рамкам. На кадр косуль9 последовательных вызовов сети, на кадр оленей6. Batch1, вход1024×1024, FP32/TF32 off, порог0,25, локальная NMS0,7, дополнительная NMS0,5, сопоставление IoU≥0,5. Малые цели закреплены по целому кадру1024. Полное время включает чтение/декодирование, выделение фрагментов, подготовку, все вызовы сети и объединение рамок. Три повтора, режимы чередуются; до/после — отдельные последовательные процессы. Частоты, температура и питание не зафиксированы, файл может быть в кэше ОС.','',f"GPU: {reports['after']['conditions']['gpu']}. CPU: {reports['after']['conditions']['cpu']}. Модель после обучения: {reports['after']['model_file_bytes']/2**20:.2f} МиБ, {reports['after']['parameters']} параметров. Память, версии библиотек и разбивка задержки — в before.json/after.json. Энергия кадра не измерена; скорость будущей платы1TOPS не прогнозируется.",'','Все80 снимков исключены из этого обучения. Контроль разработки многократно использован; независимость по полётам для косуль не подтверждена. Числа относятся к появлениям животных в кадрах, а не событиям наблюдения в видео. Цель90% наблюдений здесь не подтверждается. Основной кандидат сохранён.']
    (EVAL/'COMPARISON.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    waid.save_json(EVAL/'verification.json',{'passed':True,'all_80_frames_scored_in_all_four_modes':True,'matching_recomputed_against_source_labels':True,'fixed_small_membership':True,'full_frame_grid_call_counts_checked':True,'protected_artifacts_unchanged':True,'no_model_promotion':True,'checkpoint_preparation_verified':json.loads((OUT/'checkpoint_preparation.json').read_text(encoding='utf-8'))['passed']})
    waid.save_json(EVAL/'published.json',{'dataset_revision':index['revision'],'reports':{stage:{'path':str((EVAL/(stage+'.json')).relative_to(ROOT)),'sha256':sha(EVAL/(stage+'.json'))} for stage in ('before','after')},'active_variant':'after_tiles','main_candidate_promoted':False})
    waid.save_json(OUT/'status.json',{'state':'completed','epochs':3,'training_seconds':run['training_seconds'],'evaluation_images':80,'main_candidate_promoted':False})
    print(json.dumps({'rows':rows},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
