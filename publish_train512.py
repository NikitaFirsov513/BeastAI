"""Publish the completed 512-frame experiment to the local review viewer."""
from datetime import datetime,timezone
import hashlib,json,re
from pathlib import Path
import waid
ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data/sherbrooke';OUT=DATA/'train512'


def main():
    pilot=json.loads((DATA/'index.json').read_text(encoding='utf-8'))
    analysis=json.loads((OUT/'analysis.json').read_text(encoding='utf-8'))
    report=json.loads((OUT/'baseline/train512_1024.json').read_text(encoding='utf-8'))
    training=json.loads((OUT/'latest_run.json').read_text(encoding='utf-8'))
    assert analysis['dataset_revision']==report['dataset_revision']==pilot['revision']
    assert report['model_sha256']==training['weights_sha256']==analysis['experiments'][2]['model_sha256']
    assert hashlib.sha256(Path(training['weights']).read_bytes()).hexdigest()==training['weights_sha256']
    assert len(report['items'])==96 and report['validation_preview']['images']==32
    report['model']='YOLO11n: 512 учебных снимков, 20 эпох; класс animal — олень'
    report['training_dataset_revision']=training['dataset_revision']
    analysis['active_model_sha256']=report['model_sha256']
    waid.save_json(DATA/'baseline/latest.json',report)
    waid.save_json(DATA/'analysis.json',analysis)
    old,new=analysis['experiments'][1:]
    result={'completed_utc':datetime.now(timezone.utc).isoformat(),'training_images':512,'training_boxes':2342,'validation_images':32,'validation_boxes':115,'old':old,'new':new,'weights':training['weights'],'comparison_report':str(OUT/'TRAINING_COMPARISON.md'),'inference_conditions':report['conditions'],'limitations':analysis['limitations']}
    waid.save_json(OUT/'result.json',result)
    plan=json.loads((OUT/'experiment_plan.json').read_text(encoding='utf-8'));plan.update(status='completed',weights=training['weights'],weights_sha256=training['weights_sha256'],completed_utc=result['completed_utc'])
    waid.save_json(OUT/'experiment_plan.json',plan)
    verification=json.loads((OUT/'verification.json').read_text(encoding='utf-8'));verification.update(model_training_started=True,training_completed=True)
    waid.save_json(OUT/'verification.json',verification)
    folder_readme=f'''# Обучение на 512 снимках завершено

512 учебных кадров: 256 с оленями, 256 пустых; 2342 рамки. Все первые 64 учебных кадра включены. Проверка: те же 32 снимка и 115 рамок, файлы не менялись.

20 эпох, YOLO11n с исходными COCO-весами, 1024, batch 2, FP32. Обучение: {training['training_seconds']/60:.1f} мин. Для сравнения взяты веса после фиксированной последней эпохи.

На проверке: было найдено {old['matched']}/115, пропущено {old['unmatched_reference']}, ложных рамок {old['unmatched_predictions']}. Стало найдено {new['matched']}/115, пропущено {new['unmatched_reference']}, ложных рамок {new['unmatched_predictions']}. Малые животные: {old['small_fixed_matched']}/81 → {new['small_fixed_matched']}/81. Порог 0,25, IoU 0,5.

[Полный отчёт]({(OUT/'TRAINING_COMPARISON.md').as_posix()}). В нём скорость всей цепочки, этапы, память, условия измерений и ограничения.

Новые веса: `{training['weights']}`. SHA256: `{training['weights_sha256']}`.

Изображения и разметка хранятся отдельно в ../images и ../labels. Состав задают train.txt и valid.txt. Исходный dataset.yaml первого опыта закреплён на списках из 64/32 файлов. В просмотрщике сохранены прежние 96 кадров с новыми предсказаниями; качество считается только по 32 проверочным.

Источник: RGB-съёмка оленьей фермы в Канаде. CC BY-NC 4.0, исследовательский учебный опыт. Новые виды и места съёмки не добавлены. Проверочная часть использовалась для разработки; отдельного итогового test нет.

Протокол: latest_run.json. Результаты: result.json, analysis.json, baseline/*.json. Исходный опыт: before/. Повторный полный запуск из папки проекта: .venv/Scripts/python.exe run_train512.py.
'''
    (OUT/'README.md').write_text(folder_readme,encoding='utf-8')
    path=ROOT/'README.md';text=path.read_text(encoding='utf-8')
    text=re.sub(r'<!-- train512-result -->.*?<!-- end train512-result -->\s*','',text,flags=re.S)
    section=f'''<!-- train512-result -->
## Текущий результат: обучение на 512 снимках

Обучение завершено: 20 эпох, 512 учебных кадров, 2342 рамки. На прежних 32 проверочных снимках: найдено {new['matched']}/115 (ранее {old['matched']}), пропущено {new['unmatched_reference']} (ранее {old['unmatched_reference']}), ложных рамок {new['unmatched_predictions']} (ранее {old['unmatched_predictions']}). Малые животные: {new['small_fixed_matched']}/81 (ранее {old['small_fixed_matched']}). Порог и разрешение сохранены: 0,25 и 1024.

Полный отчёт — `data/sherbrooke/train512/TRAINING_COMPARISON.md`. Все три модели заново измерены на одних 96 кадрах: batch 1, FP32; качество считается только на valid. В просмотрщике — новые предсказания на прежних кадрах. Отчёт первого опыта сохранён в train512/before; старые веса не перезаписывались.

Повторный полный опыт из PowerShell в `D:\\PROJECTS\\PYTHON\\BeastAI`: `.\\.venv\\Scripts\\python.exe run_train512.py`. Команда обучает 20 эпох, сравнивает модели, создаёт отчёт и обновляет местный просмотрщик. Пересчёт отчёта из сохранённых предсказаний: `.\\.venv\\Scripts\\python.exe analyze_train512.py`.
<!-- end train512-result -->

'''
    marker='## Расширенная учебная подборка: 512 кадров'
    assert marker in text;text=text.replace(marker,section+marker,1)
    text=text.replace('Расширенная подборка готова; обучение на 512 кадрах ещё не запускалось.','Обучение на расширенной подборке завершено; результаты приведены выше.',1)
    text=text.replace('Сохранённый просмотрщик показывает результаты прежнего опыта.','Просмотрщик показывает новые предсказания; исходный опыт сохранён в train512/before.',1)
    text=text.replace('## Текущий опыт','## Первый опыт: 64 учебных снимка',1)
    text=text.replace('## Следующий опыт: порог обнаружения','## Архив: порог первого опыта',1)
    text=text.replace('Настройки следующего опыта сохранены','Настройки этого опыта сохранены',1)
    path.write_text(text,encoding='utf-8')
    waid.save_json(OUT/'experiment_status.json',{'state':'completed','updated_utc':result['completed_utc'],'weights':training['weights'],'report':str(OUT/'TRAINING_COMPARISON.md')})
    print(json.dumps({'matched':new['matched'],'missed':new['unmatched_reference'],'false':new['unmatched_predictions'],'small_matched':new['small_fixed_matched'],'median_total_ms':new['median_total_ms'],'weights':training['weights']}),flush=True)

if __name__=='__main__':main()
