"""Add the completed pilot to project notes after HTTP verification."""
from pathlib import Path
import json
import waid
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'data/crop_pilot';EVAL=OUT/'evaluation'
def main():
    assert json.loads((EVAL/'viewer-verification.json').read_text(encoding='utf-8'))['passed']
    comparison=json.loads((EVAL/'comparison.json').read_text(encoding='utf-8'));reports={s:json.loads((EVAL/(s+'.json')).read_text(encoding='utf-8')) for s in ('before','after')}
    rows=comparison['rows'];lines=['<!-- crop-pilot-result -->','## Короткое обучение на фрагментах','',f"Три фиксированные эпохи ({comparison['training_seconds']/60:.1f} мин), без выбора лучшей по контролю.58 учебных кадров косуль →522 фрагмента512×512; прежние512 кадров оленей сохранены целиком. Начальные веса — основной расширенный кандидат. Сравнены четыре режима поиска на тех же64+16 отложенных кадрах.",'','| Проверка | Прежние веса, целиком | Прежние веса, фрагменты | Новые веса, целиком | Новые веса, фрагменты |','|---|---|---|---|---|']
    for split,source,label in [('control','roe','Прежние косули'),('control','sherbrooke','Прежние олени'),('fresh','roe','Новые отложенные косули')]:
        values=[]
        for stage,mode in [('before','full'),('before','tiles'),('after','full'),('after','tiles')]:
            q=reports[stage]['modes'][mode]['partitions'][split][source]['quality'];values.append(f"{q['tp']}/{q['reference_boxes']}, FP{q['fp']}")
        lines.append('| '+label+' | '+' | '.join(values)+' |')
    lines+=['','Поиск по фрагментам покрывает весь кадр, без подсказки разметки.9 вызовов сети на кадр косуль,6 на кадр оленей, вход1024, batch1, FP32. Замерена вся цепочка; подробные задержки, память и ограничения — в отчёте. Основная модель сохранена.','', 'Просмотр: http://127.0.0.1:8765/?dataset=crop_pilot&split=valid ; новые отложенные кадры: http://127.0.0.1:8765/?dataset=crop_pilot_fresh&split=diagnostic . Внутри — переключатель четырёх вариантов. Отчёт: `data/crop_pilot/evaluation/COMPARISON.md`.','<!-- end-crop-pilot-result -->','']
    p=ROOT/'README.md';s=p.read_text(encoding='utf-8');assert '<!-- crop-pilot-result -->' not in s
    s=s.replace('<!-- crop-lesson-prepared -->','\n'.join(lines)+'\n<!-- crop-lesson-prepared -->',1);p.write_text(s,encoding='utf-8')
    next_step={'crop_pilot_completed':True,'report':str(EVAL/'COMPARISON.md'),'main_candidate_promoted':False,'next_action':'Review measured held-out recall, false boxes and full latency. Choose one subsequent experiment from observed failure modes; do not increase training budget or complexity without evidence.','requirements_not_validated':['Independent video event recall90%','False confirmed events per10min','Energy per frame','Performance of future1TOPS board']}
    waid.save_json(OUT/'next_step.json',next_step)
    p=ROOT/'data/roe_expanded/next_step.json';data=json.loads(p.read_text(encoding='utf-8'));data.update({'crop_pilot_completed':True,'crop_pilot_report':str(EVAL/'COMPARISON.md'),'new_training_started':True,'crop_training_completed':True,'next_action':next_step['next_action']});waid.save_json(p,data)
    print('Completed pilot notes published')
if __name__=='__main__':main()
