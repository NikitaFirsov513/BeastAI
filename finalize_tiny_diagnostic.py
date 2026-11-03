"""Verify the paired fixed-input diagnostic and write a compact report."""
from pathlib import Path
import hashlib,json,csv
ROOT=Path(__file__).resolve().parent;A=ROOT/'data/tiny_memorization';B=ROOT/'data/tiny_grid_shift'

def main():
    original=json.loads((A/'comparison.json').read_text(encoding='utf-8'));shifted=json.loads((B/'comparison.json').read_text(encoding='utf-8'));pa=json.loads((A/'experiment_plan.json').read_text(encoding='utf-8'));pb=json.loads((B/'experiment_plan.json').read_text(encoding='utf-8'))
    assert pa['base_weights_sha256']==pb['base_weights_sha256']=='cb33d9a97d08d53f11ca665c49659f1d8d8f869e620b81775abcbfed7f1c27ef'
    assert all(v==pb['training_options'][k] for k,v in pa['training_options'].items() if k not in ('data','project','name'))
    ia=json.loads((A/'index.json').read_text(encoding='utf-8'));ib=json.loads((B/'index.json').read_text(encoding='utf-8'));byname={e['name']:e for e in ia['entries']}
    for e in ib['entries']:
        old=byname[e['name']];assert e['source_name']==old['source_name'] and e['reference_boxes']==old['reference_boxes']
        if e['fixed_integer_shift']==[0,0]:
            assert e['image_sha256']==old['image_sha256']
            assert Path(e['label_path']).read_text().split()==Path(old['label_path']).read_text().split()
    for folder in (A,B):
        run=json.loads((folder/'latest_run.json').read_text(encoding='utf-8'));results=Path(run['weights']).parents[1]/'results.csv'
        with results.open(encoding='utf-8',newline='') as stream:rows=list(csv.DictReader(stream))
        assert len(rows)==200
        frozen=json.loads((folder/'protected_artifacts.json').read_text(encoding='utf-8'));assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==sha for p,sha in frozen.items())
    comparison={'scope':'Paired memorization on the same five TRAINING inputs; not held-out quality.','same_initial_weights':True,'same_training_options_except_paths':True,'epochs_each':200,'batches_per_epoch':1,'augmentation_disabled':True,'original_zero_candidate_group':{'references':5,'unshifted_found':original['after']['groups']['tiny_zero_candidates']['matched'],'shifted_found':shifted['after']['groups']['tiny_zero_candidates']['matched']},'original_supported_group':{'references':3,'unshifted_found':original['after']['groups']['tiny_with_candidates']['matched'],'shifted_found':shifted['after']['groups']['tiny_with_candidates']['matched']},'large_control':{'references':1,'unshifted_found':original['after']['groups']['large_control']['matched'],'shifted_found':shifted['after']['groups']['large_control']['matched']},'shifts':{e['source_name']:e['fixed_integer_shift'] for e in ib['entries']},'training_seconds':{'unshifted':original['training_seconds'],'shifted':shifted['training_seconds']},'original_models_and_holdouts_unchanged':True,'label_comparison_policy':'Compare numeric content; Windows CRLF versus LF serialization is not a label-coordinate change.','limits':['Geometric eligibility is necessary, not sufficient, for actual positive assignment.','Earlier random augmentation can change anchor eligibility; static counts do not describe every earlier epoch.','Memorization on five inputs does not estimate recognition on unseen animals.','A universal1px shift is not an operational fix.']}
    assert comparison['original_zero_candidate_group']=={'references':5,'unshifted_found':0,'shifted_found':5}
    assert comparison['original_supported_group']=={'references':3,'unshifted_found':3,'shifted_found':3}
    for folder in (A,B):(folder/'paired_comparison.json').write_text(json.dumps(comparison,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    report=f'''# Диагностика усвоения маленьких животных

Только учебные данные:4 кадра с8 маленькими целями и1 кадр с1 крупной контрольной. Прежние64 контрольных кадра и16 новых отложенных кадров не использовались в обучении.

| Группа целей | Без сдвига после обучения | После согласованного сдвига |
|---|---:|---:|
|5 маленьких целей без допустимых точек в исходном положении|0/5|5/5|
|3 маленькие цели с допустимыми точками|3/3|3/3|
|Крупная контрольная цель|1/1|1/1|
|Всего|4/9|9/9|

До обоих опытов исходная модель находила1/9 (крупную цель) и давала1 лишнюю рамку. После каждого опыта лишних рамок на этих же5 снимках0. Это усвоение знакомых примеров, не проверочное качество.

## Постановка и проверки

Начальные веса одинаковые: SHA256 `{pa['base_weights_sha256']}`. По200 эпох, один пакет из5 изображений за эпоху; по200 строк в results.csv. Параметры обучения совпадают, кроме путей. Вход1024, batch5, nbs5, FP32, AdamW lr0,001, без weight decay и прогрева. Случайные преобразования отключены собственным загрузчиком: только LetterBox+Format. Используются последние веса200-й эпохи, не best.pt.

Для совпадения пикселей обучения и использования подготовлены lossless PNG1024×1024, точно совпадающие со стандартным преобразованием исходников1280×1280. Исходные JPEG и разметка не менялись. Проверены повторяемость тензоров, число классов/рамок, нормированные координаты и соответствие RGB-пикселей. Различие LF/CRLF в копиях текстовых файлов не меняет числовую разметку.

Штатная TaskAlignedAssigner.select_candidates_in_gts допускает для положительного назначения только опорные точки строго внутри рамки. На реальных преобразованных входах при шагах8/16/32 у5 из8 маленьких целей допустимых точек нет. У трёх других маленьких целей и крупной есть. Допустимые точки — необходимое, но не достаточное условие назначения.

В парном опыте03650_1.jpg сдвинут на(-1,-1),07152_1.jpg на(-1,+1); остальные три кадра не двигались. Изображение и рамки сдвигались вместе. Сдвиг целочисленный без интерполяции: пиксели, размеры животных и полный охват целей сохранены, новые края заполнены значением114. Теперь у всех целей есть допустимые точки. Группы в таблице закреплены по исходному положению.

Первое обучение заняло{original['training_seconds']:.1f}с, второе{shifted['training_seconds']:.1f}с. Разница не является доказанным ускорением от сдвига: частоты, питание и кэши не фиксировались. Оценка: тот же вход1024, batch1, FP32, порог0,25, IoU сопоставления0,5, дополнительный фильтр повторов0,5; по3 повтора после прогрева. В before.json/after.json сохранены полная задержка чтения/декодирования, подготовки, сети и рамок, память и оборудование; загрузка модели, прогрев и подсчёт по эталону исключены.

## Ограничения и следующий шаг

Выявлена конкретная проблема назначения учебных ответов в данном фиксированном опыте. Ранее случайные переносы и масштабирование могли менять положение рамок: нельзя утверждать, что эти цели никогда не получали поддержки в прежнем обучении.9/9 показывает запоминание пяти входов, а не распознавание новых животных. Постоянный сдвиг на1пиксель не является исправлением для любого изображения.

Диагностические веса не заменяют основной детектор. Основные модели и контрольные файлы проверены по SHA256. Следующий отдельный шаг — подготовить учебные фрагменты с достаточно крупными животными, затем оценить настоящий поиск по целому неизвестному кадру на тех же отложенных данных, с учётом пропусков, ложных рамок, памяти и полной задержки. Вырезки по эталону для оценки поиска не подходят.

Просмотр исходного опыта: http://127.0.0.1:8765/?dataset=tiny_memorization&split=train
Просмотр парного опыта: http://127.0.0.1:8765/?dataset=tiny_grid_shift&split=train

Все файлы: D:\\PROJECTS\\PYTHON\\BeastAI.
'''
    for folder in (A,B):(folder/'REPORT.md').write_text(report,encoding='utf-8')
    section='''<!-- tiny-diagnostic-result -->
## Диагностика маленьких целей —2 ноября2026

На5 учебных кадрах проверено усвоение8 маленьких и1 крупной цели. У5 маленьких рамок в фиксированном входе нет допустимых опорных точек штатного назначателя YOLO. После200 коротких проходов выучены3 поддержанные маленькие цели и крупная (4/9);5 неподдержанных пропущены.

Парный опыт с теми же начальными весами и параметрами после согласованного сдвига2 изображений и рамок на1пиксель по осям дал9/9. Размеры и пиксели животных сохранены. Это усвоение знакомых примеров, не качество на новых кадрах и не универсальное исправление сдвигом. Прежние случайные преобразования могли менять поддержку целей.

Основной детектор не заменён. Отчёт: `data/tiny_memorization/REPORT.md`; парное сравнение: `data/tiny_memorization/paired_comparison.json`. Просмотр: http://127.0.0.1:8765/?dataset=tiny_memorization&split=train и http://127.0.0.1:8765/?dataset=tiny_grid_shift&split=train . Следующий отдельный шаг — обучение на подходящих фрагментах с последующей проверкой настоящего поиска по кадру.
<!-- end-tiny-diagnostic-result -->

'''
    p=ROOT/'README.md';text=p.read_text(encoding='utf-8');assert '<!-- expanded-result -->' in text
    if '<!-- tiny-diagnostic-result -->' not in text:text=text.replace('<!-- expanded-result -->',section+'<!-- expanded-result -->',1)
    p.write_text(text,encoding='utf-8')
    p=ROOT/'data/roe_expanded/next_step.json';next_step=json.loads(p.read_text(encoding='utf-8'));next_step.update(tiny_memorization_diagnostic_completed=True,tiny_diagnostic_report=str(A/'REPORT.md'),next_action='Explain and prepare crop-based training with reliable target support; validate unknown-frame search on frozen controls with latency/memory. A fixed1px shift is not an operational fix.',new_training_started=False);p.write_text(json.dumps(next_step,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Paired diagnostic verified: 4/9 versus9/9; primary detector and held-out data unchanged')

if __name__=='__main__':main()
