from project_paths import PROJECT_ROOT
from pathlib import Path
import json
root=PROJECT_ROOT
p=root/'data/roe_expanded/next_step.json';data=json.loads(p.read_text(encoding='utf-8'))
data['crop_lesson_prepared']=True;data['crop_lesson_report']=str(root/'data/crop_lesson/README.md');data['new_training_started']=False
data['next_action']='Review full-frame versus fixed-grid crop examples. Then run a bounded crop-learning experiment from the main expanded weights, not five-image memorization weights; evaluate unknown full frames with fixed grid and full-chain timing on frozen controls.'
p.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
p=root/"docs/EXPERIMENT_HISTORY.md";text=p.read_text(encoding='utf-8');marker='<!-- crop-lesson-prepared -->'
assert marker not in text
text=text.replace('<!-- tiny-diagnostic-result -->',marker+'\n## Учебный пример фрагментов\n\nИз пяти исходных учебных кадров подготовлены45 участков512 ×512 по фиксированной сетке;10 с целевыми рамками,35 без них. Все9 исходных животных целиком покрыты хотя бы одним участком. В просмотре12 картинок: пять пар «целый кадр → фрагмент» с входом1024 и два участка фона. Это подготовка, без нового обучения или оценки качества. Проверочные группы, модели и прежние результаты сохранены.\n\nПросмотр: http://127.0.0.1:8765/?dataset=crop_lesson&split=train . Описание: `data/crop_lesson/README.md`; проверка: `data/crop_lesson/viewer-verification.json`.\n<!-- end-crop-lesson-prepared -->\n\n<!-- tiny-diagnostic-result -->',1)
p.write_text(text,encoding='utf-8')
print('Lesson and next-step notes saved on D:')
