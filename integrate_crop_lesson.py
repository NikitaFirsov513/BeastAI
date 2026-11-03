from pathlib import Path
import py_compile
root=Path(__file__).resolve().parent
p=root/'waid.py';s=p.read_text(encoding='utf-8')
anchor='    class Handler(BaseHTTPRequestHandler):'
assert s.count(anchor)==1 and 'crops=CropLessonPreview' not in s
s=s.replace(anchor,"    crops=None\n    if (ROOT/'data/crop_lesson/verification.json').is_file():\n        from crop_preview import CropLessonPreview\n        crops=CropLessonPreview(ROOT,parse_labels)\n        allowed_images.update(crops.allowed_images)\n        dataset_options.append({'id':'crop_lesson','label':'Фрагменты · подготовка обучения'})\n\n"+anchor)
old="collection=roe if requested=='roe' else"
assert s.count(old)==1
s=s.replace(old,"collection=crops if requested=='crop_lesson' else roe if requested=='roe' else")
p.write_text(s,encoding='utf-8')
p=root/'viewer.html';s=p.read_text(encoding='utf-8')
anchor="if(initialQuery.get('dataset')==='tiny_grid_shift')"
assert s.count(anchor)==1
s=s.replace(anchor,"if(initialQuery.get('dataset')==='crop_lesson'){const option=document.createElement('option');option.value='crop_lesson';option.textContent='Фрагменты · подготовка обучения';$('dataset').append(option);$('dataset').value='crop_lesson';$('split').value='train';relatedGroup='';}"+anchor)
s=s.replace("'tiny_memorization','tiny_grid_shift']","'tiny_memorization','tiny_grid_shift','crop_lesson']")
anchor=" if(!item.prediction)info.append"
assert s.count(anchor)==1
s=s.replace(anchor," if(item.caption)info.append(element('p','',item.caption));\n"+anchor)
p.write_text(s,encoding='utf-8')
for name in ('prepare_crop_lesson.py','crop_preview.py','waid.py'):py_compile.compile(str(root/name),doraise=True)
script=s.split('<script>')[1].split('</script>')[0]
(root/'data/crop_lesson/viewer-script-check.js').write_text(script,encoding='utf-8')
print('Viewer integration prepared; Python syntax checked')
