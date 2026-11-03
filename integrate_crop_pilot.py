from pathlib import Path
import py_compile
root=Path(__file__).resolve().parent
p=root/'waid.py';s=p.read_text(encoding='utf-8')
anchor='    class Handler(BaseHTTPRequestHandler):'
assert s.count(anchor)==1 and 'crop_pilots={}' not in s
s=s.replace(anchor,"    crop_pilots={}\n    if (ROOT/'data/crop_pilot/evaluation/published.json').is_file():\n        from crop_pilot_preview import CropPilotPreview\n        for crop_split in ('control','fresh'):\n            collection=CropPilotPreview(ROOT,parse_labels,crop_split)\n            crop_pilots[collection.dataset_id]=collection\n            allowed_images.update(collection.allowed_images)\n            dataset_options.append({'id':collection.dataset_id,'label':collection.title})\n\n"+anchor)
anchor="collection=crops if requested=='crop_lesson' else"
assert s.count(anchor)==1
s=s.replace(anchor,"collection=crop_pilots[requested] if requested in crop_pilots else crops if requested=='crop_lesson' else")
p.write_text(s,encoding='utf-8')
p=root/'viewer.html';s=p.read_text(encoding='utf-8')
anchor=' <label><input id="boxes"'
assert s.count(anchor)==1
s=s.replace(anchor,' <label id="variantpicker" hidden>Режим <select id="variant" aria-label="Вариант опыта"><option value="after_tiles">После обучения · фрагменты</option></select></label>\n'+anchor)
s=s.replace('</style>','#variantpicker[hidden]{display:none}\n</style>')
anchor="if(initialQuery.get('dataset')==='crop_lesson')"
assert s.count(anchor)==1
s=s.replace(anchor,"if(['crop_pilot','crop_pilot_fresh'].includes(initialQuery.get('dataset'))){const id=initialQuery.get('dataset'),option=document.createElement('option');option.value=id;option.textContent='Обучение на фрагментах';$('dataset').append(option);$('dataset').value=id;$('split').value=id==='crop_pilot_fresh'?'diagnostic':'valid';relatedGroup='';}if(initialQuery.get('variant')){const option=document.createElement('option');option.value=initialQuery.get('variant');option.textContent='Выбранный режим';$('variant').append(option);$('variant').value=option.value;}"+anchor)
anchor="group:relatedGroup,page,split:"
assert s.count(anchor)==1
s=s.replace(anchor,"group:relatedGroup,page,variant:$('variant').value,split:")
anchor="  const selectedCategory=$('category').value;"
assert s.count(anchor)==1
s=s.replace(anchor,anchor+"\n  $('variantpicker').hidden=!data.variant_options;if(data.variant_options){$('variant').replaceChildren(...data.variant_options.map(o=>{const option=element('option','',o.label);option.value=o.id;return option}));$('variant').value=data.variant;}")
s=s.replace("'tiny_grid_shift','crop_lesson']","'tiny_grid_shift','crop_lesson','crop_pilot','crop_pilot_fresh']",1)
s=s.replace("['roe','roe_expanded_fresh']","['roe','roe_expanded_fresh','crop_pilot_fresh']")
anchor="address.searchParams.set('split',$('split').value);"
assert s.count(anchor)==1
s=s.replace(anchor,anchor+"if(data.variant){address.searchParams.set('variant',data.variant);}else{address.searchParams.delete('variant');}")
anchor="$('split').onchange="
assert s.count(anchor)==1
s=s.replace(anchor,"$('variant').onchange=()=>{page=0;load()};\n"+anchor)
p.write_text(s,encoding='utf-8')
for name in ('waid.py','crop_pilot_preview.py','train_crop_pilot.py','evaluate_crop_pilot.py'):py_compile.compile(str(root/name),doraise=True)
(root/'data/crop_pilot/viewer-script-check.js').write_text(s.split('<script>')[1].split('</script>')[0],encoding='utf-8')
print('Viewer integration and syntax prepared; collection appears only after verified publication')
