from pathlib import Path
import py_compile,json,hashlib
import waid
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'data/dronemegadetector'
p=ROOT/'waid.py';s=p.read_text(encoding='utf-8');anchor='    class Handler(BaseHTTPRequestHandler):'
assert s.count(anchor)==1 and 'drone_references={}' not in s
s=s.replace(anchor,"    drone_references={}\n    if (ROOT/'data/dronemegadetector/evaluation/published.json').is_file():\n        from drone_reference_preview import DroneReferencePreview\n        for reference_split in ('control','fresh'):\n            collection=DroneReferencePreview(ROOT,parse_labels,reference_split)\n            drone_references[collection.dataset_id]=collection\n            allowed_images.update(collection.allowed_images)\n            dataset_options.append({'id':collection.dataset_id,'label':collection.title})\n\n"+anchor)
anchor="collection=crop_pilots[requested] if requested in crop_pilots else"
assert s.count(anchor)==1
s=s.replace(anchor,"collection=drone_references[requested] if requested in drone_references else crop_pilots[requested] if requested in crop_pilots else")
p.write_text(s,encoding='utf-8')
p=ROOT/'viewer.html';s=p.read_text(encoding='utf-8');anchor="if(['crop_pilot','crop_pilot_fresh'].includes(initialQuery.get('dataset')))"
assert s.count(anchor)==1
s=s.replace(anchor,"if(['drone_reference','drone_reference_fresh'].includes(initialQuery.get('dataset'))){const id=initialQuery.get('dataset'),option=document.createElement('option');option.value=id;option.textContent='Готовая модель · сравнение';$('dataset').append(option);$('dataset').value=id;$('split').value=id==='drone_reference_fresh'?'diagnostic':'valid';relatedGroup='';}"+anchor)
s=s.replace("'crop_pilot','crop_pilot_fresh'].includes(data.dataset)","'crop_pilot','crop_pilot_fresh','drone_reference','drone_reference_fresh'].includes(data.dataset)")
s=s.replace("['roe','roe_expanded_fresh','crop_pilot_fresh']","['roe','roe_expanded_fresh','crop_pilot_fresh','drone_reference_fresh']")
anchor="$('dataset').onchange=()=>{relatedGroup='';"
assert s.count(anchor)==1
s=s.replace(anchor,"$('dataset').onchange=()=>{$('variant').value='';relatedGroup='';")
p.write_text(s,encoding='utf-8')
for filename in ('waid.py','drone_reference_preview.py','evaluate_drone_reference.py'):py_compile.compile(str(ROOT/filename),doraise=True)
(OUT/'viewer-script-check.js').write_text(s.split('<script>')[1].split('</script>')[0],encoding='utf-8')
waid.save_json(OUT/'source_overlap_audit.json',{'paper_url':'https://wildlabs.net/sites/default/files/2026-10/2026.10.12.758430v1.full_.pdf','paper_sha256':hashlib.sha256((OUT/'paper.pdf').read_bytes()).hexdigest(),'paper_pages':31,'known_included_datasets':['BAMBI','WAID'],'evidence':'Dataset sources in author preprint, Table9 page31.','current_evaluation_sources':['Sherbrooke red deer','Young deer RGB Zenodo18206908'],'exact_names_not_found':['Sherbrooke','Young deer','roe deer','fawn'],'overlap_with_our80_verified_absent':False,'reason':'Exact filename/flight membership not supplied; alternate dataset names and republished derivatives are possible. Absence of keyword is not absence of overlap.','scope':'All results are development diagnostics. No new BAMBI benchmark is claimed independent.'})
print('Viewer routes and source-overlap note prepared')
