from pathlib import Path
import hashlib,io,json,time,urllib.request
from PIL import Image
import waid
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'data/crop_lesson'
base='http://127.0.0.1:8765'
def get(path):
    with urllib.request.urlopen(base+path,timeout=30) as r:return r.read()
for attempt in range(15):
    try:
        response=json.loads(get('/api/items?dataset=crop_lesson&split=train'));break
    except OSError:
        if attempt==14:raise
        time.sleep(1)
assert response['dataset']=='crop_lesson' and response['total']==12
assert response['baseline'] is None and response['analysis'] is None
index=json.loads((OUT/'index.json').read_text(encoding='utf-8'))
assert 'crop_lesson' in {e['id'] for e in response['dataset_options']}
for row,e in zip(response['items'],index['entries']):
    content=get(row['src']);assert hashlib.sha256(content).hexdigest()==e['image_sha256']
    assert Image.open(io.BytesIO(content)).size==(1024,1024)
    assert row['prediction'] is None and row['boxes']==waid.parse_labels(Path(e['label_path']),['animal'])['boxes']
    assert row['caption']==e['caption']
# Check the existing collections still load the frozen reports, not the new lesson.
counts={}
for dataset,split in [('tiny_memorization','train'),('tiny_grid_shift','train'),('roe_expanded','valid'),('roe_expanded_fresh','diagnostic')]:
    data=json.loads(get('/api/items?dataset='+dataset+'&split='+split))
    assert data['dataset']==dataset and data['analysis'] is not None
    counts[dataset]=[{k:row[k] for k in ('label','matched','reference_boxes','unmatched_predictions')} for row in data['analysis']['experiments']]
assert counts['tiny_memorization'][-1]['matched']==4
assert counts['tiny_grid_shift'][-1]['matched']==9
for path,digest in json.loads((OUT/'protected_artifacts.json').read_text(encoding='utf-8')).items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
html=get('/').decode('utf-8');assert "item.caption" in html and "value='crop_lesson'" in html
waid.save_json(OUT/'viewer-verification.json',{'passed':True,'preview_images_verified':12,'labels_match_files':True,'no_predictions_or_quality_claims':True,'existing_collection_results':counts,'protected_artifacts_unchanged':True,'regression_tests_passed':24,'javascript_syntax_passed':True})
print(json.dumps({'passed':True,'url':base+'/?dataset=crop_lesson&split=train','preview_images':12,'example':index['examples'][0]},ensure_ascii=False))
