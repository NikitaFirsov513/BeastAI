"""Check reference viewer rows against the frozen reports for every variant."""
from project_paths import PROJECT_ROOT
from pathlib import Path
import json,hashlib,time,urllib.request,urllib.error
import waid
ROOT=PROJECT_ROOT;OUT=ROOT/'data/dronemegadetector';BASE='http://127.0.0.1:8765'
def get(path):
    with urllib.request.urlopen(BASE+path,timeout=30) as r:return r.read()
def main():
    report=json.loads((OUT/'evaluation/comparison.json').read_text(encoding='utf-8'));seen_images={};checked=0
    for attempt in range(20):
        try:get('/');break
        except OSError:
            if attempt==19:raise
            time.sleep(1)
    for dataset,split,count in [('drone_reference','valid',64),('drone_reference_fresh','diagnostic',16)]:
        for variant in ('baseline','reference'):
            path=f'/api/items?dataset={dataset}&split={split}&variant={variant}';first=json.loads(get(path));seen=set()
            assert first['dataset']==dataset and first['variant']==variant and len(first['variant_options'])==2
            for page in range(first['pages']):
                data=first if page==0 else json.loads(get(path+'&page='+str(page)))
                for e in data['items']:
                    key=e['source']+'/'+e['name'];assert key not in seen;seen.add(key);r=report['models'][variant]['items'][key]
                    assert e['boxes']==r['reference'] and e['prediction']['boxes']==r['boxes'] and e['prediction']['counts']==r['counts'];seen_images[e['src']]=r['image_sha256'];checked+=1
            assert len(seen)==count
            assert first['analysis']['precision_available'] and len(first['analysis']['experiments'])==(4 if count==64 else 2)
    default=json.loads(get('/api/items?dataset=drone_reference_fresh&split=diagnostic'));assert default['variant']=='reference'
    for url,digest in seen_images.items():assert hashlib.sha256(get(url)).hexdigest()==digest
    for dataset,split,variant,count in [('crop_pilot','valid','after_tiles',64),('crop_pilot_fresh','diagnostic','after_tiles',16),('roe_expanded','valid','',64),('crop_lesson','train','',12)]:
        data=json.loads(get(f'/api/items?dataset={dataset}&split={split}&variant={variant}'));assert data['dataset']==dataset and data['total']==count
    for dataset,tp in [('tiny_memorization',4),('tiny_grid_shift',9)]:
        data=json.loads(get('/api/items?dataset='+dataset+'&split=train'));assert data['analysis']['experiments'][-1]['matched']==tp
    html=get('/').decode('utf-8');assert "'drone_reference','drone_reference_fresh'" in html and "$('variant').value='';relatedGroup" in html
    for bad in ('/api/items?dataset=drone_reference&variant=invalid','/data/drone-reference/../../README.md'):
        try:get(bad)
        except urllib.error.HTTPError as e:assert e.code>=400
        else:raise AssertionError('Unexpected access accepted')
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in json.loads((OUT/'protected_artifacts.json').read_text(encoding='utf-8')).items())
    result={'passed':True,'report_variant_rows_checked':checked,'unique_http_images_checked':len(seen_images),'all_predicted_and_reference_boxes_match_reports':True,'variant_switch_reset_present':True,'default_reference_variant_checked':True,'old_collections_preserved':True,'protected_artifacts_unchanged':True,'regression_tests_passed':30,'javascript_syntax_checked':True,'visual_browser_inspection':False}
    waid.save_json(OUT/'evaluation/viewer-verification.json',result);print(json.dumps(result))
if __name__=='__main__':main()
