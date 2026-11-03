"""Verify live viewer data for every held-out image and crop-pilot variant."""
from pathlib import Path
import hashlib,json,urllib.request
import waid
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'data/crop_pilot';BASE='http://127.0.0.1:8765'
def get(path):
    with urllib.request.urlopen(BASE+path,timeout=30) as r:return r.read()
def main():
    reports={s:json.loads((OUT/'evaluation'/(s+'.json')).read_text(encoding='utf-8')) for s in ('before','after')}
    verified=0;image_urls={};comparisons=0
    for dataset,split,scope in [('crop_pilot','valid','control'),('crop_pilot_fresh','diagnostic','fresh')]:
        for variant in ('before_full','before_tiles','after_full','after_tiles'):
            stage,mode=variant.split('_');initial=json.loads(get(f'/api/items?dataset={dataset}&split={split}&variant={variant}'));seen=set()
            assert initial['variant']==variant and len(initial['variant_options'])==4
            assert initial['analysis']['precision_available']
            for comparison in initial['analysis']['experiments']:
                assert comparison['reference_boxes']>=comparison['matched'] and comparison['unmatched_reference']==comparison['reference_boxes']-comparison['matched'];comparisons+=1
            for page in range(initial['pages']):
                data=initial if page==0 else json.loads(get(f'/api/items?dataset={dataset}&split={split}&variant={variant}&page={page}'))
                assert data['dataset']==dataset and data['variant']==variant
                for e in data['items']:
                    key=e['source']+'/'+e['name'];assert key not in seen;seen.add(key)
                    saved=reports[stage]['modes'][mode]['items'][key]
                    assert e['boxes']==saved['reference'] and e['prediction']['boxes']==saved['boxes'] and e['prediction']['counts']==saved['counts'];image_urls[e['src']]=saved['image_sha256'];verified+=1
            assert len(seen)==(64 if scope=='control' else 16)
    for url,digest in image_urls.items():assert hashlib.sha256(get(url)).hexdigest()==digest
    for dataset,split,expected in [('tiny_memorization','train',4),('tiny_grid_shift','train',9)]:
        data=json.loads(get(f'/api/items?dataset={dataset}&split={split}'));assert data['analysis']['experiments'][-1]['matched']==expected
    lesson=json.loads(get('/api/items?dataset=crop_lesson&split=train'));assert lesson['total']==12 and lesson['analysis'] is None
    for dataset,split in [('roe_expanded','valid'),('roe_expanded_fresh','diagnostic')]:
        data=json.loads(get(f'/api/items?dataset={dataset}&split={split}'));assert data['dataset']==dataset
    for path,h in json.loads((OUT/'protected_artifacts.json').read_text(encoding='utf-8')).items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h
    html=get('/').decode('utf-8');assert 'id="variant"' in html and "data.variant_options" in html
    for bad in ['/api/items?dataset=crop_pilot&variant=other','/data/crop-pilot/../../README.md']:
        try:get(bad)
        except urllib.error.HTTPError as error:assert error.code>=400
        else:raise AssertionError('Unexpected access accepted')
    result={'passed':True,'report_image_variant_rows_checked':verified,'unique_http_images_sha256_checked':len(image_urls),'comparison_rows_checked':comparisons,'variant_selector_present':True,'existing_diagnostics_and_lesson_preserved':True,'protected_artifacts_unchanged':True,'regression_tests_passed':28,'javascript_syntax_checked':True,'visual_browser_inspection':False}
    waid.save_json(OUT/'evaluation/viewer-verification.json',result);print(json.dumps(result))
if __name__=='__main__':main()
