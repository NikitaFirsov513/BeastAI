"""Verify published mixed results and original collections through the live local viewer."""
from pathlib import Path
import argparse,hashlib,json,shutil,subprocess,urllib.request
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'data/mixed_pilot';BASE='http://127.0.0.1:8765'

def fetch(path):
    with urllib.request.urlopen(BASE+path,timeout=10) as response:return response.read()
def api(query):return json.loads(fetch('/api/items?'+query))

def main():
    global OUT
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--experiment',choices=('mixed_pilot','roe_focus'),default='mixed_pilot');args=parser.parse_args();OUT=ROOT/'data'/args.experiment
    dataset='roe_focus' if args.experiment=='roe_focus' else 'mixed'
    after=json.loads((OUT/'evaluation/after.json').read_text(encoding='utf-8'))
    first=api('dataset='+dataset+'&split=valid');assert first['total']==64 and len(first['dataset_options'])>=(5 if dataset=='roe_focus' else 4)
    assert len(first['analysis']['experiments'])==(6 if dataset=='roe_focus' else 4) and first['baseline']['conditions']==after['conditions']
    seen=set();boxes=predictions=0
    for page in range(first['pages']):
        data=api('dataset='+dataset+'&split=valid&page='+str(page))
        for row in data['items']:
            key=row['source']+'/'+row['name'];expected=after['items'][key]
            assert key not in seen;seen.add(key)
            assert row['prediction']['boxes']==expected['boxes'] and row['prediction']['counts']==expected['counts']
            assert row['boxes']==expected['reference'] and row['status'] in ('ok','empty')
            assert hashlib.sha256(fetch(row['src'])).hexdigest()==expected['image_sha256']
            boxes+=len(row['boxes']);predictions+=len(row['prediction']['boxes'])
    assert seen==set(after['items']) and boxes==159
    for category,source in [('roe_deer','roe'),('red_deer','sherbrooke')]:
        data=api('dataset='+dataset+'&split=valid&category='+category);assert data['total']==32
        assert all(row['source']==source for row in data['items']) and len(data['analysis']['experiments'])==(3 if dataset=='roe_focus' else 2)
    if dataset=='roe_focus':
        prior=api('dataset=mixed&split=valid');assert prior['total']==64
        assert any(e['matched']==2 and e['reference_boxes']==44 and e['unmatched_predictions']==3 for e in prior['analysis']['experiments'])
        assert any(e['matched']==94 and e['reference_boxes']==115 and e['unmatched_predictions']==11 for e in prior['analysis']['experiments'])
    old=api('dataset=sherbrooke&split=valid');assert old['total']==32
    assert any(e.get('matched')==83 and e.get('unmatched_predictions')==3 for e in old['analysis']['experiments'])
    roe=api('dataset=roe&split=diagnostic');assert roe['total']==32 and roe['analysis']['experiments'][0]['matched']==0
    assert api('dataset=roe&split=diagnostic&group=G1')['total']==2
    train=api('dataset=roe_train&split=train');assert train['total']==27 and train['baseline'] is None
    assert all(row['prediction'] is None for row in train['items'])
    html=fetch('/').decode('utf-8');script=html.split('<script>',1)[1].split('</script>',1)[0]
    script_path=OUT/'viewer-syntax-check.js';script_path.write_text(script,encoding='utf-8')
    subprocess.run([shutil.which('node'),'--check',str(script_path)],check=True)
    report={'passed':True,'mixed_control_images':len(seen),'reference_boxes':boxes,'predicted_boxes':predictions,'new_predictions_and_reference_labels_equal_saved_reports':True,'all_control_image_sha256_verified':True,'source_filters_passed':True,'old_deer_results_preserved':True,'old_roe_results_preserved':True,'old_related_group_G1_preserved':True,'training_preview_still_has_no_scoring_claim':True,'viewer_js_syntax_passed':True,'dataset_options':first['dataset_options']}
    (OUT/'evaluation/viewer-verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=True))

if __name__=='__main__':main()
