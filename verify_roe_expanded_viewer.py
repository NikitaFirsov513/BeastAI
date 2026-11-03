"""Verify expanded collections over the live local viewer; preserve prior experiments."""
from pathlib import Path
import hashlib,json,shutil,subprocess,urllib.request
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'data/roe_expanded';BASE='http://127.0.0.1:8765'

def fetch(path):
    with urllib.request.urlopen(BASE+path,timeout=15) as response:return response.read()
def api(query):return json.loads(fetch('/api/items?'+query))

def main():
    index=json.loads((OUT/'index.json').read_text(encoding='utf-8'));pub=OUT/'evaluation/published.json';publication=json.loads(pub.read_text(encoding='utf-8')) if pub.exists() else None
    result={}
    for mode,field,dataset,split in [('train','roe_training','roe_expanded_train','train'),('control','validation','roe_expanded','valid'),('fresh','fresh_validation','roe_expanded_fresh','diagnostic')]:
        entries=index[field]
        if not entries:continue
        expected={e['source']+'/'+e['name']:e for e in entries};first=api('dataset='+dataset+'&split='+split);assert first['total']==len(entries)
        report=None;fit=None
        if publication and mode=='train':fit=json.loads((OUT/'evaluation/training_fit_diagnostic.json').read_text(encoding='utf-8'))['models']['after']
        if publication and mode!='train':
            stage=publication['active_stage'] if publication['active_stage'] in publication['reports'][mode] else 'after'
            report=json.loads((ROOT/publication['reports'][mode][stage]['path']).read_text(encoding='utf-8'))
            assert first['baseline']['conditions']==report['conditions']
        else:assert first['baseline'] is None and first['analysis'] is None
        seen=set();refs=predictions=0
        for page in range(first['pages']):
            data=api('dataset='+dataset+'&split='+split+'&page='+str(page))
            for row in data['items']:
                key=row['source']+'/'+row['name'];assert key not in seen;seen.add(key);entry=expected[key]
                assert row['status'] in ('ok','empty') and len(row['boxes'])==entry['reference_boxes']
                assert hashlib.sha256(fetch(row['src'])).hexdigest()==entry['image_sha256'];refs+=len(row['boxes'])
                if report:
                    saved=report['items'][key];assert row['boxes']==saved['reference'] and row['prediction']['boxes']==saved['boxes'] and row['prediction']['counts']==saved['counts'];predictions+=len(row['prediction']['boxes'])
                elif fit:
                    saved=fit['items'][row['name']];assert row['prediction']['boxes']==saved['predictions'] and row['boxes']==saved['reference'];predictions+=len(row['prediction']['boxes'])
                else:assert row['prediction'] is None
        assert seen==set(expected)
        result[mode]={'images':len(seen),'reference_boxes':refs,'predictions':predictions if (report or fit) else None,'all_images_sha256_verified':True,'predictions_equal_saved_reports':bool(report or fit)}
    assert {e['image_path'] for e in index['training']}.isdisjoint(e['image_path'] for e in index['validation']+index['fresh_validation'])
    old=api('dataset=sherbrooke&split=valid');assert old['total']==32 and any(e.get('matched')==83 and e.get('unmatched_predictions')==3 for e in old['analysis']['experiments'])
    for dataset,roe_tp,deer_tp,deer_fp in [('mixed',2,94,11),('roe_focus',8,89,15)]:
        d=api('dataset='+dataset+'&split=valid');assert d['total']==64
        assert any(e['matched']==roe_tp and e['reference_boxes']==44 for e in d['analysis']['experiments'])
        assert any(e['matched']==deer_tp and e['reference_boxes']==115 and e['unmatched_predictions']==deer_fp for e in d['analysis']['experiments'])
    assert api('dataset=roe&split=diagnostic')['total']==32 and api('dataset=roe&split=diagnostic&group=G1')['total']==2
    html=fetch('/').decode('utf-8');script=html.split('<script>',1)[1].split('</script>',1)[0];path=OUT/'viewer-syntax-check.js';path.write_text(script,encoding='utf-8');subprocess.run([shutil.which('node'),'--check',str(path)],check=True)
    output={'passed':True,'collections':result,'previous_results_preserved':True,'training_control_paths_disjoint':True,'javascript_syntax_passed':True,'reports_published':publication is not None}
    dest=OUT/'evaluation/viewer-verification.json' if publication else OUT/'viewer-prepared-verification.json';dest.parent.mkdir(exist_ok=True);dest.write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(output,ensure_ascii=True))

if __name__=='__main__':main()
