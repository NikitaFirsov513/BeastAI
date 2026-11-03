"""Categorize emitted false boxes and missed references using saved predictions and annotations."""
from pathlib import Path
from collections import Counter,defaultdict
import json
import baseline,waid
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'data/roe_expanded/evaluation'

def errors(row,persons):
    truth=row['reference'];unmatched=set(range(len(truth)));false=[]
    for pred in sorted(row['boxes'],key=lambda b:b['score'],reverse=True):
        best=max(unmatched,key=lambda i:baseline.overlap(pred,truth[i]),default=None)
        if best is not None and baseline.overlap(pred,truth[best])>=.5:unmatched.remove(best);continue
        animal=max((baseline.overlap(pred,b) for b in truth),default=0.);person=max((baseline.overlap(pred,b) for b in persons),default=0.)
        kind='duplicate_of_matched_reference' if animal>=.5 else 'overlaps_annotated_person' if person>=.5 else 'partial_animal_overlap_below_match_iou' if animal>0 else 'no_overlap_with_annotated_targets'
        false.append({'type':kind,'best_animal_iou':animal,'best_person_iou':person,'prediction':pred})
    assert len(false)==row['counts']['fp'] and len(unmatched)==row['counts']['fn']
    width,height=row['original_size'];scale=min(1024/width,1024/height);missed=[]
    for i in sorted(unmatched):
        box=truth[i];best=max((baseline.overlap(box,b) for b in row['boxes']),default=0.)
        kind='no_emitted_overlap' if best==0 else 'overlap_below_match_iou' if best<.5 else 'one_to_one_assignment_conflict'
        area=box['width']*width*box['height']*height*scale**2
        missed.append({'type':kind,'reference_index':i,'area_at_1024':area,'small_at_1024':area<32**2,'best_emitted_iou':best,'reference':box})
    return false,missed

def main():
    coco=json.loads((ROOT/'data/roe/source_metadata/annotation_coco.json').read_text(encoding='utf-8'));ims={im['id']:im for im in coco['images']};people=defaultdict(list)
    for a in coco['annotations']:
        if a['category_id']!=2:continue
        im=ims[a['image_id']];x,y,w,h=a['bbox'];people[im['file_name']].append({'x':x/im['width'],'y':y/im['height'],'width':w/im['width'],'height':h/im['height']})
    del coco,ims
    output={'scope':'Error categories use emitted boxes at the fixed0.25 threshold and author annotations. They are not a semantic review of every image.','limitations':['No emitted overlap does not establish why the network failed; low-confidence suppressed boxes are unavailable.','No target overlap can mean background or an unannotated object; annotation completeness is not independently certified.','Person annotations are available only for roe source.','Partial overlap below0.5 is a geometric description, not proof of the intended object.'],'reports':{}}
    files=['before.json','after.json','after_1280.json','fresh_before.json','fresh_after.json','fresh_after_1280.json']
    for name in files:
        path=OUT/name
        if not path.exists():continue
        report=json.loads(path.read_text(encoding='utf-8'));sources={};items={}
        for key,row in report['items'].items():
            source=row['source'];false,missed=errors(row,people[row['name']] if source=='roe' else [])
            if source not in sources:sources[source]={'false_box_types':Counter(),'miss_types':Counter(),'small_misses':0,'large_misses':0,'person_annotations_available':source=='roe'}
            stats=sources[source];stats['false_box_types'].update(f['type'] for f in false);stats['miss_types'].update(m['type'] for m in missed);stats['small_misses']+=sum(m['small_at_1024'] for m in missed);stats['large_misses']+=sum(not m['small_at_1024'] for m in missed)
            items[key]={'false_boxes':false,'missed_references':missed}
        for source,stats in sources.items():
            assert sum(stats['false_box_types'].values())==report['sources'][source]['quality']['fp'];assert sum(stats['miss_types'].values())==report['sources'][source]['quality']['fn']
        output['reports'][name]={'sources':sources,'items':items}
    fit_path=OUT/'training_fit_diagnostic.json'
    if fit_path.exists():
        fit=json.loads(fit_path.read_text(encoding='utf-8'));output['training_fit']={}
        for stage,r in fit['models'].items():
            missed=[b for row in r['items'].values() for b in row.get('reference_coverage',[]) if not b['matched']]
            output['training_fit'][stage]={'counts':r['counts'],'missed_with_no_emitted_overlap':sum(b['best_emitted_prediction_iou']==0 for b in missed),'missed_with_partial_overlap':sum(0<b['best_emitted_prediction_iou']<.5 for b in missed)}
    waid.save_json(OUT/'error_analysis.json',output)
    print(json.dumps({name:r['sources'] for name,r in output['reports'].items()}),flush=True)

if __name__=='__main__':main()
