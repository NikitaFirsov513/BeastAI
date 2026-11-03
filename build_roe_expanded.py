"""Online conservative grouping of the larger pool against frozen controls and legacy groups."""
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import hashlib,json,time
import cv2,numpy as np,psutil
import waid
from group_roe_sample import features,relation
ROOT=Path(__file__).resolve().parent;OLD=ROOT/'data/mixed_pilot';POOL=ROOT/'data/roe/expanded_candidates';OUT=ROOT/'data/roe_expanded'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    assert ROOT.drive.lower()=='d:';OUT.mkdir(parents=True,exist_ok=True);cv2.setNumThreads(1);cv2.setRNGSeed(42);started=time.perf_counter()
    old=json.loads((OLD/'index.json').read_text(encoding='utf-8'));deer=[e for e in old['training'] if e['source']=='sherbrooke'];validation=old['validation'];assert len(deer)==512 and len(validation)==64
    legacy={e['id']:dict(e) for e in old['excluded_related_candidates']+old['omitted_repeated_candidates']+old['roe_training']}
    for e in validation:
        key=e.get('id') or 'sherbrooke_control/'+e['name'];legacy[key]={'id':key,'role':'control',**e}
    assert len(legacy)==160
    selected_before={e['id'] for e in old['roe_training']}
    nodes={};normal={};flipped={};matchers={};parent={};blocked={};edges=[];checked=0
    def find(k):
        while parent[k]!=k:parent[k]=parent[parent[k]];k=parent[k]
        return k
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b:parent[b]=a;blocked[a]=blocked[a] or blocked[b]
    def index(f):
        if f['desc'] is None:return None
        m=cv2.FlannBasedMatcher(dict(algorithm=1,trees=4),dict(checks=64));m.add([f['desc']]);m.train();return m
    def add(key,e):
        assert key not in nodes
        assert sha(e['image_path'])==e['image_sha256'] and sha(e['label_path'])==e['label_sha256']
        image=cv2.imdecode(np.fromfile(e['image_path'],dtype=np.uint8),cv2.IMREAD_COLOR);assert image is not None
        e=dict(e);e['id']=key;e['near_black_fraction']=float(np.mean(cv2.resize(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),(128,128),interpolation=cv2.INTER_AREA)<8))
        nodes[key]=e;parent[key]=key;blocked[key]=e['role']=='control';normal[key]=features(image);flipped[key]=features(cv2.flip(image,1));matchers[key]=(index(normal[key]),index(flipped[key]))
    def compare(a,b):
        cv2.setRNGSeed(int.from_bytes(hashlib.sha256((a+'\0'+b).encode()).digest()[:4],'little')%2147483647)
        if nodes[a]['image_sha256']==nodes[b]['image_sha256']:return {'method':'exact_sha256'}
        found=[]
        for j,orientation in enumerate(('normal','horizontal_flip')):
            score=relation(normal[a],normal[b] if j==0 else flipped[b],matcher=matchers[b][j])
            if score:found.append({'method':'geometric_features','target_orientation':orientation,**score})
        return max(found,key=lambda x:x['inliers']) if found else None
    for key,e in legacy.items():add(key,e)
    for group in old['visual_components']:
        for key in group[1:]:union(group[0],key)
    check=compare('roe_control/00006_1.jpg','roe_control/13825_1.jpg');assert check is not None;checked+=1
    pool_threads=ThreadPoolExecutor(max_workers=4)
    known_pairs=[];used_targets=set()
    for edge in old['related_edges']:
        if edge['b'] not in used_targets:
            used_targets.add(edge['b']);known_pairs.append((edge['a'],edge['b']))
        if len(known_pairs)==12:break
    serial=[compare(a,b) for a,b in known_pairs]
    parallel=list(pool_threads.map(lambda pair:compare(*pair),known_pairs));checked+=2*len(known_pairs)
    assert all((a is None)==(b is None) and (a is None or a.get('inliers')==b.get('inliers')) for a,b in zip(serial,parallel)),'Parallel matching differs from serial matching'
    waid.save_json(OUT/'parallel_matching_verification.json',{'passed':True,'pairs':len(known_pairs),'same_decisions_and_inlier_counts':True,'workers':4,'opencv_threads_per_worker':1,'target_matchers_not_shared_concurrently':True})
    known=set(legacy);processed=[]
    while len(processed)<512:
        path=POOL/'partial_index.json'
        rows=json.loads(path.read_text(encoding='utf-8'))['entries'] if path.exists() else []
        fresh=[e for e in rows if 'roe_new/'+e['name'] not in nodes]
        if not fresh:
            state=json.loads((POOL/'status.json').read_text(encoding='utf-8')) if (POOL/'status.json').exists() else {}
            if state.get('state')=='failed':raise RuntimeError('Download failed: '+state.get('error',''))
            waid.save_json(OUT/'preparation_status.json',{'state':'waiting_for_download','processed':len(processed),'target':512,'download_completed':len(rows),'pair_comparisons':checked,'elapsed_seconds':time.perf_counter()-started})
            time.sleep(3);continue
        for entry in fresh:
            a='roe_new/'+entry['name'];prior_keys=list(nodes);add(a,{'role':'candidate',**entry});processed.append(a)
            # Same all-pairs geometric rule, processed online. Skip only an already united
            # pair or two components already excluded through a protected control.
            prior_keys.sort(key=lambda b:not blocked[find(b)])
            for offset in range(0,len(prior_keys),16):
                batch=[b for b in prior_keys[offset:offset+16] if find(a)!=find(b) and not (blocked[find(a)] and blocked[find(b)])]
                scores=list(pool_threads.map(lambda b:compare(a,b),batch));checked+=len(batch)
                for b,score in zip(batch,scores):
                    if score:edges.append({'a':a,'b':b,**score});union(a,b)
            if len(processed)%4==0:
                excluded=sum(blocked[find(k)] for k in processed)
                waid.save_json(OUT/'preparation_status.json',{'state':'grouping_new_candidates','processed':len(processed),'target':512,'download_completed':len(rows),'new_candidates_related_to_controls':excluded,'pair_comparisons':checked,'elapsed_seconds':time.perf_counter()-started,'process_rss_MiB':psutil.Process().memory_info().rss/2**20})
                print(f'Grouped {len(processed)}/512; excluded {excluded}; checked {checked} pairs',flush=True)
    pool_threads.shutdown()
    pool=json.loads((POOL/'index.json').read_text(encoding='utf-8')) if (POOL/'index.json').exists() else None
    while pool is None:
        time.sleep(1)
        if (POOL/'index.json').exists():pool=json.loads((POOL/'index.json').read_text(encoding='utf-8'))
    assert len(pool['entries'])==512
    groups=defaultdict(list)
    for key in nodes:groups[find(key)].append(key)
    selected=[];excluded=[];redundant=[]
    for number,(root,keys) in enumerate(sorted(groups.items()),1):
        candidates=[k for k in keys if nodes[k]['role']=='candidate']
        if not candidates:continue
        if blocked[find(root)]:excluded.extend(candidates);continue
        chosen=min(candidates,key=lambda k:(k not in selected_before,nodes[k]['near_black_fraction'],nodes[k]['name']))
        e=dict(nodes[chosen]);e['source']='roe';e['visual_group']='E'+str(number);selected.append(e);redundant.extend(k for k in candidates if k!=chosen)
    selected.sort(key=lambda e:e['name']);assert selected,'All candidates excluded; no training dataset created'
    assert len(selected)+len(excluded)+len(redundant)==608
    training=deer+selected;train_paths={e['image_path'] for e in training};assert train_paths.isdisjoint(e['image_path'] for e in validation)
    for e in training+validation:
        assert sha(e['image_path'])==e['image_sha256'] and sha(e['label_path'])==e['label_sha256']
        assert waid.parse_labels(Path(e['label_path']),['animal'])['status'] in ('ok','empty')
    assert all(not blocked[find(e['id'])] for e in selected)
    revision=hashlib.sha256(json.dumps([(e['image_path'],e['image_sha256'],e['label_sha256']) for e in training+validation]).encode()).hexdigest()
    result={'created_utc':datetime.now(timezone.utc).isoformat(),'revision':revision,'training':training,'validation':validation,'roe_training':selected,'source_previous_revision':old['revision'],'excluded_related_candidates':[nodes[k] for k in excluded],'omitted_repeated_candidates':[nodes[k] for k in redundant],'visual_components':list(groups.values()),'related_edges':edges,'legacy_relations_file':str(OLD/'index.json'),'legacy_matching_positive_control':check,'legacy_selected_retained':sum(e['id'] in selected_before for e in selected),'legacy_selected_now_connected_to_control':sum(blocked[find(k)] for k in selected_before),'new_candidate_images':512,'new_candidate_boxes':pool['candidate_boxes'],'pair_comparisons':checked,'preparation_seconds':time.perf_counter()-started,'process_rss_MiB':psutil.Process().memory_info().rss/2**20,'validation_scope':'Unchanged development controls; geometric components are not flight IDs and do not prove independent test data.'}
    waid.save_json(OUT/'index.json',result)
    for name,entries in [('train_unique.txt',training),('valid.txt',validation)]:
        (OUT/name).write_text(''.join(Path(e['image_path']).as_posix()+'\n' for e in entries),encoding='utf-8')
    summary={'state':'prepared','unique_roe_training_images':len(selected),'roe_training_boxes':sum(e['reference_boxes'] for e in selected),'deer_training_images':512,'validation_images':64,'candidate_images_total':608,'new_candidate_images':512,'excluded_related_to_control':len(excluded),'omitted_repeated':len(redundant),'legacy_selected_retained':result['legacy_selected_retained'],'legacy_selected_now_connected_to_control':result['legacy_selected_now_connected_to_control'],'revision':revision,'preparation_seconds':result['preparation_seconds']}
    waid.save_json(OUT/'summary.json',summary);waid.save_json(OUT/'preparation_status.json',summary);print(json.dumps(summary),flush=True)

if __name__=='__main__':
    try:main()
    except Exception as error:
        waid.save_json(OUT/'preparation_status.json',{'state':'failed','error':str(error)})
        raise
