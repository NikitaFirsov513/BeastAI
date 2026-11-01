"""Build a small mixed pilot after conservative visual-group exclusion against both controls."""
from pathlib import Path
from collections import defaultdict,Counter
from datetime import datetime,timezone
import hashlib,json,time
import cv2,numpy as np,psutil
import waid
from group_roe_sample import features,relation
ROOT=Path(__file__).resolve().parent;ROE=ROOT/'data/roe';OUT=ROOT/'data/mixed_pilot'


def main():
    assert ROOT.drive.lower()=='d:';OUT.mkdir(parents=True,exist_ok=True)
    cv2.setNumThreads(4);cv2.setRNGSeed(42);started=time.perf_counter()
    candidates=json.loads((ROE/'mixed_candidates/index.json').read_text(encoding='utf-8'))
    audit=json.loads((ROE/'audit/summary.json').read_text(encoding='utf-8'))
    old=json.loads((ROOT/'data/sherbrooke/train512/index.json').read_text(encoding='utf-8'))
    waid.configure_dataset('sherbrooke');nodes={};protected=[];candidate_ids=[];old_train=[];old_valid=[]
    for item in audit['sample']:
        name=item['name'];key='roe_control/'+name
        nodes[key]={'id':key,'source':'roe','role':'control','name':name,'image_path':str(ROE/'audit/images'/name),'label_path':str(ROE/'audit/labels'/Path(name).with_suffix('.txt').name),'image_sha256':item['sha256'],'label_sha256':item['label_sha256'],'reference_boxes':item['target_labels']};protected.append(key)
    for pair in old['pairs']:
        entry={'source':'sherbrooke','name':pair['name'],'image_path':str(waid.local_file(pair['image'])),'label_path':str(waid.local_file(pair['label'])),'image_sha256':pair['image']['sha256'],'label_sha256':pair['label']['sha256'],'reference_boxes':pair['reference_count']}
        if pair['split']=='train':old_train.append(entry)
        else:
            key='sherbrooke_control/'+pair['name'];nodes[key]={'id':key,'role':'control',**entry};protected.append(key);old_valid.append(entry)
    for item in candidates['entries']:
        key='roe_candidate/'+item['name'];nodes[key]={'id':key,'source':'roe','role':'candidate',**item};candidate_ids.append(key)
    assert len(protected)==64 and len(candidate_ids)==96 and len(old_train)==512
    plan={'created_utc':datetime.now(timezone.utc).isoformat(),'candidates':96,'protected_roe_images':32,'protected_deer_images':32,'old_training_images':512,'representative_policy':'One candidate per unprotected visual component, lowest near-black-pixel fraction, then lexical filename. No model predictions used.','relation_policy':'Same SIFT and homography thresholds as original 32-image scan; original/mirrored targets; cached target indexes.','limits':'Visual grouping cannot verify flight identity or guarantee absence of undetected related copies; this remains developmental evidence.'}
    waid.save_json(OUT/'preparation_plan.json',plan)
    normal={};flipped={};matchers={};parent={k:k for k in nodes};blocked={k:k in protected for k in nodes};edges=[];pairs_checked=0
    def find(k):
        while parent[k]!=k:parent[k]=parent[parent[k]];k=parent[k]
        return k
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b:parent[b]=a;blocked[a]=blocked[a] or blocked[b]
    def make_index(f):
        if f['desc'] is None:return None
        m=cv2.FlannBasedMatcher(dict(algorithm=1,trees=4),dict(checks=64));m.add([f['desc']]);m.train();return m
    for number,(key,item) in enumerate(nodes.items(),1):
        path=Path(item['image_path']);assert path.resolve().is_relative_to(ROOT)
        content=path.read_bytes();assert hashlib.sha256(content).hexdigest()==item['image_sha256']
        assert hashlib.sha256(Path(item['label_path']).read_bytes()).hexdigest()==item['label_sha256']
        image=cv2.imdecode(np.frombuffer(content,dtype=np.uint8),cv2.IMREAD_COLOR);assert image is not None
        thumb=cv2.resize(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),(128,128),interpolation=cv2.INTER_AREA);item['near_black_fraction']=float(np.mean(thumb<8))
        normal[key]=features(image);flipped[key]=features(cv2.flip(image,1));matchers[key]=(make_index(normal[key]),make_index(flipped[key]))
        if number%16==0:
            print(f'Features {number}/{len(nodes)}',flush=True);waid.save_json(OUT/'preparation_status.json',{'state':'extracting_features','completed':number,'total':len(nodes)})
    known=json.loads((ROE/'grouping/sample_groups.json').read_text(encoding='utf-8'))
    assert known['input_image_sha256']=={i['name']:i['sha256'] for i in audit['sample']}
    for names in known['nontrivial_groups']:
        for name in names[1:]:union('roe_control/'+names[0],'roe_control/'+name)
    def compare(a,b):
        nonlocal pairs_checked
        pairs_checked+=1;found=[]
        if nodes[a]['image_sha256']==nodes[b]['image_sha256']:return {'method':'exact_sha256'}
        for index,orientation in enumerate(('normal','horizontal_flip')):
            score=relation(normal[a],normal[b] if index==0 else flipped[b],matcher=matchers[b][index])
            if score:found.append({'method':'geometric_features','target_orientation':orientation,**score})
        return max(found,key=lambda s:s['inliers']) if found else None
    control_pair=compare('roe_control/00006_1.jpg','roe_control/13825_1.jpg');assert control_pair is not None,'Cached matching failed its known-related positive control'
    # Compare every candidate with protected controls until it is already excluded.
    for number,a in enumerate(candidate_ids,1):
        for b in protected:
            if blocked[find(a)]:break
            score=compare(a,b)
            if score:edges.append({'a':a,'b':b,**score});union(a,b)
        if number%4==0:
            excluded=sum(blocked[find(k)] for k in candidate_ids)
            print(f'Protected comparison {number}/96; excluded so far {excluded}',flush=True);waid.save_json(OUT/'preparation_status.json',{'state':'checking_protected_images','completed':number,'total':96,'excluded_candidates':excluded,'elapsed_seconds':time.perf_counter()-started})
    # Candidate links propagate exclusions transitively and reduce repeated training scenes.
    for number,a in enumerate(candidate_ids,1):
        for b in candidate_ids[number:]:
            if find(a)==find(b) or (blocked[find(a)] and blocked[find(b)]):continue
            score=compare(a,b)
            if score:edges.append({'a':a,'b':b,**score});union(a,b)
        if number%4==0:
            print(f'Candidate grouping {number}/96',flush=True);waid.save_json(OUT/'preparation_status.json',{'state':'grouping_candidates','completed':number,'total':96,'elapsed_seconds':time.perf_counter()-started})
    components=defaultdict(list)
    for key in nodes:components[find(key)].append(key)
    selected=[];excluded=[];redundant=[]
    for group_number,(root,keys) in enumerate(sorted(components.items()),1):
        pool=[k for k in keys if nodes[k]['role']=='candidate']
        if not pool:continue
        if blocked[find(root)]:excluded.extend(pool);continue
        chosen=min(pool,key=lambda k:(nodes[k]['near_black_fraction'],nodes[k]['name']))
        entry=dict(nodes[chosen]);entry['visual_group']='V'+str(group_number);selected.append(entry);redundant.extend(k for k in pool if k!=chosen)
    selected.sort(key=lambda i:i['name']);assert selected,'No candidates passed exclusion; do not produce a misleading training manifest'
    assert len(selected)+len(excluded)+len(redundant)==96
    training=old_train+selected
    validation=old_valid+[nodes[k] for k in protected if nodes[k]['source']=='roe']
    assert len(validation)==64
    train_paths={entry['image_path'] for entry in training};val_paths={entry['image_path'] for entry in validation};assert train_paths.isdisjoint(val_paths)
    for entry in training+validation:
        assert hashlib.sha256(Path(entry['image_path']).read_bytes()).hexdigest()==entry['image_sha256']
        assert hashlib.sha256(Path(entry['label_path']).read_bytes()).hexdigest()==entry['label_sha256']
        assert waid.parse_labels(Path(entry['label_path']),['animal'])['status'] in ('ok','empty')
    assert all(not blocked[find(entry['id'])] for entry in selected)
    revision=hashlib.sha256(json.dumps([(entry['image_path'],entry['image_sha256'],entry['label_sha256']) for entry in training+validation]).encode()).hexdigest()
    (OUT/'train.txt').write_text(''.join(Path(entry['image_path']).as_posix()+'\n' for entry in training),encoding='utf-8')
    (OUT/'valid.txt').write_text(''.join(Path(entry['image_path']).as_posix()+'\n' for entry in validation),encoding='utf-8')
    (OUT/'dataset.yaml').write_text(f'path: {ROOT.as_posix()}\ntrain: {(OUT/"train.txt").as_posix()}\nval: {(OUT/"valid.txt").as_posix()}\nnames:\n  0: animal\n',encoding='utf-8')
    index={'revision':revision,'training':training,'validation':validation,'roe_training':selected,'excluded_related_candidates':[nodes[k] for k in excluded],'omitted_repeated_candidates':[nodes[k] for k in redundant],'visual_components':list(components.values()),'related_edges':edges,'plan':plan,'control_matching_check':control_pair,'relation_pairs_checked':pairs_checked,'preparation_seconds':time.perf_counter()-started,'process_rss_MiB':psutil.Process().memory_info().rss/2**20,'training_started':False,'validation_scope':'Developmental only: 32 old deer frames and frozen 32 roe diagnostics; grouping is not verified flight independence.'}
    waid.save_json(OUT/'index.json',index)
    summary={'training_images':len(training),'deer_training_images':len(old_train),'roe_training_images':len(selected),'training_boxes':sum(e['reference_boxes'] for e in training),'roe_training_boxes':sum(e['reference_boxes'] for e in selected),'validation_images':64,'deer_validation_images':32,'roe_diagnostic_images':32,'roe_diagnostic_boxes':44,'candidate_images':96,'excluded_related_to_control':len(excluded),'omitted_repeated_candidates':len(redundant),'known_cross_split_relation_edges':sum(1 for edge in edges if (edge['a'] in {e['id'] for e in selected} and nodes[edge['b']]['role']=='control') or (edge['b'] in {e['id'] for e in selected} and nodes[edge['a']]['role']=='control')),'revision':revision,'dataset_yaml':str(OUT/'dataset.yaml'),'preparation_seconds':index['preparation_seconds'],'training_started':False,'validation_flight_independence_verified':False}
    # Any relation leaving a selected component would be an invalid grouping or split.
    selected_ids={e['id'] for e in selected}
    assert not any((edge['a'] in selected_ids and blocked[find(edge['b'])]) or (edge['b'] in selected_ids and blocked[find(edge['a'])]) for edge in edges)
    waid.save_json(OUT/'summary.json',summary);waid.save_json(OUT/'preparation_status.json',{'state':'prepared',**summary})
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
