"""Prepare a training-only crop lesson. No downloads, training or inference."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json
import cv2
import numpy as np
import waid
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'data/crop_lesson'
NAMES=['03650_1.jpg','07152_1.jpg','15276_1.jpg','19700_1.jpg','00128_1.jpg']
SIZE=512
POSITIONS=(0,384,768)

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def clipped_labels(boxes,x,y):
    rows=[]; mapping=[]
    for i,b in enumerate(boxes):
        left=max(x,b['x']*1280);top=max(y,b['y']*1280)
        right=min(x+SIZE,(b['x']+b['width'])*1280)
        bottom=min(y+SIZE,(b['y']+b['height'])*1280)
        if right<=left or bottom<=top: continue
        w,h=right-left,bottom-top
        rows.append([0,((left+right)/2-x)/SIZE,((top+bottom)/2-y)/SIZE,w/SIZE,h/SIZE])
        visible=w*h/(b['width']*1280*b['height']*1280)
        mapping.append({'source_box':i,'visible_fraction':min(1.0,visible),'truncated':visible<1-1e-7})
    return rows,mapping

def write_item(name,image,labels,**extra):
    ip=OUT/'images'/name;lp=OUT/'labels'/(Path(name).stem+'.txt')
    assert cv2.imwrite(str(ip),image)
    lp.write_text(''.join('0 '+' '.join(f'{v:.12f}' for v in r[1:])+'\n' for r in labels),encoding='utf-8')
    parsed=waid.parse_labels(lp,['animal']);assert not parsed['errors']
    assert len(parsed['boxes'])==len(labels)
    assert np.array_equal(cv2.imread(str(ip)),image)
    return {'name':name,'image_path':str(ip),'label_path':str(lp),'image_sha256':sha(ip),'label_sha256':sha(lp),'dimensions':[image.shape[1],image.shape[0]],'reference_boxes':len(labels),**extra}

def main():
    assert ROOT.drive.lower()=='d:'
    if (OUT/'index.json').exists(): raise RuntimeError('Lesson exists; preserve it')
    for p in ('images','labels'): (OUT/p).mkdir(parents=True,exist_ok=True)
    source=ROOT/'data/roe_expanded';index=json.loads((source/'index.json').read_text(encoding='utf-8'))
    training={e['name']:e for e in index['roe_training']}
    protected={}
    for key in ('training','validation','fresh_validation'):
        for e in index[key]:
            for field,digest in [('image_path','image_sha256'),('label_path','label_sha256')]:
                assert sha(e[field])==e[digest];protected[e[field]]=e[digest]
    for p in [source/'index.json',source/'reserved_fresh_groups.json',source/'latest_run.json',source/'evaluation/after.json',source/'evaluation/fresh_after.json']:
        protected[str(p)]=sha(p)
    run=json.loads((source/'latest_run.json').read_text(encoding='utf-8'))
    protected[run['weights']]=sha(run['weights'])
    excluded={(e['source'],e['name']) for k in ('validation','fresh_validation') for e in index[k]}
    assert all(('roe',name) not in excluded and name in training for name in NAMES)
    tiles=[];preview=[];examples=[];background=[]
    cv2.setNumThreads(4)
    # Geometry regression: an intersecting animal is clipped, never silently erased.
    synthetic=[{'x':500/1280,'y':100/1280,'width':24/1280,'height':20/1280}]
    rr,mm=clipped_labels(synthetic,0,0)
    assert len(rr)==1 and abs(rr[0][3]-12/512)<1e-9 and abs(mm[0]['visible_fraction']-.5)<1e-9
    assert not clipped_labels(synthetic,768,768)[0]
    assert not clipped_labels([{'x':512/1280,'y':0,'width':10/1280,'height':10/1280}],0,0)[0]
    for name in NAMES:
        e=training[name];image=cv2.imread(e['image_path']);assert image.shape[:2]==(1280,1280)
        labels=waid.parse_labels(Path(e['label_path']),['animal']);assert not labels['errors'];boxes=labels['boxes']
        full_labels=[[0,b['x']+b['width']/2,b['y']+b['height']/2,b['width'],b['height']] for b in boxes]
        full=write_item(Path(name).stem+'__full.png',cv2.resize(image,(1024,1024),interpolation=cv2.INTER_LINEAR),full_labels,source_name=name,kind='full',caption='Целый кадр → 1024 × 1024')
        preview.append(full);coverage=np.zeros((1280,1280),dtype=np.uint8);owned=[];contained=set()
        for y in POSITIONS:
            for x in POSITIONS:
                coverage[y:y+SIZE,x:x+SIZE]+=1
                lab,mapping=clipped_labels(boxes,x,y)
                tile=write_item(f'{Path(name).stem}__x{x}_y{y}.png',image[y:y+SIZE,x:x+SIZE],lab,source_name=name,kind='tile',origin_xy=[x,y],source_box_mapping=mapping,source_image_sha256=e['image_sha256'],source_label_sha256=e['label_sha256'])
                tiles.append(tile);owned.append(tile)
                contained.update(m['source_box'] for m in mapping if not m['truncated'])
                if not lab: background.append(tile)
        assert coverage.min()>=1
        assert contained==set(range(len(boxes))), 'Every animal must be fully visible in at least one tile for this lesson'
        chosen=max(owned,key=lambda t:(sum(not m['truncated'] for m in t['source_box_mapping']),-sum(m['truncated'] for m in t['source_box_mapping'])))
        native=cv2.imread(chosen['image_path']);native_labels=[[0,b['x']+b['width']/2,b['y']+b['height']/2,b['width'],b['height']] for b in waid.parse_labels(Path(chosen['label_path']),['animal'])['boxes']]
        shown=write_item(Path(name).stem+'__crop_input.png',cv2.resize(native,(1024,1024),interpolation=cv2.INTER_LINEAR),native_labels,source_name=name,kind='crop_input',origin_xy=chosen['origin_xy'],native_tile=chosen['name'],caption='Фрагмент 512 × 512 → вход 1024 × 1024')
        preview.append(shown)
        examples.append({'source_name':name,'tile':chosen['name'],'boxes':[{'source_box':m['source_box'],'full_input_wh':[boxes[m['source_box']]['width']*1024,boxes[m['source_box']]['height']*1024],'crop_input_wh':[boxes[m['source_box']]['width']*2560,boxes[m['source_box']]['height']*2560]} for m in chosen['source_box_mapping'] if not m['truncated']]})
    for n,t in enumerate(background[:2]):
        preview.append(write_item(f'background_{n+1}.png',cv2.resize(cv2.imread(t['image_path']),(1024,1024),interpolation=cv2.INTER_LINEAR),[],source_name=t['source_name'],kind='background',origin_xy=t['origin_xy'],caption='Фон: животных в целевой разметке нет'))
    payload={'source_dataset_revision':index['revision'],'source_names':NAMES,'tile_size':SIZE,'stride':384,'positions':list(POSITIONS),'network_input_demonstration':1024,'tiles':tiles,'entries':preview,'examples':examples,'scope':'Training-only preparation. No model predictions or new quality measurements. Fixed grid, not target-centered. Not a finalized training recipe.','truncated_label_policy':'Every positive intersection receives its clipped label; no unlabeled target fragments. All original animal boxes fully contained in at least one tile in this lesson.','created_utc':datetime.now(timezone.utc).isoformat()}
    payload['revision']=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
    assert all(sha(p)==v for p,v in protected.items())
    waid.save_json(OUT/'protected_artifacts.json',protected);waid.save_json(OUT/'index.json',payload)
    summary={'passed':True,'source_frames':len(NAMES),'source_boxes':sum(training[n]['reference_boxes'] for n in NAMES),'tiles':len(tiles),'positive_tiles':sum(t['reference_boxes']>0 for t in tiles),'background_tiles':sum(t['reference_boxes']==0 for t in tiles),'truncated_box_appearances':sum(m['truncated'] for t in tiles for m in t['source_box_mapping']),'all_targets_fully_contained_in_at_least_one_tile':True,'held_out_inputs_and_models_unchanged':True,'preview_items':len(preview),'training_started':False,'disk_bytes':sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())}
    waid.save_json(OUT/'verification.json',summary)
    (OUT/'README.md').write_text('# Учебный пример фрагментов\n\nПять исходных учебных кадров, без сдвигов из игрушечного опыта. Каждый разделён по постоянной сетке на девять фрагментов 512 × 512, шаг 384, перекрытие 128 пикселей. Сетка не зависит от разметки. Разметка нужна для переноса всех пересечённых рамок и для выбора наглядных примеров в просмотрщике.\n\nВ просмотре пары: целый кадр, затем один его фрагмент. Оба показаны как вход 1024 × 1024; относительный размер животного увеличивается в 2,5 раза по каждой оси. Интерполяция не создаёт новых исходных деталей. Последние две картинки — фон по целевой разметке.\n\nЭто подготовка и проверка координат, без обучения и измерений качества. Повторы рамок между перекрывающимися фрагментами не являются новыми животными. Позже поиск на проверочных кадрах должен обрабатывать всю фиксированную сетку, без выбора участков по правильным ответам. Девять вызовов сети могут сильно увеличить полное время обработки; нужный размер входа и число фрагментов предстоит выбрать по измерениям. Контрольные группы и исходные веса сохранены.\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False))
if __name__=='__main__':main()
