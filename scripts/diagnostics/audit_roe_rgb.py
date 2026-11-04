"""Audit the roe-deer RGB COCO dataset and fetch a fixed 32-image sample via bounded ranges."""
from project_paths import PROJECT_ROOT
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path
import hashlib,io,json,random,statistics,urllib.request,zipfile,zlib
from PIL import Image
from remote_zip import RemoteZipReader
import waid
ROOT=PROJECT_ROOT;DATA=ROOT/'data/roe';OUT=DATA/'audit'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    with urllib.request.urlopen('https://zenodo.org/api/records/18206908',timeout=35) as response:
        metadata=json.loads(response.read(2*1024**2))
    waid.save_json(OUT/'source_record.json',metadata)
    source=next(f for f in metadata['files'] if f['key']=='Dataset_young_deer_RGB.zip')
    cached=json.loads((ROOT/'data/dataset_candidates/roe_rgb.json').read_text(encoding='utf-8'))['files'][0]
    assert source['size']==cached['size'] and source['checksum']==cached['checksum'],'Source archive revision changed'
    coco=json.loads((DATA/'source_metadata/annotation_coco.json').read_text(encoding='utf-8'))
    members=json.loads((DATA/'members.json').read_text(encoding='utf-8'))
    archive_names={m['name'] for m in members}
    images={im['id']:im for im in coco['images']};categories={c['id']:c['name'] for c in coco['categories']}
    assert len(images)==len(coco['images']) and set(categories)=={1,2}
    by_image=defaultdict(list);errors=Counter();examples=[];class_counts=Counter();attributes=Counter();target_widths=[];target_heights=[];target_min_sides=[];poly_bbox_checks=poly_bbox_mismatches=0
    ann_ids=set()
    for ann in coco['annotations']:
        issue=[];iid=ann['image_id'];category=ann['category_id']
        if ann['id'] in ann_ids:issue.append('duplicate_annotation_id')
        ann_ids.add(ann['id'])
        if iid not in images:issue.append('unknown_image_id')
        if category not in categories:issue.append('unknown_category_id')
        if ann.get('bbox_mode')!=1:issue.append('bbox_mode_not_XYWH')
        x,y,w,h=ann['bbox']
        if w<=0 or h<=0:issue.append('non_positive_box')
        if iid in images and (x<0 or y<0 or x+w>images[iid]['width']+1e-6 or y+h>images[iid]['height']+1e-6):issue.append('box_outside_image')
        if ann.get('iscrowd',0):issue.append('crowd_annotation_requires_policy')
        class_counts[category]+=1;by_image[iid].append(ann)
        attributes.update((ann.get('attributes') or {}).keys())
        if category==1 and iid in images:
            scale=min(1024/images[iid]['width'],1024/images[iid]['height'])
            target_widths.append(w*scale);target_heights.append(h*scale);target_min_sides.append(min(w,h)*scale)
        polygons=ann.get('segmentation')
        if isinstance(polygons,list) and polygons and all(isinstance(p,list) for p in polygons):
            coords=[v for poly in polygons for v in poly]
            if coords and len(coords)%2==0:
                xs,ys=coords[::2],coords[1::2];poly_bbox_checks+=1
                if min(xs)<x-2 or max(xs)>x+w+2 or min(ys)<y-2 or max(ys)>y+h+2:poly_bbox_mismatches+=1
        errors.update(issue)
        if issue and len(examples)<50:examples.append({'annotation_id':ann['id'],'image_id':iid,'issues':issue})
    missing_images=[im['file_name'] for im in images.values() if im['file_name'] not in archive_names]
    bad_dimensions=[im['id'] for im in images.values() if im['width']<=0 or im['height']<=0]
    group_fields={'flight_id','video_id','sequence_id','original_id','source_id','track_id'}
    provenance={'image_group_fields_found':sorted(group_fields.intersection(k for im in images.values() for k in im)),'annotation_group_fields_found':sorted(group_fields.intersection(k for ann in coco['annotations'] for k in ann)),'attribute_keys':dict(attributes),'nonzero_capture_dates':sum(bool(im.get('date_captured')) for im in images.values()),'nonempty_source_urls':sum(bool(im.get('coco_url') or im.get('flickr_url')) for im in images.values())}
    rng=random.Random(20261101);chosen={im['id'] for im in coco['images'][:8]}
    reasons={iid:['first_eight_neighboring_archive_names'] for iid in chosen}
    bins={'below_16':lambda n:n<16,'16_to_32':lambda n:16<=n<32,'32_to_64':lambda n:32<=n<64,'at_least_64':lambda n:n>=64}
    for name,check in bins.items():
        pool=[]
        for iid,anns in by_image.items():
            im=images[iid];scale=min(1024/im['width'],1024/im['height'])
            if iid not in chosen and any(a['category_id']==1 and check(min(a['bbox'][2:])*scale) for a in anns):pool.append(iid)
        for iid in rng.sample(sorted(pool),min(4,len(pool))):chosen.add(iid);reasons[iid]=[name+'_at_full_input_1024']
    for name,pool in [('no_annotations',[iid for iid in images if not by_image[iid]]),('person_only',[iid for iid,anns in by_image.items() if anns and all(a['category_id']==2 for a in anns)])]:
        pool=sorted(iid for iid in pool if iid not in chosen)
        for iid in rng.sample(pool,min(2,len(pool))):chosen.add(iid);reasons[iid]=[name]
    remaining=sorted(set(images)-chosen)
    for iid in rng.sample(remaining,32-len(chosen)):chosen.add(iid);reasons[iid]=['random_fill']
    selection=[{'image_id':iid,'name':images[iid]['file_name'],'reasons':reasons[iid]} for iid in sorted(chosen)]
    frozen=OUT/'selection.json'
    if frozen.exists():assert json.loads(frozen.read_text(encoding='utf-8'))==selection
    else:waid.save_json(frozen,selection)
    reader=RemoteZipReader(source['links']['self'],source['size'],budget=64*1024**2)
    sample=[];label_bytes={};used=[]
    with zipfile.ZipFile(reader) as archive:
        infos={i.filename:i for i in archive.infolist()}
        for number,item in enumerate(selection,1):
            iid=item['image_id'];im=images[iid];name=im['file_name']
            assert Path(name).name==name and not Path(name).is_absolute()
            dest=OUT/'images'/name;dest.parent.mkdir(exist_ok=True)
            if dest.exists():content=dest.read_bytes()
            else:
                content=archive.read(name);temp=dest.with_suffix('.jpg.part');temp.write_bytes(content);temp.replace(dest)
            assert zlib.crc32(content)&0xffffffff==infos[name].CRC,'Cached JPEG differs from source ZIP member'
            with Image.open(io.BytesIO(content)) as decoded:
                decoded.load();assert decoded.size==(im['width'],im['height'])
                exif=decoded.getexif();rgb=decoded.convert('RGB');pixel_sha=hashlib.sha256(rgb.tobytes()).hexdigest()
                exif_info={'tag_count':len(exif),'has_datetime':any(exif.get(k) for k in (306,36867,36868)),'has_gps':bool(exif.get(34853))}
                mode=decoded.mode
            anns=by_image[iid];rows=[];sample_issues=[]
            for ann in anns:
                if ann['category_id']!=1:continue
                x,y,w,h=ann['bbox']
                if not (w>0 and h>0 and x>=0 and y>=0 and x+w<=im['width'] and y+h<=im['height']):sample_issues.append(ann['id']);continue
                rows.append(f"0 {(x+w/2)/im['width']:.8f} {(y+h/2)/im['height']:.8f} {w/im['width']:.8f} {h/im['height']:.8f}")
            label=('\n'.join(rows)+ ('\n' if rows else '')).encode('utf-8')
            label_path=OUT/'labels'/Path(name).with_suffix('.txt').name;label_path.parent.mkdir(exist_ok=True);label_path.write_bytes(label)
            sample.append({'name':name,'image_id':iid,'size_bytes':len(content),'sha256':hashlib.sha256(content).hexdigest(),'decoded_rgb_sha256':pixel_sha,'source_member_crc32':infos[name].CRC,'source_zip_modification_time':list(infos[name].date_time),'dimensions':[im['width'],im['height']],'decoded_mode':mode,'exif':exif_info,'class_counts':dict(Counter(a['category_id'] for a in anns)),'target_labels':len(rows),'invalid_target_annotation_ids':sample_issues,'label_sha256':hashlib.sha256(label).hexdigest()})
            print(f'[{number}/32] decoded and labels checked: {name}',flush=True)
    exact_groups=defaultdict(list);pixel_groups=defaultdict(list)
    for item in sample:exact_groups[item['sha256']].append(item['name']);pixel_groups[item['decoded_rgb_sha256']].append(item['name'])
    summary={'source_record':'https://zenodo.org/records/18206908','source_project':'https://fly4future.com/TQ16000038-v7','source_version':metadata.get('metadata',{}).get('version'),'source_checksum':source['checksum'],'license':metadata['metadata'].get('license'),'images':len(images),'annotations':len(coco['annotations']),'class_counts':{categories[k]:v for k,v in class_counts.items()},'image_dimension_counts':{str(k):v for k,v in Counter((im['width'],im['height']) for im in images.values()).items()},'images_with_target':sum(any(a['category_id']==1 for a in by_image[iid]) for iid in images),'images_without_target':sum(not any(a['category_id']==1 for a in by_image[iid]) for iid in images),'empty_images':sum(not by_image[iid] for iid in images),'missing_archive_image_count':len(missing_images),'bad_dimensions':len(bad_dimensions),'annotation_errors':dict(errors),'annotation_error_examples':examples,'polygon_bbox_checks':poly_bbox_checks,'polygon_bbox_mismatches_tolerance_2px':poly_bbox_mismatches,'provenance':provenance,'target_box_at_full_input_1024':{'count':len(target_min_sides),'median_width':statistics.median(target_widths),'median_height':statistics.median(target_heights),'min_short_side':min(target_min_sides),'median_short_side':statistics.median(target_min_sides),'short_side_below_16':sum(v<16 for v in target_min_sides),'short_side_below_32':sum(v<32 for v in target_min_sides)},'sample_images':len(sample),'sample_image_bytes':sum(i['size_bytes'] for i in sample),'sample_exif_datetime_count':sum(i['exif']['has_datetime'] for i in sample),'sample_exif_gps_count':sum(i['exif']['has_gps'] for i in sample),'sample_exact_duplicate_groups':[g for g in exact_groups.values() if len(g)>1],'sample_decoded_pixel_duplicate_groups':[g for g in pixel_groups.values() if len(g)>1],'network_archive_bytes_received_this_run':reader.bytes_received,'network_range_requests_this_run':reader.requests,'full_archive_bytes':source['size'],'sample':sample,'limitations':['Annotation geometry is checked, not independent semantic correctness or exhaustiveness.','Sample is stratified and deliberately enriched; not a representative model benchmark.','Missing flight/source IDs prevent an audited independent random image train/test split of the augmented collection.','No near-duplicate absence claim: exact duplicate checks cover only the 32 downloaded samples.','No video timing or per-frame track IDs: this sample cannot validate event recall or false events per minute.','The person category remains in source COCO but is excluded from the one-class animal YOLO labels.','No model training or threshold selection was performed on these samples.']}
    targets=[ann for ann in coco['annotations'] if ann['category_id']==1]
    summary['target_box_at_full_input_1024'].update(small_bbox_area_below_32_squared=sum(ann['bbox'][2]*ann['bbox'][3]*min(1024/images[ann['image_id']]['width'],1024/images[ann['image_id']]['height'])**2<32**2 for ann in targets),source_min_width=min(ann['bbox'][2] for ann in targets),source_min_height=min(ann['bbox'][3] for ann in targets))
    summary['occlusion_attribute_values']=dict(Counter(str(ann.get('attributes',{}).get('occluded')) for ann in coco['annotations']))
    summary['sample_target_boxes']=sum(item['target_labels'] for item in sample)
    summary['sample_person_boxes']=sum(item['class_counts'].get(2,0) for item in sample)
    summary['sample_target_positive_images']=sum(item['target_labels']>0 for item in sample)
    summary['description_count_discrepancy']={'declared_fawn':43298,'actual_fawn':class_counts[1],'declared_person':3080,'actual_person':class_counts[2],'note':'Counts differ by one annotation per class; no annotation was invented or removed to match the prose.'}
    waid.save_json(OUT/'summary.json',summary)
    subcoco={k:v for k,v in coco.items() if k not in ('images','annotations')};subcoco.update(images=[images[iid] for iid in sorted(chosen)],annotations=[a for a in coco['annotations'] if a['image_id'] in chosen])
    waid.save_json(OUT/'sample_coco.json',subcoco)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('sample','annotation_error_examples')},ensure_ascii=True),flush=True)

if __name__=='__main__':main()
