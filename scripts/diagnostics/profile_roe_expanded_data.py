"""Record image and target scale distributions without changing labels or training."""
from project_paths import PROJECT_ROOT
from pathlib import Path
from collections import Counter,defaultdict
import json,statistics
from PIL import Image
import waid
ROOT=PROJECT_ROOT;OUT=ROOT/'data/roe_expanded'

def main():
    index=json.loads((OUT/'index.json').read_text(encoding='utf-8'));manifest=(OUT/'train_exposure.txt').read_text(encoding='utf-8').splitlines();exposure=Counter(str(Path(p).resolve()) for p in manifest);output={'dataset_revision':index['revision'],'small_reference_input':1024,'partitions':{}}
    for name,entries in [('training',index['training']),('old_control',index['validation']),('fresh_diagnostic',index['fresh_validation'])]:
        grouped=defaultdict(list)
        for e in entries:grouped[e['source']].append(e)
        summary={}
        for source,rows in grouped.items():
            widths=[];heights=[];bins=Counter();shapes=Counter();empty=0;exposed_boxes=0
            for e in rows:
                with Image.open(e['image_path']) as im:w,h=im.size
                shapes[f'{w}x{h}']+=1;scale=min(1024/w,1024/h);truth=waid.parse_labels(Path(e['label_path']),['animal'])['boxes'];empty+=not truth
                assert len(truth)==e['reference_boxes']
                if name=='training':exposed_boxes+=exposure[str(Path(e['image_path']).resolve())]*len(truth)
                for b in truth:
                    width=b['width']*w*scale;height=b['height']*h*scale;area=width*height;widths.append(width);heights.append(height);bins['below16' if area<16**2 else '16to32' if area<32**2 else '32to64' if area<64**2 else 'atleast64']+=1
            summary[source]={'unique_images':len(rows),'images_without_target':empty,'target_boxes':len(widths),'source_shapes':dict(shapes),'size_groups_at1024':dict(bins),'median_box_width_at1024':statistics.median(widths) if widths else None,'median_box_height_at1024':statistics.median(heights) if heights else None,'minimum_box_width_at1024':min(widths,default=None),'minimum_box_height_at1024':min(heights,default=None)}
            if name=='training':summary[source]['target_box_presentations_per_epoch']=exposed_boxes
        output['partitions'][name]=summary
    waid.save_json(OUT/'data_profile.json',output)
    print(json.dumps(output,ensure_ascii=True))

if __name__=='__main__':main()
