from pathlib import Path
import json,math
from PIL import Image,ImageDraw,ImageFont
import waid,baseline
waid.configure_dataset('bambi');index=waid.get_index();report=json.loads((waid.DATA/'baseline/latest.json').read_text())
folder=waid.DATA/'review';folder.mkdir(exist_ok=True)
font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',16)
smallfont=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',13)
tiles=[];records=[]
for pair in index['pairs']:
    if pair['split']!='valid':continue
    image=Image.open(waid.local_file(pair['image'])).convert('RGB');w,h=image.size
    truth=waid.parse_labels(waid.local_file(pair['label']),index['classes'])['boxes']
    preds=report['items']['valid/'+pair['name']]['boxes']
    for number,box in enumerate(truth):
        cx=(box['x']+box['width']/2)*w;cy=(box['y']+box['height']/2)*h
        side=max(160,math.ceil(max(box['width']*w,box['height']*h)*2))
        left=max(0,min(w-side,round(cx-side/2)));top=max(0,min(h-side,round(cy-side/2)))
        crop=image.crop((left,top,left+side,top+side)).resize((300,300))
        draw=ImageDraw.Draw(crop);scale=300/side
        def draw_box(b,color,line=2):
            coords=[(b['x']*w-left)*scale,(b['y']*h-top)*scale,((b['x']+b['width'])*w-left)*scale,((b['y']+b['height'])*h-top)*scale]
            draw.rectangle(coords,outline=color,width=line)
        for pred in preds:draw_box(pred,'#69b9ff')
        draw_box(box,'#80ff9b',3)
        tile=Image.new('RGB',(320,350),'#101719');tile.paste(crop,(10,38));d=ImageDraw.Draw(tile)
        title=f"F{pair['flight_id']} / frame {pair['frame_index']} / box {number+1}"
        d.text((10,7),title,font=font,fill='white')
        d.text((10,334),f"GT {box['width']*w:.0f} x {box['height']*h:.0f} px",font=smallfont,fill='#b5c6bd')
        tiles.append(tile);records.append({'name':pair['name'],'box_index':number,'flight':pair['flight_id'],'crop':[left,top,side]})
for page in range(math.ceil(len(tiles)/6)):
    sheet=Image.new('RGB',(960,700),'#101719')
    for offset,tile in enumerate(tiles[page*6:page*6+6]):sheet.paste(tile,((offset%3)*320,(offset//3)*350))
    sheet.save(folder/f'validation-crops-{page+1}.jpg',quality=92)
(folder/'crop-index.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
print(json.dumps({'reference_crops':len(tiles),'pages':math.ceil(len(tiles)/6)}))