"""Read-only crop lesson, labels only; no invented prediction scores."""
from pathlib import Path
import hashlib,json
class CropLessonPreview:
    def __init__(self,root,parse_labels):
        self.root=Path(root);self.folder=self.root/'data/crop_lesson';self.parse_labels=parse_labels
        self.index=json.loads((self.folder/'index.json').read_text(encoding='utf-8'))
        assert json.loads((self.folder/'verification.json').read_text(encoding='utf-8'))['passed']
        self.entries=self.index['entries'];self.allowed_images={}
        for e in self.entries:
            assert Path(e['name']).name==e['name']
            for key,digest in [('image_path','image_sha256'),('label_path','label_sha256')]:
                p=Path(e[key]).resolve()
                if not p.is_relative_to(self.folder) or hashlib.sha256(p.read_bytes()).hexdigest()!=e[digest]:raise ValueError('Crop lesson changed')
            self.allowed_images['/data/crop-lesson/'+e['name']]=Path(e['image_path'])
    def response(self,query):
        needle=query.get('q',[''])[0].lower();category=query.get('category',[''])[0];split=query.get('split',['train'])[0]
        selected=[e for e in self.entries if needle in e['name'].lower() and category in ('','europe','wild','roe_deer') and split in ('','train')]
        rows=[]
        for e in selected:
            lp=Path(e['label_path']);assert hashlib.sha256(lp.read_bytes()).hexdigest()==e['label_sha256']
            label=self.parse_labels(lp,['animal'])
            rows.append({'name':e['name'],'source':'roe','split':'train','flight_id':None,'src':'/data/crop-lesson/'+e['name'],'prediction':None,'caption':e['caption'],**label})
        note='Подготовка учебных фрагментов: пять знакомых кадров, без обучения и предсказаний. Пары по порядку: целый кадр → его фрагмент. В конце — два участка фона по разметке. Зелёные рамки перенесены из исходной разметки.'
        return {'dataset':'crop_lesson','dataset_title':'Фрагменты · подготовка обучения','revision':self.index['revision'],'effective_split':'train','category_names':{'roe_deer':'Косули'},'category_counts':{'roe_deer':len(self.entries)},'source_urls':['https://zenodo.org/records/18206908'],'collection_note':note,'additional_findings':'Каждый исходный кадр разрезан на 9 перекрывающихся участков по одинаковой сетке. Пример: участок 512 × 512 растянут до входа 1024 × 1024; животное в 2,5 раза крупнее, чем при уменьшении целого кадра до 1024. Новых деталей это не создаёт. Проверочные кадры не использованы.','prediction_status_note':'Здесь только разметка. Новое обучение ещё не запускалось.','baseline':None,'analysis':None,'threshold_analysis':None,'downloaded':len(self.entries),'repository_total':len(self.entries),'wild_downloaded':len(self.entries),'total':len(rows),'page':0,'pages':1,'items':rows,'related_group_options':[],'active_related_group':''}
