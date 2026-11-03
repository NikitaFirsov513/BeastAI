"""Read-only reference model comparison, two frozen development collections."""
from pathlib import Path
from collections import Counter
import hashlib,json
VARIANTS={'baseline':'Наша YOLO11n · вход1024','reference':'DroneMegaDetector · вход1280'}
class DroneReferencePreview:
    def __init__(self,root,parse_labels,split):
        if split not in ('control','fresh'):raise ValueError('Unknown reference split')
        self.root=Path(root);self.folder=self.root/'data/dronemegadetector';self.parse_labels=parse_labels;self.split=split
        self.index=json.loads((self.folder/'index.json').read_text(encoding='utf-8'));publication=json.loads((self.folder/'evaluation/published.json').read_text(encoding='utf-8'))
        p=self.folder/'evaluation/comparison.json';content=p.read_bytes();assert hashlib.sha256(content).hexdigest()==publication['report_sha256']
        self.report=json.loads(content);assert self.report['dataset_revision']==self.index['revision']==publication['dataset_revision']
        self.dataset_id='drone_reference' if split=='control' else 'drone_reference_fresh';self.ui_split='valid' if split=='control' else 'diagnostic'
        self.title='Готовая модель · прежний контроль' if split=='control' else 'Готовая модель · дополнительные кадры'
        self.entries=sorted(self.index['validation' if split=='control' else 'fresh_validation'],key=lambda e:(e['source']!='roe',e['name']));self.allowed_images={}
        for e in self.entries:
            if Path(e['name']).name!=e['name']:raise ValueError('Unsafe reference input name')
            for key,digest in [('image_path','image_sha256'),('label_path','label_sha256')]:
                p=Path(e[key]).resolve()
                if not p.is_relative_to(self.root) or hashlib.sha256(p.read_bytes()).hexdigest()!=e[digest]:raise ValueError('Reference input changed')
            for model in self.report['models'].values():
                r=model['items'][e['source']+'/'+e['name']];assert r['image_sha256']==e['image_sha256'] and r['label_sha256']==e['label_sha256']
            self.allowed_images[self.url(e)]=Path(e['image_path'])
    def url(self,e):return '/data/drone-reference/'+e['source']+'/'+e['name']
    def response(self,query):
        variant=query.get('variant',['reference'])[0]
        if variant not in VARIANTS:raise ValueError('Unknown reference variant')
        category=query.get('category',[''])[0];needle=query.get('q',[''])[0].lower();split=query.get('split',[self.ui_split])[0];categories={'roe':'roe_deer','sherbrooke':'red_deer'}
        selected=[e for e in self.entries if split in ('',self.ui_split) and needle in e['name'].lower() and (category in ('','wild','europe') or categories[e['source']]==category)]
        pages=max(1,(len(selected)+11)//12);page=min(pages-1,max(0,int(query.get('page',['0'])[0])));active=self.report['models'][variant];rows=[]
        for e in selected[page*12:(page+1)*12]:
            p=Path(e['label_path']);assert hashlib.sha256(p.read_bytes()).hexdigest()==e['label_sha256'];label=self.parse_labels(p,['animal']);r=active['items'][e['source']+'/'+e['name']];assert r['reference']==label['boxes']
            rows.append({'name':e['name'],'source':e['source'],'split':self.ui_split,'flight_id':None,'src':self.url(e),'prediction':{k:r[k] for k in ('boxes','counts','timings','original_size')},**label})
        experiments=[];latency=[]
        for source,title in [('roe','Косули'),('sherbrooke','Олени')]:
            if category not in ('','wild','europe',categories[source]):continue
            for key,label in VARIANTS.items():
                r=self.report['models'][key]['partitions'][self.split].get(source)
                if not r:continue
                q=r['quality'];t=r['timing_summary']['total_ms'];experiments.append({'label':title+' · '+label,'reference_boxes':q['reference_boxes'],'matched':q['tp'],'unmatched_reference':q['fn'],'unmatched_predictions':q['fp'],'small_fixed_reference_at_1024':q['small_reference'],'small_fixed_matched':q['small_matched'],'median_total_ms':t['median'],'p95_total_ms':t['p95'],'timing_images':q['images']})
            r=active['partitions'][self.split].get(source)
            if r:
                t=r['timing_summary']['total_ms'];latency.append(f"{title}: медиана {t['median']:.1f} мс, p95 {t['p95']:.1f} мс")
        scope='Это прежние данные разработки, без нового дообучения. Пересечение обучения готовой модели с нашими источниками не исключено; независимый итоговый тест и разделение косуль по полётам не подтверждены.'
        note='Порог0,25 зафиксирован заранее; одинаковое число не означает одинаковую калибровку уверенности моделей. Совпадение IoU≥0,5; один запуск сети на целый кадр, пакет1, FP32. Наша модель: вход1024 с сохранением пропорций; готовая: родной вход1280 с изменением размера до квадрата. Все малые цели закреплены по полному масштабу1024. Время всей цепочки измерено с чередованием моделей, три повтора, без параллельного обучения. p95 — время, быстрее которого прошли95% измерений.'
        mem=active['isolated_memory_probe'];additional='; '.join(latency)+f". Файл весов {active['model_file_bytes']/2**20:.2f} МиБ; параметров {mem['parameter_elements']:,}. Изолированная проба GPU на двух кадрах: {mem['gpu_peak_allocated_bytes']/2**20:.1f} МиБ выделено PyTorch; это не максимум по всем80 кадрам. Во временном сравнении обе модели находились в памяти. Энергия не измерялась. Основная модель сохранена."
        counts=Counter(categories[e['source']] for e in self.entries)
        return {'dataset':self.dataset_id,'dataset_title':self.title,'revision':self.index['revision'],'effective_split':self.ui_split,'variant':variant,'variant_options':[{'id':k,'label':v} for k,v in VARIANTS.items()],'category_names':{k:('Косули' if k=='roe_deer' else 'Благородные олени') for k in counts},'category_counts':dict(counts),'source_urls':['https://github.com/ConservationDronesAI/DroneMegaDetector','https://zenodo.org/records/18206908'],'collection_note':scope,'additional_findings':additional,'prediction_status_note':'Синие рамки: '+VARIANTS[variant]+'. Поиск по целому кадру без подсказки разметки.','baseline':None,'analysis':{'experiment_kind':'drone_reference','precision_available':True,'validation_images':len(self.entries),'scope_note':scope,'comparison_note':note,'experiments':experiments},'threshold_analysis':None,'wild_downloaded':len(self.entries),'downloaded':len(self.entries),'repository_total':len(self.entries),'total':len(selected),'page':page,'pages':pages,'items':rows,'related_group_options':[],'active_related_group':''}
