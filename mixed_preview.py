"""Read-only viewer collection for the fixed before/after mixed experiment."""
from pathlib import Path
import hashlib,json

class MixedPreview:
    def __init__(self,root,parse_labels,experiment="mixed_pilot"):
        if experiment not in ("mixed_pilot","roe_focus"):raise ValueError("Unknown experiment")
        self.dataset_id="roe_focus" if experiment=="roe_focus" else "mixed"
        self.focus=experiment=="roe_focus"
        self.root=Path(root);self.parse_labels=parse_labels
        folder=self.root/'data'/experiment;self.index=json.loads((folder/'index.json').read_text(encoding='utf-8'))
        self.publication=json.loads((folder/'evaluation/published.json').read_text(encoding='utf-8'))
        assert self.publication['dataset_revision']==self.index['revision']
        self.stages=self.publication.get('stages',['before','after']);self.reports={}
        if self.stages not in (['before','after'],['before','previous','after']):raise ValueError('Unexpected comparison stages')
        for stage in self.stages:
            path=(self.root/self.publication[stage+'_path']).resolve()
            if not path.is_relative_to((folder/'evaluation').resolve()):raise ValueError('Unsafe comparison report path')
            content=path.read_bytes()
            if hashlib.sha256(content).hexdigest()!=self.publication[stage+'_sha256']:raise ValueError('Published report changed')
            report=json.loads(content);assert report['dataset_revision']==self.index['revision']
            self.reports[stage]=report
        self.entries=sorted(self.index['validation'],key=lambda e:(e['source']!='roe',e['name']));self.allowed_images={}
        for e in self.entries:
            key=e['source']+'/'+e['name']
            for field,sha_field in [('image_path','image_sha256'),('label_path','label_sha256')]:
                path=Path(e[field]).resolve()
                if not path.is_relative_to(self.root):raise ValueError('Control file outside project')
                if hashlib.sha256(path.read_bytes()).hexdigest()!=e[sha_field]:raise ValueError('Control file changed')
            for stage in self.stages:
                row=self.reports[stage]['items'][key]
                if row['image_sha256']!=e['image_sha256'] or row['label_sha256']!=e['label_sha256']:raise ValueError('Predictions belong to another control')
            self.allowed_images['/data/mixed-control/'+key]=Path(e['image_path'])

    def response(self,query):
        category=query.get('category',[''])[0];needle=query.get('q',[''])[0].lower();split=query.get('split',['valid'])[0]
        categories={'sherbrooke':'red_deer','roe':'roe_deer'}
        entries=[e for e in self.entries if needle in e['name'].lower() and split in ('','valid') and (category in ('','europe','wild') or categories[e['source']]==category)]
        page=max(0,int(query.get('page',['0'])[0]));pages=max(1,(len(entries)+11)//12);page=min(page,pages-1)
        rows=[]
        for e in entries[page*12:(page+1)*12]:
            key=e['source']+'/'+e['name'];result=self.reports['after']['items'][key]
            label=self.parse_labels(Path(e['label_path']),['animal'])
            if hashlib.sha256(Path(e['label_path']).read_bytes()).hexdigest()!=e['label_sha256']:raise ValueError('Control label changed')
            rows.append({'name':e['name'],'source':e['source'],'split':'valid','flight_id':None,'src':'/data/mixed-control/'+key,'prediction':{'boxes':result['boxes'],'counts':result['counts'],'timings':result['timings'],'original_size':result['original_size']},**label})
        experiments=[]
        for source,title in [('roe','Косули'),('sherbrooke','Олени')]:
            if category not in ('','europe','wild',categories[source]):continue
            labels={'before':'исходные веса','previous':'прошлый опыт · 17% косуль','after':'равная доля · 50% косуль'} if self.focus else {'before':'прежние веса','after':'после дообучения'}
            for stage in self.stages:
                label=labels[stage]
                r=self.reports[stage]['sources'][source];q=r['quality']
                experiments.append({'label':title+' · '+label,'reference_boxes':q['reference_boxes'],'matched':q['tp'],'unmatched_reference':q['fn'],'unmatched_predictions':q['fp'],'small_fixed_reference_at_1024':q['small_reference'],'small_fixed_matched':q['small_matched'],'median_total_ms':r['timing_summary']['total_ms']['median'],'timing_images':q['images']})
        after=self.reports['after'];baseline={'model':('YOLO11n · равная доля косуль и оленей, 10 эпох' if self.focus else 'YOLO11n · смешанное дообучение, опыт 10 эпох'),'conditions':after['conditions'],'timing_summary':after['timing_summary'],'validation_preview':None}
        analysis={'experiment_kind':'mixed','precision_available':True,'dataset_revision':self.index['revision'],'validation_images':64,'scope_note':'Те же 32 контрольных кадра оленей и 32 диагностических кадра косуль. Таблица относится ко всем кадрам выбранного источника. Это проверка разработки; независимость по полётам не подтверждена.','comparison_note':'Синие рамки внизу — результат новых весов. Обе модели заново измерены при входе 1024, пороге 0,25, batch 1, FP32, IoU сопоставления 0,5 и одинаковом фильтре повторов. Малые животные: площадь рамки меньше 32² при входе 1024. Ноль ложных рамок при отсутствии ответов не означает хорошую модель. Время включает чтение и обработку кадра; питание и частоты не фиксировались, энергия не измерялась. Старые веса сохранены; это кандидат для сравнения.','experiments':experiments}
        return {'dataset':self.dataset_id,'dataset_title':('Равная доля косуль — сравнение трёх моделей' if self.focus else 'Смешанное дообучение — контроль до/после'),'revision':self.index['revision'],'effective_split':'valid','category_names':{'roe_deer':'Косули','red_deer':'Благородные олени'},'category_counts':{'roe_deer':32,'red_deer':32},'source_urls':['https://zenodo.org/records/18206908'],'wild_downloaded':64,'baseline':baseline,'analysis':analysis,'threshold_analysis':None,'total':len(entries),'downloaded':64,'repository_total':64,'page':page,'pages':pages,'items':rows,'related_group_options':[],'active_related_group':''}
