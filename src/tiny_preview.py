"""Read-only five-image memorization viewer; explicitly not held-out quality."""
from pathlib import Path
import hashlib,json

class TinyMemorizationPreview:
    def __init__(self,root,parse_labels,experiment="tiny_memorization"):
        if experiment not in ("tiny_memorization","tiny_grid_shift"):raise ValueError("Unknown tiny experiment")
        self.dataset_id=experiment;self.shifted=experiment=="tiny_grid_shift"
        self.prefix="/data/tiny-grid-shift/" if self.shifted else "/data/tiny-memorization/"
        self.root=Path(root);self.parse_labels=parse_labels;self.folder=self.root/'data'/experiment
        self.index=json.loads((self.folder/'index.json').read_text(encoding='utf-8'));self.entries=self.index['entries']
        self.reports={stage:json.loads((self.folder/(stage+'.json')).read_text(encoding='utf-8')) for stage in ('before','after')}
        self.verification=json.loads((self.folder/'input_verification.json').read_text(encoding='utf-8'));assert self.verification['passed']
        self.allowed_images={}
        for e in self.entries:
            assert Path(e['name']).name==e['name']
            for key,digest in [('image_path','image_sha256'),('label_path','label_sha256')]:
                p=Path(e[key]).resolve()
                if not p.is_relative_to(self.folder) or hashlib.sha256(p.read_bytes()).hexdigest()!=e[digest]:raise ValueError('Tiny diagnostic input changed')
            for report in self.reports.values():
                assert report['dataset_revision']==self.index['revision'];row=report['items'][e['name']]
                assert row['image_sha256']==e['image_sha256'] and row['label_sha256']==e['label_sha256']
            self.allowed_images[self.prefix+e['name']]=Path(e['image_path'])
    def response(self,query):
        split=query.get('split',['train'])[0];needle=query.get('q',[''])[0].lower();category=query.get('category',[''])[0]
        selected=[e for e in self.entries if split in ('','train') and category in ('','europe','wild','roe_deer') and needle in e['name'].lower()]
        rows=[]
        for e in selected:
            label=self.parse_labels(Path(e['label_path']),['animal']);r=self.reports['after']['items'][e['name']]
            assert r['reference']==label['boxes'] and hashlib.sha256(Path(e['label_path']).read_bytes()).hexdigest()==e['label_sha256']
            rows.append({'name':e['name'],'source':'roe','split':'train','flight_id':None,'src':self.prefix+e['name'],'prediction':{'boxes':r['boxes'],'counts':r['counts'],'timings':[],'original_size':[1024,1024]},**label})
        experiments=[]
        for stage,label in [('before','До диагностического обучения'),('after','После 200 коротких проходов')]:
            r=self.reports[stage];q=r['counts'];g=r['groups'];experiments.append({'label':label,'reference_boxes':q['tp']+q['fn'],'matched':q['tp'],'unmatched_reference':q['fn'],'unmatched_predictions':q['fp'],'small_fixed_reference_at_1024':g['tiny_zero_candidates']['references']+g['tiny_with_candidates']['references'],'small_fixed_matched':g['tiny_zero_candidates']['matched']+g['tiny_with_candidates']['matched'],'median_total_ms':r['timing_summary']['total_ms']['median'],'timing_images':5})
        after=self.reports['after'];g=after['groups'];groups_note=f"После опыта: из маленьких целей без допустимых опорных точек найдено {g['tiny_zero_candidates']['matched']}/{g['tiny_zero_candidates']['references']}; с допустимыми точками {g['tiny_with_candidates']['matched']}/{g['tiny_with_candidates']['references']}; крупный контроль {g['large_control']['matched']}/{g['large_control']['references']}. Условия фиксированы; это не утверждение о любых преобразованиях или новых кадрах."
        if self.shifted:groups_note=f"После согласованного сдвига: из пяти целей, не имевших опорных точек в исходном положении, найдено {g['tiny_zero_candidates']['matched']}/5; из трёх исходно поддержанных — {g['tiny_with_candidates']['matched']}/3; крупный контроль — {g['large_control']['matched']}/1. Теперь допустимые точки есть у всех целей. Сдвиг на целые пиксели сохраняет размер и значения пикселей животных."
        note='Те же пять снимков использовались для обучения и оценки усвоения. Это НЕ проверка качества на новых данных. Случайные преобразования отключены. Синие рамки — диагностические веса; основная модель не заменена.'
        baseline={'model':'YOLO11n · диагностика усвоения пяти снимков','conditions':after['conditions'],'timing_summary':after['timing_summary'],'validation_preview':None}
        analysis={'experiment_kind':'memorization','precision_available':True,'dataset_revision':self.index['revision'],'validation_images':0,'training_evaluation_images':5,'scope_note':note,'comparison_note':'Порог0,25, IoU сопоставления0,5, вход1024, batch1, FP32. Это учебные ошибки, а не оценка обобщения. '+groups_note,'experiments':experiments}
        return {'dataset':self.dataset_id,'dataset_title':('Диагностика · согласованный сдвиг на 1 пиксель' if self.shifted else 'Диагностика усвоения · 5 учебных кадров'),'revision':self.index['revision'],'effective_split':'train','category_names':{'roe_deer':'Косули'},'category_counts':{'roe_deer':5},'source_urls':['https://zenodo.org/records/18206908'],'collection_note':note,'additional_findings':groups_note,'baseline':baseline,'analysis':analysis,'threshold_analysis':None,'downloaded':5,'repository_total':5,'wild_downloaded':5,'total':len(rows),'page':0,'pages':1,'items':rows,'related_group_options':[],'active_related_group':''}
