"""Read-only collections for expanded training images, old controls and new diagnostics."""
from pathlib import Path
from collections import Counter
import hashlib,json

class ExpandedPreview:
    def __init__(self,root,parse_labels,mode):
        if mode not in ('train','control','fresh'):raise ValueError('Unknown expanded collection')
        self.root=Path(root);self.parse_labels=parse_labels;self.mode=mode;self.folder=self.root/'data/roe_expanded'
        self.index=json.loads((self.folder/'index.json').read_text(encoding='utf-8'))
        self.entries=self.index[{'train':'roe_training','control':'validation','fresh':'fresh_validation'}[mode]]
        self.entries=sorted(self.entries,key=lambda e:(e['source']!='roe',e['name']))
        self.dataset_id={'train':'roe_expanded_train','control':'roe_expanded','fresh':'roe_expanded_fresh'}[mode]
        self.split={'train':'train','control':'valid','fresh':'diagnostic'}[mode]
        self.title={'train':f'Расширенные учебные косули ({len(self.entries)})','control':'Больше данных — прежний контроль','fresh':f'Больше данных — новые отложенные кадры ({len(self.entries)})'}[mode]
        self.allowed_images={};self.reports={};self.active=None;self.training_fit=None
        fit_path=self.folder/'evaluation/training_fit_diagnostic.json'
        if mode=='train' and fit_path.exists():
            fit=json.loads(fit_path.read_text(encoding='utf-8'));assert fit['dataset_revision']==self.index['revision'];self.training_fit=fit['models']['after']
        publication=self.folder/'evaluation/published.json'
        if publication.exists() and mode!='train':
            p=json.loads(publication.read_text(encoding='utf-8'));assert p['dataset_revision']==self.index['revision']
            for stage,entry in p['reports'][mode].items():
                path=(self.root/entry['path']).resolve()
                if not path.is_relative_to((self.folder/'evaluation').resolve()):raise ValueError('Unsafe report path')
                content=path.read_bytes()
                if hashlib.sha256(content).hexdigest()!=entry['sha256']:raise ValueError('Expanded report changed')
                report=json.loads(content);assert report['dataset_revision']==self.index['revision'];self.reports[stage]=report
            self.active=p['active_stage'] if p['active_stage'] in self.reports else 'after'
        for e in self.entries:
            if Path(e['name']).name!=e['name']:raise ValueError('Unsafe image name')
            for field,digest in [('image_path','image_sha256'),('label_path','label_sha256')]:
                path=Path(e[field]).resolve()
                if not path.is_relative_to(self.root) or hashlib.sha256(path.read_bytes()).hexdigest()!=e[digest]:raise ValueError('Expanded input changed')
            key=e['source']+'/'+e['name']
            for report in self.reports.values():
                row=report['items'][key]
                if row['image_sha256']!=e['image_sha256'] or row['label_sha256']!=e['label_sha256']:raise ValueError('Expanded prediction belongs to different inputs')
            if self.training_fit:
                row=self.training_fit['items'][e['name']]
                if row['image_sha256']!=e['image_sha256'] or row['label_sha256']!=e['label_sha256']:raise ValueError('Training diagnostic belongs to another input')
            self.allowed_images[self.url(e)]=Path(e['image_path'])

    def url(self,e):return '/data/roe-expanded/'+self.mode+'/'+e['source']+'/'+e['name']

    def response(self,query):
        category=query.get('category',[''])[0];needle=query.get('q',[''])[0].lower();split=query.get('split',[self.split])[0]
        categories={'roe':'roe_deer','sherbrooke':'red_deer'}
        selected=[e for e in self.entries if needle in e['name'].lower() and split in ('',self.split) and (category in ('','europe','wild') or categories[e['source']]==category)]
        pages=max(1,(len(selected)+11)//12);page=min(pages-1,max(0,int(query.get('page',['0'])[0])))
        rows=[]
        for e in selected[page*12:(page+1)*12]:
            label=self.parse_labels(Path(e['label_path']),['animal'])
            if hashlib.sha256(Path(e['label_path']).read_bytes()).hexdigest()!=e['label_sha256']:raise ValueError('Expanded labels changed')
            prediction=None
            if self.active:
                result=self.reports[self.active]['items'][e['source']+'/'+e['name']]
                if result['reference']!=label['boxes']:raise ValueError('Expanded references differ from report')
                prediction={k:result[k] for k in ('boxes','counts','timings','original_size')}
            if self.training_fit:
                saved=self.training_fit['items'][e['name']];assert saved['reference']==label['boxes'];prediction={'boxes':saved['predictions'],'counts':saved['counts'],'timings':[],'original_size':e.get('dimensions',[1280,1280])}
            rows.append({'name':e['name'],'source':e['source'],'split':self.split,'src':self.url(e),'flight_id':None,'prediction':prediction,**label})
        baseline=analysis=None
        note='Новые учебные снимки отобраны после исключения найденных связей с контролем и повторов. Здесь показана разметка; качество оценивается отдельно. Независимость по полётам не подтверждена.' if self.mode=='train' else ('Новые группы отложены до обучения и не входили в него. Это дополнительная диагностика; независимость по полётам не подтверждена.' if self.mode=='fresh' else 'Те же 32 кадра оленей и 32 кадра косуль. Они сохранены для сравнения опытов; это повторно используемый контроль разработки.')
        if self.active:
            active=self.reports[self.active];baseline={'model':'YOLO11n · расширенный набор · вход '+str(active['conditions']['input_shape'][2]),'conditions':active['conditions'],'timing_summary':active['timing_summary'],'validation_preview':None}
            experiments=[]
            names={'before':'прежние веса · 27 учебных косуль','after':'больше данных · вход1024','after_1280':'больше данных · вход1280'}
            for source,title in [('roe','Косули'),('sherbrooke','Олени')]:
                if category not in ('','europe','wild',categories[source]):continue
                for stage,report in self.reports.items():
                    if source not in report['sources']:continue
                    r=report['sources'][source];q=r['quality']
                    experiments.append({'label':title+' · '+names[stage],'reference_boxes':q['reference_boxes'],'matched':q['tp'],'unmatched_reference':q['fn'],'unmatched_predictions':q['fp'],'small_fixed_reference_at_1024':q['small_reference'],'small_fixed_matched':q['small_matched'],'median_total_ms':r['timing_summary']['total_ms']['median'],'timing_images':q['images']})
            analysis={'experiment_kind':'expanded','precision_available':True,'dataset_revision':self.index['revision'],'validation_images':len(self.entries),'scope_note':note+' Таблица относится ко всему выбранному источнику.','comparison_note':'Синие рамки — выбранный вариант расширенной модели. Порог0,25, совпадение IoU≥0,5, batch1, FP32. Группа малых объектов закреплена по входу1024 и не меняется при входе1280. Полная задержка включает чтение и обработку; энергия не измерялась. Цель по наблюдениям животных в видео здесь не проверяется.','experiments':experiments}
        prediction_note=None
        if self.mode=='train':
            prediction_note='Это учебные примеры. Здесь пока показана только разметка; контрольные результаты появятся в отдельных подборках.'
            if self.training_fit:
                q=self.training_fit;prediction_note=f"Синие рамки — усвоение знакомых учебных примеров: найдено {q['counts']['tp']} из {q['reference_boxes']}. Это не оценка качества на новых данных."
        state_path=self.folder/'training_status.json'
        if not self.active and not self.training_fit and state_path.exists():
            state=json.loads(state_path.read_text(encoding='utf-8'))
            if state.get('state')=='training':
                progress=f"Обучение: завершено {state.get('epoch_completed',0)} из {state.get('epochs',20)} эпох. "
                prediction_note=progress+(prediction_note or 'Предсказания появятся после завершения автоматической проверки.')
            elif state.get('state')=='completed':prediction_note='Обучение завершено; выполняется автоматическая проверка. Предсказания появятся после публикации.'
        additional_findings=''
        external_path=self.folder/'evaluation/bambi_external_development.json'
        if self.mode=='control' and external_path.exists():
            ext=json.loads(external_path.read_text(encoding='utf-8'));before=ext['models']['before']['summary'];after=ext['models']['after']['summary'];additional_findings=f"Дополнительная проверка BAMBI: {ext['images']} кадров из трёх полётов, найдено {before['matched']}/{before['reference_boxes']} прежде и {after['matched']}/{after['reference_boxes']} новыми весами. Разметка выборочная: непарные предсказания не считаются автоматически ложными. Это диагностика переноса, не итоговый тест."
        counts=Counter(categories[e['source']] for e in self.entries)
        return {'dataset':self.dataset_id,'dataset_title':self.title,'revision':self.index['revision'],'effective_split':self.split,'collection_note':note,'additional_findings':additional_findings,'prediction_status_note':prediction_note,'category_names':{key:('Косули' if key=='roe_deer' else 'Благородные олени') for key in counts},'category_counts':dict(counts),'source_urls':['https://zenodo.org/records/18206908'],'wild_downloaded':len(self.entries),'baseline':baseline,'analysis':analysis,'threshold_analysis':None,'total':len(selected),'downloaded':len(self.entries),'repository_total':len(self.entries),'page':page,'pages':pages,'items':rows,'related_group_options':[],'active_related_group':''}
