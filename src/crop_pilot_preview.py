"""Read-only automatic crop-pilot comparisons on frozen full-frame controls."""
from pathlib import Path
from collections import Counter
import hashlib,json
VARIANTS={'before_full':'Прежние веса · целый кадр','before_tiles':'Прежние веса · фрагменты','after_full':'После обучения · целый кадр','after_tiles':'После обучения · фрагменты'}
class CropPilotPreview:
    def __init__(self,root,parse_labels,split):
        if split not in ('control','fresh'):raise ValueError('Unknown crop-pilot split')
        self.root=Path(root);self.folder=self.root/'data/crop_pilot';self.parse_labels=parse_labels;self.split=split
        self.publication=json.loads((self.folder/'evaluation/published.json').read_text(encoding='utf-8'))
        self.index=json.loads((self.folder/'index.json').read_text(encoding='utf-8'));assert self.publication['dataset_revision']==self.index['revision']
        self.reports={}
        for stage in ('before','after'):
            p=self.folder/'evaluation'/(stage+'.json');content=p.read_bytes()
            assert hashlib.sha256(content).hexdigest()==self.publication['reports'][stage]['sha256']
            report=json.loads(content);assert report['dataset_revision']==self.index['revision'];self.reports[stage]=report
        self.entries=sorted(self.index['validation' if split=='control' else 'fresh_validation'],key=lambda e:(e['source']!='roe',e['name']))
        self.dataset_id='crop_pilot' if split=='control' else 'crop_pilot_fresh'
        self.title='Фрагменты · прежний контроль' if split=='control' else 'Фрагменты · новые отложенные кадры'
        self.ui_split='valid' if split=='control' else 'diagnostic';self.allowed_images={}
        for e in self.entries:
            assert Path(e['name']).name==e['name']
            for field,digest in [('image_path','image_sha256'),('label_path','label_sha256')]:
                p=Path(e[field]).resolve()
                if not p.is_relative_to(self.root) or hashlib.sha256(p.read_bytes()).hexdigest()!=e[digest]:raise ValueError('Crop control input changed')
            for report in self.reports.values():
                for mode in ('full','tiles'):
                    row=report['modes'][mode]['items'][e['source']+'/'+e['name']]
                    assert row['image_sha256']==e['image_sha256'] and row['label_sha256']==e['label_sha256']
            self.allowed_images[self.url(e)]=Path(e['image_path'])
    def url(self,e):return '/data/crop-pilot/'+e['source']+'/'+e['name']
    def response(self,query):
        variant=query.get('variant',['after_tiles'])[0]
        if variant not in VARIANTS:raise ValueError('Unknown crop-pilot variant')
        stage,mode=variant.split('_');active=self.reports[stage];category=query.get('category',[''])[0];needle=query.get('q',[''])[0].lower();split=query.get('split',[self.ui_split])[0]
        categories={'roe':'roe_deer','sherbrooke':'red_deer'}
        selected=[e for e in self.entries if split in ('',self.ui_split) and needle in e['name'].lower() and (category in ('','europe','wild') or categories[e['source']]==category)]
        pages=max(1,(len(selected)+11)//12);page=min(pages-1,max(0,int(query.get('page',['0'])[0])));rows=[]
        for e in selected[12*page:12*(page+1)]:
            lp=Path(e['label_path']);assert hashlib.sha256(lp.read_bytes()).hexdigest()==e['label_sha256']
            label=self.parse_labels(lp,['animal']);r=active['modes'][mode]['items'][e['source']+'/'+e['name']];assert r['reference']==label['boxes']
            rows.append({'name':e['name'],'source':e['source'],'split':self.ui_split,'flight_id':None,'src':self.url(e),'prediction':{k:r[k] for k in ('boxes','counts','timings','original_size')},**label})
        experiments=[];latency=[]
        for source,label in [('roe','Косули'),('sherbrooke','Олени')]:
            if category not in ('','europe','wild',categories[source]):continue
            for key,title in VARIANTS.items():
                st,md=key.split('_');report=self.reports[st]['modes'][md];r=report['partitions'][self.split].get(source)
                if not r:continue
                q=r['quality'];t=r['timing_summary']['total_ms'];experiments.append({'label':label+' · '+title,'reference_boxes':q['reference_boxes'],'matched':q['tp'],'unmatched_reference':q['fn'],'unmatched_predictions':q['fp'],'small_fixed_reference_at_1024':q['small_reference'],'small_fixed_matched':q['small_matched'],'median_total_ms':t['median'],'p95_total_ms':t['p95'],'timing_images':q['images']})
            r=active['modes'][mode]['partitions'][self.split].get(source)
            if r:
                t=r['timing_summary']['total_ms'];latency.append(f"{label}: медиана {t['median']:.1f} мс, p95 {t['p95']:.1f} мс")
        counts=Counter(categories[e['source']] for e in self.entries)
        scope='Все эти кадры исключены из данного обучения. Это повторно используемые данные разработки; независимость по полётам для косуль не подтверждена.'
        comparison='Одинаковый порог 0,25, совпадение IoU ≥ 0,5, вход 1024 × 1024, пакет 1, FP32. Фрагменты 512 × 512 покрывают весь кадр по постоянной сетке: 9 запусков на кадр косуль, 6 на кадр оленей. Малые животные — неизменная группа при масштабе целого кадра 1024. Время включает чтение, подготовку, все запуски и объединение рамок. p95 — время, быстрее которого прошли 95% измерений. Энергия не измерена.'
        mem=active['modes'][mode]['memory'];additional='; '.join(latency)+f". Память GPU, выделенная PyTorch: {mem['gpu_peak_allocated_bytes']/2**20:.1f} МиБ; модель {active['model_file_bytes']/2**20:.2f} МиБ. Это отдельный опыт, основная модель сохранена."
        fit_file=self.folder/'evaluation/training_fit.json'
        if fit_file.exists():
            fit=json.loads(fit_file.read_text(encoding='utf-8'));assert fit['dataset_revision']==self.index['revision']
            for key,report in self.reports.items():assert fit['models'][key]['model_sha256']==report['model_sha256']
            a=fit['models']['after'];q=a['counts'];tiny=a['size_groups']['below16'];additional+=f" Учебные кадры, поиск по сетке после обучения: {q['tp']}/89, лишних рамок {q['fp']}; самые маленькие {tiny['found']}/{tiny['reference']}. Это усвоение знакомых кадров, не качество на новых данных."
        additional+=' Текущий рецепт фрагментов не принят: на новых малых целях улучшения нет, лишних рамок много. Это не доказывает бесполезность любых фрагментов.'
        return {'dataset':self.dataset_id,'dataset_title':self.title,'revision':self.index['revision'],'effective_split':self.ui_split,'variant':variant,'variant_options':[{'id':k,'label':v} for k,v in VARIANTS.items()],'category_names':{k:('Косули' if k=='roe_deer' else 'Благородные олени') for k in counts},'category_counts':dict(counts),'source_urls':['https://zenodo.org/records/18206908'],'collection_note':scope,'additional_findings':additional,'prediction_status_note':'Синие рамки: '+VARIANTS[variant]+'. Поиск выполнен на целом кадре без подсказок из разметки.','baseline':None,'analysis':{'experiment_kind':'crop_pilot','precision_available':True,'validation_images':len(self.entries),'scope_note':scope,'comparison_note':comparison,'experiments':experiments},'threshold_analysis':None,'wild_downloaded':len(self.entries),'downloaded':len(self.entries),'repository_total':len(self.entries),'total':len(selected),'page':page,'pages':pages,'items':rows,'related_group_options':[],'active_related_group':''}
