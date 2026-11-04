"""Read-only preview adapter for the already downloaded roe audit sample."""
import hashlib,json
from pathlib import Path


class RoePreview:
    def __init__(self,root,parse_labels,training=False):
        self.root=Path(root);self.training=training;self.parse_labels=parse_labels
        self.dataset_id='roe_train' if training else 'roe';self.split='train' if training else 'diagnostic'
        self.data=self.root/('data/roe/mixed_candidates' if training else 'data/roe/audit')
        self.url_prefix='/data/roe-training/images/' if training else '/data/roe-audit/images/'
        if training:
            prepared=json.loads((self.root/'data/mixed_pilot/index.json').read_text(encoding='utf-8'))
            summary={'sample':[{'name':e['name'],'sha256':e['image_sha256'],'label_sha256':e['label_sha256'],'visual_group':e['visual_group']} for e in prepared['roe_training']]}
        else:
            summary=json.loads((self.data/'summary.json').read_text(encoding='utf-8'))
        self.samples=summary['sample'];self.images={};self.labels={};self.allowed_images={}
        fingerprints=[]
        for item in self.samples:
            name=item['name']
            if Path(name).name!=name or '\\' in name or Path(name).suffix.lower() not in ('.jpg','.jpeg','.png'):
                raise ValueError('Unsafe roe sample filename')
            image=(self.data/'images'/name).resolve();label=(self.data/'labels'/Path(name).with_suffix('.txt').name).resolve()
            if not image.is_relative_to((self.data/'images').resolve()) or not label.is_relative_to((self.data/'labels').resolve()):
                raise ValueError('Roe sample path escapes its folder')
            if hashlib.sha256(image.read_bytes()).hexdigest()!=item['sha256'] or hashlib.sha256(label.read_bytes()).hexdigest()!=item['label_sha256']:
                raise ValueError('Roe sample file differs from frozen audit')
            self.images[name]=image;self.labels[name]=label
            self.allowed_images[self.url_prefix+name]=image
            fingerprints.append((name,item['sha256'],item['label_sha256']))
        if len(self.images)!=len(self.samples):raise ValueError('Duplicate roe sample filename')
        self.revision=hashlib.sha256(json.dumps(sorted(fingerprints)).encode()).hexdigest()

    def response(self,query):
        needle=query.get('q',[''])[0].lower();category=query.get('category',[''])[0];split=query.get('split',[self.split])[0]
        group_path=self.root/'data/roe/grouping/sample_groups.json'
        grouped={};group_options=[]
        if group_path.exists():
            groups=json.loads(group_path.read_text(encoding='utf-8'))
            expected={item['name']:item['sha256'] for item in self.samples}
            if groups.get('input_image_sha256')==expected:
                for number,names in enumerate(groups['nontrivial_groups'],1):
                    if any(name not in self.images for name in names):raise ValueError('Related-image group belongs to another sample')
                    entry={'id':'G'+str(number),'names':names,'label':'Похожие кадры: группа '+str(number)}
                    group_options.append(entry)
                    for name in names:grouped[name]=entry
        active_group=query.get('group',[''])[0]
        group_names=next((set(entry['names']) for entry in group_options if entry['id']==active_group),set())
        page=max(0,int(query.get('page',['0'])[0]))
        selected=[item for item in self.samples if (not active_group or item['name'] in group_names) and needle in item['name'].lower() and category in ('','europe','wild','roe_deer') and split in (('','train') if self.training else ('','diagnostic','valid'))]
        pages=max(1,(len(selected)+11)//12);page=min(page,pages-1)
        path=self.root/'data/roe/evaluation/latest.json'
        report=json.loads(path.read_text(encoding='utf-8')) if path.exists() and not self.training else None
        if report and (report['sample_revision']!=self.revision or set(report['items'])!=set(self.images)):
            raise ValueError('Roe prediction report belongs to another sample')
        rows=[]
        for item in selected[page*12:(page+1)*12]:
            name=item['name'];label=self.parse_labels(self.labels[name],['animal'])
            if hashlib.sha256(self.labels[name].read_bytes()).hexdigest()!=item['label_sha256']:
                raise ValueError('Roe labels changed after the audit')
            prediction=None
            if report:
                result=report['items'][name]
                if result['image_sha256']!=item['sha256'] or result['label_sha256']!=item['label_sha256']:
                    raise ValueError('Roe prediction checksum differs from audit')
                prediction={'boxes':result['predictions'],'counts':result['counts'],'timings':result['timings'],'original_size':result['original_size']}
            rows.append({'name':name,'split':self.split,'flight_id':None,'src':self.url_prefix+name,'prediction':prediction,'related_group':grouped.get(name),**label})
        baseline=analysis=None
        if report:
            q=report['quality'];t=report['timing_summary']['total_ms'];qview={'images':q['images'],'tp':q['matched'],'fp':q['false_boxes'],'fn':q['missed'],'precision':q['precision'],'recall':q['reference_recall'],'small_reference':q['small_reference'],'small_matched':q['small_found']}
            baseline={'model':'YOLO11n после обучения на 512 кадрах оленей · проверка косуль','conditions':report['conditions'],'timing_summary':report['timing_summary'],'validation_preview':qview}
            note='Те же веса, вход 1024 и порог 0,25. Результат посчитан программно по разметке косуль. Люди не входят в целевой класс animal.'
            if q['matched']+q['false_boxes']==0:note+=' Модель обработала все снимки и не выдала ни одной рамки; ноль ложных рамок здесь связан с отсутствием ответов.'
            control=report.get('positive_control')
            if control and control.get('passed'):note+=f" Контроль на прежних кадрах воспроизвёл {control['counts']['tp']} из 115 обнаружений."
            analysis={'experiment_kind':'transfer','dataset_revision':self.revision,'precision_available':True,'validation_images':len(self.samples),'validation_flights':[],'scope_note':f"{len(self.samples)} новых диагностических снимка, {q['reference_boxes']} рамки животных. Это образец другого источника, а не независимый итоговый тест по полётам.",'comparison_note':note,'experiments':[{'label':'Новые кадры косуль · прежние веса','reference_boxes':q['reference_boxes'],'matched':q['matched'],'unmatched_reference':q['missed'],'unmatched_predictions':q['false_boxes'],'small_fixed_reference_at_1024':q['small_reference'],'small_fixed_matched':q['small_found'],'median_total_ms':t['median'],'timing_images':len(self.samples)}]}
        return {'related_group_options':group_options,'active_related_group':active_group,'dataset':self.dataset_id,'dataset_title':f'Косули — отобранные учебные кадры ({len(self.samples)})' if self.training else 'Косули — новые 32 RGB-снимка','revision':self.revision,'effective_split':self.split,'category_names':{'roe_deer':'Косули'},'category_counts':{'roe_deer':len(self.samples)},'source_urls':['https://zenodo.org/records/18206908'],'wild_downloaded':len(self.samples),'baseline':baseline,'analysis':analysis,'threshold_analysis':None,'total':len(selected),'downloaded':len(self.samples),'repository_total':len(self.samples),'page':page,'pages':pages,'items':rows}
