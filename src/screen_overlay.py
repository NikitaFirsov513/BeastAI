"""Local screen-region animal detector with a transparent, click-through Windows overlay."""
from project_paths import PROJECT_ROOT
from overlay_process import DetectorWorker
from pathlib import Path
from dataclasses import dataclass
from collections import deque
import argparse,hashlib,json,logging,logging.handlers,os,queue,statistics,threading,time
from datetime import datetime,timezone

ROOT=PROJECT_ROOT
OUT=ROOT/'data/screen_overlay'
CACHE=ROOT/'data/dronemegadetector'
for key,value in [('HF_HOME',CACHE/'hf-cache'),('TORCH_HOME',CACHE/'torch-cache'),('MPLCONFIGDIR',CACHE/'matplotlib-cache'),('YOLO_CONFIG_DIR',ROOT/'data/ultralytics-config')]:os.environ[key]=str(value)
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',YOLO_AUTOINSTALL='false')
MODELS={'DroneMegaDetector · 1280':'reference','Наша YOLO11n · 1024':'baseline'}

@dataclass(frozen=True)
class Region:
    left:int
    top:int
    right:int
    bottom:int
    @property
    def width(self):return self.right-self.left
    @property
    def height(self):return self.bottom-self.top
    @property
    def bbox(self):return (self.left,self.top,self.right,self.bottom)

def selected_region(start,end,desktop):
    x,y,w,h=desktop
    left=max(x,min(start[0],end[0]));right=min(x+w,max(start[0],end[0]))
    top=max(y,min(start[1],end[1]));bottom=min(y+h,max(start[1],end[1]))
    if right-left<100 or bottom-top<80:return None
    return Region(left,top,right,bottom)

def screen_rect(box,width,height):
    x1=max(0.,min(width,box['x']*width));y1=max(0.,min(height,box['y']*height))
    x2=max(0.,min(width,(box['x']+box['width'])*width));y2=max(0.,min(height,(box['y']+box['height'])*height))
    return None if x2<=x1 or y2<=y1 else (x1,y1,x2,y2)

def put_latest(target,item):
    try:target.put_nowait(item)
    except queue.Full:
        try:target.get_nowait()
        except queue.Empty:pass
        try:target.put_nowait(item)
        except queue.Full:pass

class DetectorTask:
    def __init__(self,region,model_key,threshold,fps,frames,events,generation,cancel):
        self.region=region;self.key=model_key;self.threshold=threshold;self.fps=fps
        self.frames=frames;self.events=events;self.generation=generation;self.cancel=cancel
        self.samples=deque(maxlen=600);self.frame_count=0;self.model=None;self.conditions={};self.failed=False
    def event(self,kind,value):self.events.put((self.generation,kind,value))
    def run(self):
        torch=None;grabber=None
        try:
            self.event('status','Загрузка модели… Первый запуск требует времени.')
            import torch,cv2,numpy as np
            from PIL import Image,ImageGrab
            from tiled_inference import map_box,merge_boxes
            torch.set_num_threads(4);cv2.setNumThreads(4)
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
            if not torch.cuda.is_available():raise RuntimeError('CUDA недоступна. Проверь запуск через start_screen_overlay.cmd.')
            plan=json.loads((CACHE/'experiment_plan.json').read_text(encoding='utf-8'));cfg=plan['models'][self.key]
            weights=Path(cfg['path'])
            if hashlib.sha256(weights.read_bytes()).hexdigest()!=cfg['sha256']:raise RuntimeError('Контрольная сумма весов изменилась.')
            if self.cancel.is_set():return
            if self.key=='reference':
                import rfdetr
                self.model=rfdetr.from_checkpoint(str(weights),device='cuda',amp=False)
                assert self.model.class_names==['person','animal','vehicle']
                self.model.model.model.float().eval()
                def predict(image):
                    detections=self.model.predict(image,threshold=self.threshold,include_source_image=False)
                    w,h=image.size;boxes=[]
                    for coords,score,cls in zip(detections.xyxy,detections.confidence,detections.class_id):
                        if int(cls)!=1:continue
                        b=map_box([*map(float,coords),float(score),0],(0,0,w,h),(w,h),{0:'animal'})
                        if b:boxes.append(b)
                    return merge_boxes(boxes,.5,300)
            else:
                from ultralytics import YOLO,settings
                settings.update({'sync':False});self.model=YOLO(str(weights))
                def predict(image):
                    rgb=np.asarray(image);bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
                    r=self.model.predict(source=bgr,imgsz=1024,rect=False,device=0,half=False,batch=1,conf=self.threshold,iou=.7,agnostic_nms=True,classes=[0],max_det=300,verbose=False,save=False)[0]
                    w,h=image.size
                    boxes=[map_box(row,(0,0,w,h),(w,h),{0:'animal'}) for row in r.boxes.data.detach().cpu().tolist()]
                    return merge_boxes([b for b in boxes if b],.5,300)
            # Warm up on synthetic pixels; no screen capture until the user presses Start.
            with torch.inference_mode(),torch.autocast('cuda',enabled=False):
                for _ in range(2):
                    if self.cancel.is_set():return
                    predict(Image.new('RGB',(640,480)))
            self.conditions={'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,'input':cfg['input'],'precision':'FP32','batch':1,'weights_sha256':cfg['sha256']}
            self.event('ready',self.conditions)
            from screen_overlay_win32 import RegionCapture
            if self.cancel.is_set():return
            grabber=RegionCapture(self.region)
            while not self.cancel.is_set():
                torch.cuda.synchronize();started=time.perf_counter()
                image=grabber.grab()
                captured=time.perf_counter()
                if image.size!=(self.region.width,self.region.height):raise RuntimeError('Размер захвата изменился. Выдели область заново.')
                with torch.inference_mode(),torch.autocast('cuda',enabled=False):boxes=predict(image)
                torch.cuda.synchronize();finished=time.perf_counter()
                row={'generation':self.generation,'capture_started':started,'finished':finished,'capture_ms':(captured-started)*1000,'detector_and_boxes_ms':(finished-captured)*1000,'capture_to_result_ms':(finished-started)*1000,'boxes':boxes}
                if not self.cancel.is_set():put_latest(self.frames,row)
                self.frame_count+=1
                self.samples.append({k:row[k] for k in ('capture_ms','detector_and_boxes_ms','capture_to_result_ms')})
                self.cancel.wait(max(0.,1/self.fps-(time.perf_counter()-started)))
        except Exception as error:
            self.failed=True;logging.exception('Detector failed');self.event('error',str(error))
        finally:
            try:
                if grabber is not None:grabber.close()
                if torch is not None:
                    self.model=None
                    import gc
                    gc.collect()
                    if torch.cuda.is_available():torch.cuda.empty_cache()
                if self.samples:
                    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
                    payload={'model':self.key,'region':self.region.bbox,'input':1280 if self.key=='reference' else 1024,'precision':'FP32; TF32 off','batch':1,'fps_limit':self.fps,'capture_backend':'Cached GDI DIB, selected ROI only','confidence':self.threshold,'frames_processed':self.frame_count,'conditions':self.conditions,'timing_window':'Last600 processed frames at most','timings':list(self.samples),'images_saved':False,'scope':'Live screen capture plus detector/box processing; model load and warmup excluded. UI handoff measured separately; no quality score or physical-display latency claim.'}
                    (OUT/f'session_{stamp}_{os.getpid()}_{self.generation}.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
            except Exception:
                logging.exception('Detector cleanup failed')
            finally:
                self.event('stopped',None)

class Overlay:
    def __init__(self,root,region):
        import tkinter as tk
        from screen_overlay_win32 import place_window,configure_overlay
        self.region=region;self.window=tk.Toplevel(root);self.window.withdraw()
        self.window.overrideredirect(True);self.window.attributes('-topmost',True)
        self.background='#010101';self.window.configure(bg=self.background)
        self.window.attributes('-transparentcolor',self.background)
        self.canvas=tk.Canvas(self.window,bg=self.background,highlightthickness=0,bd=0)
        self.canvas.pack(fill='both',expand=True)
        self.window.deiconify();place_window(self.window,region.left,region.top,region.width,region.height)
        try:self.native=configure_overlay(self.window)
        except Exception:self.window.destroy();raise
    def draw(self,boxes):
        self.clear();w,h=self.region.width,self.region.height
        for box in boxes:
            rect=screen_rect(box,w,h)
            if rect is None:continue
            x1,y1,x2,y2=rect
            self.canvas.create_rectangle(x1,y1,x2,y2,outline='#70ff9d',width=3,tags='detections')
            label=f"animal {box['score']:.2f}"
            text=self.canvas.create_text(max(3,x1+4),max(12,y1-10),anchor='w',text=label,fill='#b9ffce',font=('Segoe UI',11,'bold'),tags='detections')
            bounds=self.canvas.bbox(text)
            if bounds:
                background=self.canvas.create_rectangle(bounds[0]-3,bounds[1]-1,bounds[2]+3,bounds[3]+1,fill='#133326',outline='',tags='detections')
                self.canvas.tag_lower(background,text)
    def clear(self):self.canvas.delete('detections')
    def destroy(self):self.window.destroy()

class App:
    def __init__(self,root,demo=False):
        import tkinter as tk
        from tkinter import ttk
        self.tk=tk;self.root=root;self.worker=None;self.overlay=None;self.hotkey=None;self.region=None;self.failure=None
        self.frames=queue.Queue(maxsize=1);self.events=queue.Queue();self.generation=0;self.closing=False
        self.last_capture=None;self.render_times=deque(maxlen=120);self.arrivals=deque(maxlen=60);self.selector=None;self.fixture=None
        root.title('BeastAI — рамки поверх видео');root.geometry('560x540');root.resizable(False,False)
        root.configure(bg='#12201b');root.attributes('-topmost',True)
        style=ttk.Style(root);style.theme_use('clam')
        style.configure('TFrame',background='#12201b');style.configure('TLabel',background='#12201b',foreground='#e0f0e7',font=('Segoe UI',10))
        style.configure('TButton',font=('Segoe UI',10),padding=(12,9));style.configure('Accent.TButton',background='#377e58',foreground='white')
        panel=ttk.Frame(root,padding=20);panel.pack(fill='both',expand=True)
        ttk.Label(panel,text='BEASTAI  /  ЭКРАН',font=('Segoe UI',17,'bold')).pack(anchor='w')
        ttk.Label(panel,text='Открой видео, выдели его область и нажми «Старт».\nРамки пропускают клики. F8 — остановить захват.',wraplength=480).pack(anchor='w',pady=(9,14))
        self.region_text=tk.StringVar(value='Область ещё не выбрана')
        ttk.Label(panel,textvariable=self.region_text,wraplength=480).pack(anchor='w')
        self.choose_button=ttk.Button(panel,text='1. Выбрать область видео',command=self.choose_region);self.choose_button.pack(fill='x',pady=(7,13))
        self.model_name=tk.StringVar(value=next(iter(MODELS)))
        self.model_picker=ttk.Combobox(panel,textvariable=self.model_name,values=list(MODELS),state='readonly');self.model_picker.pack(fill='x')
        options=ttk.Frame(panel);options.pack(fill='x',pady=12)
        ttk.Label(options,text='Порог рамки:').pack(side='left')
        self.threshold=tk.DoubleVar(value=.25);self.threshold_widget=ttk.Spinbox(options,from_=.05,to=.95,increment=.05,textvariable=self.threshold,width=5);self.threshold_widget.pack(side='left',padx=(6,18))
        ttk.Label(options,text='Не чаще кадров/с:').pack(side='left')
        self.fps=tk.IntVar(value=3);self.fps_widget=ttk.Spinbox(options,from_=1,to=10,textvariable=self.fps,width=3);self.fps_widget.pack(side='left',padx=6)
        buttons=ttk.Frame(panel);buttons.pack(fill='x')
        self.start_button=ttk.Button(buttons,text='2. Старт',command=self.start,style='Accent.TButton',state='disabled');self.start_button.pack(side='left',fill='x',expand=True,padx=(0,8))
        self.stop_button=ttk.Button(buttons,text='Стоп / F8',command=self.stop,state='disabled');self.stop_button.pack(side='left',fill='x',expand=True)
        ttk.Button(buttons,text='Выход',command=self.close).pack(side='left',padx=(8,0))
        self.status=self.tk.StringVar(value='Остановлено. Кадры не захватываются.')
        ttk.Label(panel,textvariable=self.status,wraplength=480).pack(anchor='w',pady=(15,6))
        self.metrics=tk.StringVar(value='Кадры не сохраняются и не отправляются в сеть.')
        ttk.Label(panel,textvariable=self.metrics,wraplength=480,font=('Segoe UI',9)).pack(anchor='w')
        ttk.Label(panel,text='Если передвинул окно видео — выдели область снова.\nЭто детектор без трекинга: на движении рамки могут отставать.',wraplength=480,font=('Segoe UI',9)).pack(anchor='w',pady=(8,0))
        root.protocol('WM_DELETE_WINDOW',self.close);root.bind('<Escape>',lambda e:self.stop())
        root.update_idletasks();self.panel_size=(max(560,root.winfo_reqwidth()),max(540,root.winfo_reqheight()))
        root.geometry(f'{self.panel_size[0]}x{self.panel_size[1]}')
        root.after(40,self.poll)
        if demo:root.after(200,self.open_demo)
    def set_region(self,region):
        self.region=region
        self.region_text.set('Область ещё не выбрана' if region is None else f'{region.width} × {region.height} пикселей · положение {region.left}, {region.top}')
        self.start_button.configure(state='normal' if region and not self.worker else 'disabled')
    def choose_region(self):
        if self.worker or self.closing:return
        from screen_overlay_win32 import virtual_desktop,place_window
        desktop=virtual_desktop();x,y,w,h=desktop
        self.root.withdraw();selection=self.tk.Toplevel(self.root);self.selector=selection
        selection.overrideredirect(True);selection.attributes('-topmost',True);selection.attributes('-alpha',.35)
        canvas=self.tk.Canvas(selection,bg='#102b43',cursor='crosshair',highlightthickness=0);canvas.pack(fill='both',expand=True)
        canvas.create_text(35,35,anchor='nw',fill='white',text='Обведи видео мышью. Esc — отмена.',font=('Segoe UI',22,'bold'))
        place_window(selection,x,y,w,h);selection.focus_force();points={}
        def press(event):points['start']=(event.x_root,event.y_root)
        def drag(event):
            if 'start' not in points:return
            canvas.delete('choice');sx,sy=points['start'];canvas.create_rectangle(sx-x,sy-y,event.x_root-x,event.y_root-y,outline='white',width=4,tags='choice')
        def finish(event=None):
            chosen=selected_region(points['start'],(event.x_root,event.y_root),desktop) if event is not None and 'start' in points else None
            selection.destroy();self.selector=None;self.root.deiconify()
            if chosen:self.set_region(chosen);self.status.set('Область выбрана. Нажми «Старт».')
        canvas.bind('<ButtonPress-1>',press);canvas.bind('<B1-Motion>',drag);canvas.bind('<ButtonRelease-1>',finish);selection.bind('<Escape>',lambda event:finish())
    def start(self):
        from tkinter import messagebox
        from screen_overlay_win32 import StopHotkey
        if self.worker or self.region is None or self.closing:return
        try:
            threshold=float(self.threshold_widget.get().replace(',','.'));fps=int(self.fps_widget.get())
            if not .05<=threshold<=.95 or not 1<=fps<=10:raise ValueError('Порог: 0,05–0,95; частота: 1–10 кадров/с.')
            self.overlay=Overlay(self.root,self.region)
        except Exception as error:
            if self.overlay:self.overlay.destroy();self.overlay=None
            logging.exception('Overlay start failed');messagebox.showerror('Не удалось запустить',str(error));return
        self.failure=None;self.generation+=1;generation=self.generation;self.render_times.clear();self.arrivals.clear();self.last_capture=None
        try:
            self.worker=DetectorWorker(self.region,MODELS[self.model_name.get()],threshold,fps,self.frames,self.events,generation)
            self.worker.start()
            self.hotkey=StopHotkey(lambda:self.events.put((generation,'hotkey',None)),lambda text:self.events.put((generation,'warning',text)))
            self.hotkey.start()
        except Exception as error:
            logging.exception('Detector launch failed');self.failure='Ошибка запуска: '+str(error)
            self.stop('launch_error');self.status.set(self.failure);return
        self.start_button.configure(state='disabled');self.stop_button.configure(state='normal');self.choose_button.configure(state='disabled')
        for widget in (self.model_picker,self.threshold_widget,self.fps_widget):widget.configure(state='disabled')
        self.status.set('Запуск… F8 или «Стоп» — остановить.');self.metrics.set('Модель загружается. Обработка локальная.')
        logging.info('Started model=%s region=%s threshold=%s fps_limit=%s affinity=%s',MODELS[self.model_name.get()],self.region,threshold,fps,self.overlay.native)
    def stop(self,reason='user_stop'):
        if self.worker:self.worker.request_stop(reason);self.status.set('Останавливаем…')
        if self.hotkey:
            try:self.hotkey.stop()
            except Exception:logging.exception('Hotkey cleanup failed')
            self.hotkey=None
        if self.overlay:
            try:self.overlay.destroy()
            except self.tk.TclError:pass
            self.overlay=None
        self.stop_button.configure(state='disabled');self.last_capture=None
    def finish_worker(self):
        worker=self.worker
        if worker is None or worker.is_alive():return
        if worker.exitcode not in (0,None) and worker.stop_reason is None and not self.failure:
            self.failure='Процесс детектора завершился с ошибкой. См. журнал worker в data/screen_overlay.'
        self.stop('worker_exit');self.worker=None
        try:
            worker.close()
            payload={'pid':worker.pid,'exit_code':worker.exitcode,'reason':worker.stop_reason,'forced':worker.forced,'closing':self.closing,'generation':self.generation,'grace_seconds':worker.grace_seconds}
            (OUT/'last_stop.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
            logging.info('Detector process finished: %s',payload)
            if self.render_times:
                values=sorted(self.render_times);payload={'generation':self.generation,'model':MODELS[self.model_name.get()],'region':self.region.bbox if self.region else None,'sample_count':len(values),'median_capture_to_draw_submit_ms':statistics.median(values),'p95_capture_to_draw_submit_ms':values[min(len(values)-1,int(.95*(len(values)-1)))],'scope':'Last120 visible results at most. Includes process-to-UI handoff; ends at drawing submission, not physical display.'}
                (OUT/'last_ui_handoff.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
        except Exception:logging.exception('Could not save shutdown measurements')
        self.status.set(self.failure or 'Остановлено. Кадры не захватываются.')
        if self.render_times:self.metrics.set(f'Последняя серия: медиана до отрисовки {statistics.median(self.render_times):.0f} мс. Новые кадры не захватываются.')
        self.choose_button.configure(state='normal');self.model_picker.configure(state='readonly')
        for widget in (self.threshold_widget,self.fps_widget):widget.configure(state='normal')
        self.set_region(self.region)
    def poll(self):
        while True:
            try:generation,kind,value=self.events.get_nowait()
            except queue.Empty:break
            if generation!=self.generation:continue
            if kind=='hotkey':self.stop()
            elif kind in ('status','warning'):
                if not self.closing and self.worker and not self.worker.cancel.is_set():self.status.set(value)
            elif kind=='ready':
                if not self.closing and self.worker and not self.worker.cancel.is_set():self.status.set('Работает · '+value['gpu']+' · FP32 · F8 — стоп')
            elif kind=='error':self.failure='Ошибка: '+value;self.stop('error');self.status.set(self.failure)
            elif kind=='stopped':self.stop('completed')
        if self.worker:
            self.worker.poll_shutdown()
            if not self.worker.is_alive():self.finish_worker()
        if self.closing:
            if self.worker is None:self.root.destroy();return
            self.root.after(40,self.poll);return
        try:result=self.frames.get_nowait()
        except queue.Empty:result=None
        now=time.perf_counter()
        if result and result['generation']==self.generation and self.overlay and self.worker and not self.worker.cancel.is_set():
            age=now-result['capture_started']
            if age<=1.0:
                self.overlay.draw(result['boxes']);self.last_capture=result['capture_started']
                total=(time.perf_counter()-result['capture_started'])*1000;self.render_times.append(total);self.arrivals.append(now)
                rate=(len(self.arrivals)-1)/(self.arrivals[-1]-self.arrivals[0]) if len(self.arrivals)>1 else 0.
                self.metrics.set(f"Рамок: {len(result['boxes'])} · обработано {rate:.1f} кадра/с\nЗахват → команда отрисовки: {total:.0f} мс · медиана {statistics.median(self.render_times):.0f} мс")
            else:self.overlay.clear();self.metrics.set('Результат старше 1 с скрыт. Снизь нагрузку или выбери нашу YOLO11n.')
        if self.overlay and self.last_capture is not None and now-self.last_capture>1.0:self.overlay.clear()
        self.root.after(40,self.poll)
    def close(self):
        if self.closing:return
        self.closing=True;logging.info('Overlay window close requested')
        if self.selector:
            try:self.selector.destroy()
            except self.tk.TclError:pass
            self.selector=None
        self.stop('window_close')
        if self.fixture:
            try:self.fixture.destroy()
            except self.tk.TclError:pass
            self.fixture=None
        self.root.withdraw()
        if self.worker is None:self.root.destroy()
    def open_demo(self):
        from PIL import Image,ImageTk
        from screen_overlay_win32 import place_window,virtual_desktop
        report=json.loads((CACHE/'evaluation/comparison.json').read_text(encoding='utf-8'))
        candidates=[r for r in report['models']['baseline']['items'].values() if r['source']=='sherbrooke' and r['counts']['tp']>=5]
        e=max(candidates,key=lambda r:r['counts']['tp'])
        window=self.tk.Toplevel(self.root);self.fixture=window;window.title('BeastAI — тестовый снимок, не видео')
        image=Image.open(e['image_path']).convert('RGB');x,y,desktop_w,desktop_h=virtual_desktop()
        scale=min(1.,(desktop_w-80)/image.width,(desktop_h-120)/image.height)
        image=image.resize((int(image.width*scale),int(image.height*scale)))
        canvas=self.tk.Canvas(window,width=image.width,height=image.height,highlightthickness=0,bd=0);canvas.pack()
        photo=ImageTk.PhotoImage(image);canvas.create_image(0,0,anchor='nw',image=photo);canvas.image=photo
        clicks=self.tk.StringVar(value='Щелчков по снимку: 0');counter=[0]
        def clicked(event):counter[0]+=1;clicks.set(f'Щелчков по снимку: {counter[0]}')
        canvas.bind('<Button-1>',clicked)
        self.tk.Label(window,textvariable=clicks).pack();self.tk.Label(window,text='Это проверка окна и захвата. Оценки качества по ней нет.').pack()
        window.update_idletasks();place_window(window,x+20,y+30,image.width,image.height+70,False)
        def closed():window.destroy();self.fixture=None;self.stop();self.set_region(None)
        window.protocol('WM_DELETE_WINDOW',closed)
        self.root.update_idletasks()
        if desktop_w>image.width+self.panel_size[0]+70:place_window(self.root,x+image.width+40,y+40,*self.panel_size)
        else:place_window(self.root,x+desktop_w-self.panel_size[0]-20,y+40,*self.panel_size)
        self.root.after(150,lambda:self.set_region(Region(canvas.winfo_rootx(),canvas.winfo_rooty(),canvas.winfo_rootx()+image.width,canvas.winfo_rooty()+image.height)))
        self.status.set('Выбран тестовый снимок. «Старт» проверит рамки поверх него.')
        self.root.after(500,self.demo_capture_check)

    def demo_capture_check(self):
        # Integration check on our own static fixture only; never save screen pixels.
        if self.region is None or not self.fixture or self.worker:return
        from PIL import ImageGrab,ImageChops
        from screen_overlay_win32 import RegionCapture
        native_capture=RegionCapture(self.region)
        native_before=native_capture.grab()
        before=ImageGrab.grab(bbox=self.region.bbox,all_screens=True,include_layered_windows=False).convert('RGB')
        probe=Overlay(self.root,self.region)
        probe.canvas.create_rectangle(20,20,300,130,fill='#ff00ff',outline='#ff00ff')
        def finish():
            try:
                after=native_capture.grab()
                roi_matches_reference=ImageChops.difference(before,native_before).getbbox() is None
                equal=ImageChops.difference(native_before,after).getbbox() is None
                result={'passed':equal and roi_matches_reference,'roi_capture_matches_pillow_reference':roi_matches_reference,'capture_backend':'Cached GDI DIB, selected ROI only','own_static_fixture_only':True,'capture_unchanged_with_visible_test_overlay':equal,'windows_flags':probe.native,'size':before.size,'no_pixels_saved':True}
                (OUT/'demo_capture_check.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
                logging.info('Own-fixture capture exclusion: %s',result)
                if not equal:self.status.set('Самопроверка захвата не прошла. См. overlay.log.')
            finally:probe.destroy();native_capture.close()
        self.root.after(250,finish)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--demo',action='store_true');args=parser.parse_args()
    if os.name!='nt':raise RuntimeError('This overlay requires Windows')
    OUT.mkdir(parents=True,exist_ok=True)
    handler=logging.handlers.RotatingFileHandler(OUT/'overlay.log',maxBytes=1_000_000,backupCount=2,encoding='utf-8')
    logging.basicConfig(level=logging.INFO,handlers=[handler],format='%(asctime)s %(levelname)s %(message)s')
    import ctypes
    from ctypes import wintypes
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.CreateMutexW.argtypes=[ctypes.c_void_p,wintypes.BOOL,wintypes.LPCWSTR];kernel.CreateMutexW.restype=wintypes.HANDLE
    kernel.CloseHandle.argtypes=[wintypes.HANDLE]
    mutex=kernel.CreateMutexW(None,False,'Local\\BeastAIOverlay_'+hashlib.sha256(str(ROOT).encode()).hexdigest()[:12])
    if not mutex:raise ctypes.WinError(ctypes.get_last_error())
    if ctypes.get_last_error()==183:
        ctypes.windll.user32.MessageBoxW(None,'BeastAI уже запущен. Используй открытое окно или значок на панели задач.','BeastAI',0x40)
        kernel.CloseHandle(mutex);return
    from screen_overlay_win32 import enable_physical_pixels
    logging.info('Starting overlay; DPI=%s',enable_physical_pixels())
    import tkinter as tk
    app=None
    try:
        root=tk.Tk();app=App(root,demo=args.demo);root.mainloop()
    finally:
        if app and app.worker:
            try:app.worker.force_shutdown()
            except Exception:logging.exception('Final detector shutdown failed')
        kernel.CloseHandle(mutex)
if __name__=='__main__':main()
