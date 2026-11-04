"""Win32 helpers used by the BeastAI overlay itself (not external UI automation)."""
import ctypes
from ctypes import wintypes
import threading

user32 = ctypes.WinDLL('user32', use_last_error=True)
kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
user32.GetAncestor.argtypes = [wintypes.HWND, wintypes.UINT]
user32.GetAncestor.restype = wintypes.HWND
user32.SetWindowPos.argtypes = [wintypes.HWND,wintypes.HWND,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,wintypes.UINT]
user32.SetWindowPos.restype = wintypes.BOOL
user32.GetWindowRect.argtypes = [wintypes.HWND,ctypes.POINTER(wintypes.RECT)]
user32.GetClientRect.argtypes = [wintypes.HWND,ctypes.POINTER(wintypes.RECT)]
user32.SetWindowDisplayAffinity.argtypes = [wintypes.HWND,wintypes.DWORD]
user32.SetWindowDisplayAffinity.restype = wintypes.BOOL
user32.GetWindowDisplayAffinity.argtypes = [wintypes.HWND,ctypes.POINTER(wintypes.DWORD)]
user32.GetWindowDisplayAffinity.restype = wintypes.BOOL
_get_style = user32.GetWindowLongPtrW
_get_style.argtypes = [wintypes.HWND,ctypes.c_int]
_get_style.restype = ctypes.c_ssize_t
_set_style = user32.SetWindowLongPtrW
_set_style.argtypes = [wintypes.HWND,ctypes.c_int,ctypes.c_ssize_t]
_set_style.restype = ctypes.c_ssize_t
user32.RegisterHotKey.argtypes = [wintypes.HWND,ctypes.c_int,wintypes.UINT,wintypes.UINT]
user32.RegisterHotKey.restype = wintypes.BOOL
user32.UnregisterHotKey.argtypes = [wintypes.HWND,ctypes.c_int]
user32.PostThreadMessageW.argtypes = [wintypes.DWORD,wintypes.UINT,wintypes.WPARAM,wintypes.LPARAM]
user32.GetMessageW.argtypes = [ctypes.POINTER(wintypes.MSG),wintypes.HWND,wintypes.UINT,wintypes.UINT]
user32.GetMessageW.restype = ctypes.c_int
user32.PeekMessageW.argtypes = [ctypes.POINTER(wintypes.MSG),wintypes.HWND,wintypes.UINT,wintypes.UINT,wintypes.UINT]

def enable_physical_pixels():
    try:
        user32.SetProcessDpiAwarenessContext.argtypes = [ctypes.c_void_p]
        if user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4)):
            return 'Per-monitor v2'
    except AttributeError:
        pass
    # ERROR_ACCESS_DENIED means awareness may already have been set by the host.
    return 'Existing process DPI awareness'

def virtual_desktop():
    return tuple(user32.GetSystemMetrics(i) for i in (76,77,78,79))

def top_handle(widget):
    return user32.GetAncestor(widget.winfo_id(),2)

def place_window(widget,x,y,width,height,topmost=True):
    widget.geometry(f'{width}x{height}')
    widget.update_idletasks()
    hwnd=top_handle(widget);outer=wintypes.RECT();client=wintypes.RECT()
    user32.GetWindowRect(hwnd,ctypes.byref(outer));user32.GetClientRect(hwnd,ctypes.byref(client))
    outer_width=width+(outer.right-outer.left)-(client.right-client.left)
    outer_height=height+(outer.bottom-outer.top)-(client.bottom-client.top)
    if not user32.SetWindowPos(hwnd,wintypes.HWND(-1 if topmost else -2),x,y,outer_width,outer_height,0x0010|0x0040):
        raise ctypes.WinError(ctypes.get_last_error())

def configure_overlay(widget):
    hwnd=top_handle(widget)
    style=_get_style(hwnd,-20)
    required=0x80000|0x20|0x08000000|0x80  # layered, pass-through, no activation, tool window
    ctypes.set_last_error(0)
    _set_style(hwnd,-20,style|required)
    if ctypes.get_last_error():
        raise ctypes.WinError(ctypes.get_last_error())
    if not user32.SetWindowDisplayAffinity(hwnd,0x11):
        raise ctypes.WinError(ctypes.get_last_error())
    actual=wintypes.DWORD()
    assert user32.GetWindowDisplayAffinity(hwnd,ctypes.byref(actual))
    if actual.value!=0x11 or (_get_style(hwnd,-20)&required)!=required:
        raise RuntimeError('Windows did not enable capture exclusion/click-through')
    return {'hwnd':int(hwnd),'extended_style':int(_get_style(hwnd,-20)),'display_affinity':actual.value,'click_through':True}

class StopHotkey(threading.Thread):
    """F8 is registered only during detection, on its own message-loop thread."""
    def __init__(self,callback,error_callback):
        super().__init__(daemon=True)
        self.callback=callback;self.error_callback=error_callback
        self.cancel=threading.Event();self.thread_id=None
    def run(self):
        message=wintypes.MSG()
        user32.PeekMessageW(ctypes.byref(message),None,0,0,0)
        self.thread_id=kernel32.GetCurrentThreadId()
        if not user32.RegisterHotKey(None,1,0x4000,0x77):
            self.error_callback('F8 занята другой программой. Используй кнопку «Стоп».')
            return
        try:
            while not self.cancel.is_set():
                result=user32.GetMessageW(ctypes.byref(message),None,0,0)
                if result<=0:break
                if message.message==0x0312:self.callback()
        finally:
            user32.UnregisterHotKey(None,1)
    def stop(self):
        self.cancel.set()
        if self.thread_id:user32.PostThreadMessageW(self.thread_id,0x0012,0,0)

class RegionCapture:
    """Reusable top-down GDI bitmap. BitBlt reads only the selected screen rectangle."""
    def __init__(self,region):
        self.region=region;self.screen_dc=None;self.memory_dc=None;self.bitmap=None;self.previous=None
        self.gdi=ctypes.WinDLL('gdi32',use_last_error=True)
        user32.GetDC.argtypes=[wintypes.HWND];user32.GetDC.restype=wintypes.HDC
        user32.ReleaseDC.argtypes=[wintypes.HWND,wintypes.HDC]
        self.gdi.CreateCompatibleDC.argtypes=[wintypes.HDC];self.gdi.CreateCompatibleDC.restype=wintypes.HDC
        self.gdi.SelectObject.argtypes=[wintypes.HDC,wintypes.HANDLE];self.gdi.SelectObject.restype=wintypes.HANDLE
        self.gdi.DeleteObject.argtypes=[wintypes.HANDLE];self.gdi.DeleteDC.argtypes=[wintypes.HDC]
        self.gdi.BitBlt.argtypes=[wintypes.HDC,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,wintypes.HDC,ctypes.c_int,ctypes.c_int,wintypes.DWORD]
        self.gdi.BitBlt.restype=wintypes.BOOL
        class Header(ctypes.Structure):
            _fields_=[('size',wintypes.DWORD),('width',wintypes.LONG),('height',wintypes.LONG),('planes',wintypes.WORD),('bit_count',wintypes.WORD),('compression',wintypes.DWORD),('image_size',wintypes.DWORD),('xppm',wintypes.LONG),('yppm',wintypes.LONG),('used',wintypes.DWORD),('important',wintypes.DWORD)]
        class Info(ctypes.Structure):
            _fields_=[('header',Header),('colours',wintypes.DWORD*1)]
        self.gdi.CreateDIBSection.argtypes=[wintypes.HDC,ctypes.POINTER(Info),wintypes.UINT,ctypes.POINTER(ctypes.c_void_p),wintypes.HANDLE,wintypes.DWORD]
        self.gdi.CreateDIBSection.restype=wintypes.HBITMAP
        try:
            self.screen_dc=user32.GetDC(None)
            if not self.screen_dc:raise ctypes.WinError(ctypes.get_last_error())
            self.memory_dc=self.gdi.CreateCompatibleDC(self.screen_dc)
            if not self.memory_dc:raise ctypes.WinError(ctypes.get_last_error())
            info=Info();info.header.size=ctypes.sizeof(Header);info.header.width=region.width;info.header.height=-region.height;info.header.planes=1;info.header.bit_count=32
            self.bits=ctypes.c_void_p();self.bitmap=self.gdi.CreateDIBSection(self.screen_dc,ctypes.byref(info),0,ctypes.byref(self.bits),None,0)
            if not self.bitmap or not self.bits.value:raise ctypes.WinError(ctypes.get_last_error())
            self.previous=self.gdi.SelectObject(self.memory_dc,self.bitmap)
            if not self.previous:raise ctypes.WinError(ctypes.get_last_error())
        except Exception:
            self.close();raise
    def grab(self):
        from PIL import Image
        r=self.region
        if not self.gdi.BitBlt(self.memory_dc,0,0,r.width,r.height,self.screen_dc,r.left,r.top,0x00CC0020):
            raise ctypes.WinError(ctypes.get_last_error())
        # No CAPTUREBLT: layered overlay windows are excluded; display affinity is also enabled.
        self.gdi.GdiFlush()
        pixels=ctypes.string_at(self.bits,r.width*r.height*4)
        return Image.frombytes('RGB',(r.width,r.height),pixels,'raw','BGRX')
    def close(self):
        if self.previous and self.memory_dc:self.gdi.SelectObject(self.memory_dc,self.previous);self.previous=None
        if self.bitmap:self.gdi.DeleteObject(self.bitmap);self.bitmap=None
        if self.memory_dc:self.gdi.DeleteDC(self.memory_dc);self.memory_dc=None
        if self.screen_dc:user32.ReleaseDC(None,self.screen_dc);self.screen_dc=None
