"""Read individual ZIP members over verified HTTP byte ranges, never a whole archive."""
from collections import OrderedDict
import io
import time
import urllib.error
import urllib.request


class RemoteZipReader(io.RawIOBase):
    def __init__(self, url, size, budget=128*1024**2):
        super().__init__()
        self.url, self.size, self.budget = url, int(size), budget
        self.position = 0
        self.bytes_received = 0
        self.requests = 0
        self.cache = OrderedDict()

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        if whence not in (0, 1, 2):
            raise ValueError("Invalid seek mode")
        position = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        if position < 0:
            raise ValueError('Negative seek')
        self.position = position
        return position

    def read(self, size=-1):
        wanted = self.size-self.position if size < 0 else min(size, self.size-self.position)
        if wanted <= 0:
            return b''
        if wanted > 16*1024**2:
            raise ValueError('Refusing a large read: extract individual small files only')
        for start, content in list(self.cache.items()):
            if start <= self.position and self.position + wanted <= start+len(content):
                offset = self.position-start
                self.position += wanted
                self.cache.move_to_end(start)
                return content[offset:offset+wanted]
        start = self.position
        end = min(self.size-1,start+max(wanted,64*1024)-1)
        count = end-start+1
        if self.bytes_received+count > self.budget:
            raise ValueError('Partial-download budget exceeded')
        expected = f'bytes {start}-{end}/{self.size}'
        for attempt in range(3):
            try:
                req = urllib.request.Request(self.url,headers={'User-Agent':'BeastAI-educational-preview','Accept-Encoding':'identity','Range':f'bytes={start}-{end}'})
                with urllib.request.urlopen(req,timeout=35) as response:
                    if response.status != 206 or response.headers.get('Content-Range') != expected:
                        raise ValueError('Server did not honour the exact byte range; response body was not read')
                    content = response.read(count+1)
                if len(content) != count:
                    raise ValueError('Incomplete range response')
                break
            except (urllib.error.URLError, TimeoutError, OSError):
                if attempt==2:
                    raise
                time.sleep(attempt+1)
        self.bytes_received += len(content)
        self.requests += 1
        self.cache[start] = content
        while len(self.cache)>8:
            self.cache.popitem(last=False)
        self.position += wanted
        return content[:wanted]