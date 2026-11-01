import io
import re
import unittest
from unittest.mock import patch
import zipfile
from remote_zip import RemoteZipReader


class Response:
    def __init__(self,content,start,end,status=206):
        self.status=status
        self.headers={'Content-Range':f'bytes {start}-{end}/{len(content)}'}
        self.content=content[start:end+1]
        self.was_read=False
    def __enter__(self): return self
    def __exit__(self,*args): return False
    def read(self,size):
        self.was_read=True
        return self.content[:size]


class RemoteZipTests(unittest.TestCase):
    def test_zip_members_seek_and_crc_are_read_through_ranges(self):
        stream=io.BytesIO()
        with zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr('rgb/000010.jpg',b'example-content'*100)
        content=stream.getvalue()
        def open_request(request,timeout):
            start,end=map(int,re.fullmatch(r'bytes=(\d+)-(\d+)',request.get_header('Range')).groups())
            return Response(content,start,end)
        reader=RemoteZipReader('https://example.invalid/archive.zip',len(content))
        with patch('urllib.request.urlopen',side_effect=open_request):
            with zipfile.ZipFile(reader) as archive:
                self.assertEqual(archive.read('rgb/000010.jpg'),b'example-content'*100)
        self.assertGreater(reader.bytes_received,0)
        with self.assertRaises(ValueError): reader.seek(-1)
        with self.assertRaises(ValueError): reader.seek(0,3)

    def test_ignored_range_is_rejected_without_reading_body(self):
        response=Response(b'x'*100,0,99,status=200)
        with patch('urllib.request.urlopen',return_value=response):
            with self.assertRaises(ValueError): RemoteZipReader('https://example.invalid/a',100).read(1)
        self.assertFalse(response.was_read)

    def test_budget_blocks_request_before_network_access(self):
        with patch('urllib.request.urlopen') as opener:
            with self.assertRaises(ValueError): RemoteZipReader('https://example.invalid/a',100000,budget=8).read(1)
        opener.assert_not_called()


if __name__=='__main__': unittest.main()
