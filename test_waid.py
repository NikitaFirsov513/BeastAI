import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
import urllib.request

import waid


class WaidTests(unittest.TestCase):
    def test_coordinates_and_distinct_label_states(self):
        with tempfile.TemporaryDirectory(dir=waid.ROOT) as folder:
            path = Path(folder) / "a.txt"
            self.assertEqual(waid.parse_labels(path, ["animal"])["status"], "missing")
            path.write_text("", encoding="utf-8")
            self.assertEqual(waid.parse_labels(path, ["animal"])["status"], "empty")
            path.write_text("0 0.5 0.6 0.2 0.4\n", encoding="utf-8")
            result = waid.parse_labels(path, ["animal"])
            self.assertEqual(result["status"], "ok")
            self.assertAlmostEqual(result["boxes"][0]["x"], 0.4)
            self.assertAlmostEqual(result["boxes"][0]["y"], 0.4)
            path.write_text("0 nan 0.5 0.1 0.2\n1 0.5 0.5 0.1 0.2\nwrong\n", encoding="utf-8")
            result = waid.parse_labels(path, ["animal"])
            self.assertEqual(result["status"], "invalid")
            self.assertEqual(len(result["errors"]), 3)
            self.assertEqual(result["boxes"], [])

    def test_download_resume_repairs_corruption(self):
        content = b"test-image-bytes"
        entry = {"path": "WAID/images/train/a.jpg", "sha": waid.blob_hash(content), "size": len(content)}
        with tempfile.TemporaryDirectory(dir=waid.ROOT) as folder, patch.object(waid, "DATA", Path(folder)), patch.object(waid, "fetch", return_value=content) as fetch:
            self.assertTrue(waid.download_file(entry, "revision"))
            self.assertFalse(waid.download_file(entry, "revision"))
            self.assertEqual(fetch.call_count, 1)
            waid.local_file(entry).write_bytes(b"x" * len(content))
            self.assertTrue(waid.download_file(entry, "revision"))
            self.assertEqual(fetch.call_count, 2)

    def test_failed_integrity_does_not_create_image(self):
        entry = {"path": "WAID/images/train/a.jpg", "sha": waid.blob_hash(b"good"), "size": 4}
        with tempfile.TemporaryDirectory(dir=waid.ROOT) as folder, patch.object(waid, "DATA", Path(folder)), patch.object(waid, "fetch", return_value=b"evil"):
            with self.assertRaises(ValueError):
                waid.download_file(entry, "revision")
            self.assertFalse(waid.local_file(entry).exists())

    def test_paths_cannot_escape_dataset(self):
        for path in ("WAID/images/train/../../secret", "WAID/images/train/..\\secret", "WAID/images/wrong/a.jpg"):
            with self.assertRaises(ValueError):
                waid.local_file({"path": path})

    def test_small_preview_covers_splits_and_is_repeatable(self):
        pairs = [{"split": split, "name": str(i)} for split in waid.SPLITS for i in range(10)]
        selected = waid.sample_pairs(pairs, 7)
        self.assertEqual(len(selected), 7)
        self.assertEqual({p["split"] for p in selected}, set(waid.SPLITS))
        self.assertEqual(selected, waid.sample_pairs(pairs, 7))
        self.assertEqual(len(waid.sample_pairs(pairs, 100)), 30)

    def test_http_serves_only_available_images_and_paginates(self):
        with tempfile.TemporaryDirectory(dir=waid.ROOT) as folder, patch.object(waid, "DATA", Path(folder)):
            image = {"path": "WAID/images/test/a.jpg"}
            label = {"path": "WAID/labels/test/a.txt"}
            waid.local_file(image).parent.mkdir(parents=True)
            waid.local_file(image).write_bytes(b"image")
            index = {"revision": "abc", "classes": ["animal"], "pairs": [{"split": "test", "name": "a.jpg", "image": image, "label": label}]}
            server = waid.ThreadingHTTPServer(("127.0.0.1", 0), waid.make_handler(index))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                base = "http://127.0.0.1:" + str(server.server_port)
                with urllib.request.urlopen(base + "/api/items?page=999") as response:
                    result = json.load(response)
                self.assertEqual(result["page"], 0)
                self.assertEqual(result["items"][0]["status"], "missing")
                with urllib.request.urlopen(base + "/data/images/test/a.jpg") as response:
                    self.assertEqual(response.read(), b"image")
                with self.assertRaises(urllib.error.HTTPError) as error:
                    urllib.request.urlopen(base + "/data/../index.json")
                self.assertEqual(error.exception.code, 404)
            finally:
                server.shutdown()
                server.server_close()
                thread.join()


if __name__ == "__main__":
    unittest.main()
