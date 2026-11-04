import queue,unittest
from screen_overlay import Region,selected_region,screen_rect,put_latest
class OverlayGeometryTests(unittest.TestCase):
    def test_reverse_drag_and_negative_monitor(self):
        r=selected_region((-100,600),(-1000,100),(-1920,0,3840,1080))
        self.assertEqual(r,Region(-1000,100,-100,600));self.assertEqual(r.width,900)
    def test_clamp_and_too_small(self):
        self.assertEqual(selected_region((-20,-30),(2000,2000),(0,0,1920,1080)),Region(0,0,1920,1080))
        self.assertIsNone(selected_region((0,0),(99,79),(0,0,1920,1080)))
    def test_box_coordinates_use_region_not_model_input(self):
        box={'x':.25,'y':.2,'width':.5,'height':.4}
        expected=(200,80,600,240)
        for actual,value in zip(screen_rect(box,800,400),expected):self.assertAlmostEqual(actual,value)
        self.assertIsNone(screen_rect({'x':2,'y':2,'width':1,'height':1},800,400))
    def test_latest_only_no_frame_backlog(self):
        q=queue.Queue(maxsize=1)
        for n in range(100):put_latest(q,n)
        self.assertEqual(q.qsize(),1);self.assertEqual(q.get_nowait(),99)
if __name__=='__main__':unittest.main()
