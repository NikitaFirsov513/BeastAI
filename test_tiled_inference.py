import unittest
import baseline
from tiled_inference import axis_starts,tile_windows,map_box,merge_boxes

class TileGeometryTests(unittest.TestCase):
    def test_four_crops_cover_frame_and_stay_inside_it(self):
        windows=tile_windows(1024,683);self.assertEqual(len(windows),4)
        for y in range(683):
            spans=sorted((a,c) for a,b,c,d in windows if b<=y<d)
            right=0
            for a,c in spans:self.assertLessEqual(a,right);right=max(right,c)
            self.assertEqual(right,1024)
        for a,b,c,d in windows:self.assertTrue(0<=a<c<=1024 and 0<=b<d<=683)
    def test_box_coordinates_return_to_original_frame(self):
        mapped=map_box([10,20,110,80,.9,0],(384,235,1024,683),(1024,683),{0:'animal'})
        truth=dict(x=394/1024,y=255/683,width=100/1024,height=60/683)
        self.assertAlmostEqual(baseline.overlap(mapped,truth),1)
    def test_invalid_and_outside_crop_boxes_are_clipped(self):
        self.assertIsNone(map_box([650,0,700,20,.9,0],(384,235,1024,683),(1024,683),{0:'animal'}))
        b=map_box([-10,-10,30,40,.9,0],(0,0,640,448),(1024,683),{0:'animal'})
        self.assertEqual((b['x'],b['y']),(0,0));self.assertAlmostEqual(b['width'],30/1024)
    def test_cross_crop_duplicates_removed_adjacent_animals_retained(self):
        a=dict(x=.1,y=.1,width=.1,height=.1,score=.9)
        duplicate=dict(a,score=.8);neighbor=dict(a,x=.21,score=.7)
        self.assertEqual(merge_boxes([duplicate,neighbor,a]),[a,neighbor])
    def test_animal_on_central_seams_fits_whole_in_a_crop(self):
        animal=(490,320,540,370)
        self.assertTrue(any(a<=animal[0] and b<=animal[1] and c>=animal[2] and d>=animal[3] for a,b,c,d in tile_windows(1024,683)))
        self.assertEqual(tile_windows(100,80),[(0,0,100,80)])
        with self.assertRaises(ValueError):axis_starts(1024,640,1)

if __name__=='__main__':unittest.main()
