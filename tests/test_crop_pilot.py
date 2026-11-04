"""Geometry tests for label clipping and full-frame crop search."""
import unittest
import numpy as np
from evaluate_crop_pilot import windows
from prepare_crop_lesson import clipped_labels
from tiled_inference import map_box
class CropGeometryTests(unittest.TestCase):
    def test_grid_covers_entire_image_without_reference_input(self):
        for w,h,expected in [(1280,1280,9),(1024,683,6),(300,240,1)]:
            grid=windows(w,h,'tiles');self.assertEqual(len(grid),expected);coverage=np.zeros((h,w),np.uint8)
            for x1,y1,x2,y2 in grid:
                self.assertTrue(0<=x1<x2<=w and 0<=y1<y2<=h);coverage[y1:y2,x1:x2]+=1
            self.assertTrue((coverage>0).all())
    def test_animal_crossing_boundary_gets_clipped_label(self):
        source=[{'x':500/1280,'y':100/1280,'width':24/1280,'height':20/1280}]
        labels,mapping=clipped_labels(source,0,0);self.assertEqual(len(labels),1);self.assertAlmostEqual(labels[0][3],12/512);self.assertAlmostEqual(mapping[0]['visible_fraction'],.5);self.assertTrue(mapping[0]['truncated'])
    def test_label_and_prediction_coordinates_return_to_original(self):
        source=[{'x':400/1280,'y':420/1280,'width':24/1280,'height':20/1280}]
        labels,mapping=clipped_labels(source,384,384);_,cx,cy,w,h=labels[0]
        box=map_box([(cx-w/2)*512,(cy-h/2)*512,(cx+w/2)*512,(cy+h/2)*512,.8,0],(384,384,896,896),(1280,1280),{0:'animal'})
        for key in ('x','y','width','height'):self.assertAlmostEqual(box[key],source[0][key])
        self.assertFalse(mapping[0]['truncated'])
    def test_touching_edge_does_not_create_zero_area_label(self):
        labels,_=clipped_labels([{'x':512/1280,'y':0.,'width':10/1280,'height':10/1280}],0,0);self.assertEqual(labels,[])
if __name__=='__main__':unittest.main()
