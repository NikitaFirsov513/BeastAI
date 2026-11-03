import unittest
from evaluate_drone_reference import map_animal_rows
class ReferenceMappingTests(unittest.TestCase):
    def test_filters_person_vehicle_and_normalizes_animal_class(self):
        rows=map_animal_rows([[0,0,10,10],[20,30,60,90],[1,2,3,4]],[.9,.8,.7],[0,1,2],100,120)
        self.assertEqual(len(rows),1);r=rows[0];self.assertEqual(r['class_id'],0);self.assertEqual(r['class_name'],'animal');self.assertEqual(r['native_class_id'],1)
        self.assertAlmostEqual(r['x'],.2);self.assertAlmostEqual(r['y'],.25);self.assertAlmostEqual(r['width'],.4);self.assertAlmostEqual(r['height'],.5)
    def test_clamps_boxes_and_discards_invalid_area(self):
        rows=map_animal_rows([[-10,-20,110,140],[20,20,10,10]],[.8,.7],[1,1],100,120)
        self.assertEqual(len(rows),1);self.assertEqual([rows[0][k] for k in ('x','y','width','height')],[0.,0.,1.,1.])
if __name__=='__main__':unittest.main()
