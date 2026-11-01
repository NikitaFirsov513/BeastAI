import unittest
import baseline


def box(x, y, width, height, score=0.9):
    return dict(x=x, y=y, width=width, height=height, score=score)


class BaselineTests(unittest.TestCase):
    def test_animal_filter_accepts_the_new_single_class_head(self):
        self.assertEqual(baseline.animal_class_ids({0:'animal'}),[0])
        self.assertEqual(baseline.animal_class_ids({0:'person',1:'bird',2:'cow',3:'car'}),[1,2])

    def test_iou_clips_to_image_and_handles_empty_union(self):
        self.assertAlmostEqual(baseline.overlap(box(-.1,0,.2,.2),box(0,0,.1,.2)),1)
        self.assertEqual(baseline.overlap(box(0,0,0,0),box(0,0,0,0)),0)
        self.assertEqual(baseline.overlap(box(0,0,.1,.1),box(.8,.8,.1,.1)),0)

    def test_duplicate_predictions_cannot_match_one_animal_twice(self):
        reference=[box(.1,.1,.2,.2)]
        predictions=[box(.1,.1,.2,.2,.7),box(.1,.1,.2,.2,.9),box(.7,.7,.1,.1,.6)]
        result=baseline.match_boxes(predictions,reference)
        self.assertEqual((result['tp'],result['fp'],result['fn']),(1,2,0))

    def test_missing_detections_and_imperfect_overlap(self):
        reference=[box(.1,.1,.2,.2),box(.7,.7,.1,.1)]
        self.assertEqual(baseline.match_boxes([],reference)['fn'],2)
        result=baseline.match_boxes([box(.1,.1,.01,.01)],reference)
        self.assertEqual((result['tp'],result['fp'],result['fn']),(0,1,2))


if __name__=='__main__':
    unittest.main()
