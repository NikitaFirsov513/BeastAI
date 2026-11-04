import unittest
from analyse_roe_expanded_errors import errors

def box(x=.1,y=.1,w=.1,h=.1,score=.9):return dict(x=x,y=y,width=w,height=h,score=score)
def row(truth,predictions,tp,fp,fn):return {'reference':truth,'boxes':predictions,'counts':{'tp':tp,'fp':fp,'fn':fn},'original_size':[1280,1280]}

class ErrorAnalysisTests(unittest.TestCase):
    def test_duplicate_person_and_unannotated_overlap_are_distinguished(self):
        target=box();person=box(x=.5);duplicate=box(score=.8);background=box(x=.8)
        false,missed=errors(row([target],[target,duplicate,person,background],1,3,0),[person])
        self.assertEqual({r['type'] for r in false},{'duplicate_of_matched_reference','overlaps_annotated_person','no_overlap_with_annotated_targets'})
        self.assertEqual(missed,[])
    def test_single_box_cannot_match_two_references(self):
        first=box();neighbor=box(x=.12)
        false,missed=errors(row([first,neighbor],[first],1,0,1),[])
        self.assertEqual(false,[]);self.assertEqual(len(missed),1);self.assertEqual(missed[0]['type'],'one_to_one_assignment_conflict')
    def test_partial_overlap_is_both_false_box_and_missed_small_reference(self):
        target=box(w=.01,h=.01);shifted=box(x=.105,w=.01,h=.01)
        false,missed=errors(row([target],[shifted],0,1,1),[])
        self.assertEqual(false[0]['type'],'partial_animal_overlap_below_match_iou');self.assertEqual(missed[0]['type'],'overlap_below_match_iou');self.assertTrue(missed[0]['small_at_1024'])
    def test_no_answers_are_misses(self):
        false,missed=errors(row([box()],[],0,0,1),[])
        self.assertEqual(false,[]);self.assertEqual(missed[0]['type'],'no_emitted_overlap')

if __name__=='__main__':unittest.main()
