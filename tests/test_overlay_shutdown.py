"""Shutdown regressions: hung child, partial IPC, missing completion and close during load."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import os,queue,sys,time,unittest
from overlay_process import DetectorWorker
from screen_overlay import App,Region
from project_paths import PROJECT_ROOT

@unittest.skipUnless(os.name=='nt','Windows overlay')
class WorkerShutdownTests(unittest.TestCase):
    def make_worker(self,mode,grace=.15):
        worker=DetectorWorker(Region(0,0,200,100),'reference',.25,3,queue.Queue(maxsize=1),queue.Queue(),1001,grace_seconds=grace)
        self.addCleanup(worker.force_shutdown)
        worker.start([sys.executable,'-u',str(PROJECT_ROOT/'tests/fixtures/overlay_worker_fixture.py'),worker.cancel.name,mode])
        return worker
    def wait_for_exit(self,worker,timeout=4):
        deadline=time.monotonic()+timeout
        while worker.is_alive() and time.monotonic()<deadline:
            worker.poll_shutdown();time.sleep(.01)
        self.assertFalse(worker.is_alive(),'detector process outlived shutdown deadline')
    def test_cooperative_stop_preserves_normal_exit(self):
        worker=self.make_worker('cooperative',1.5)
        self.assertEqual(worker.events.get(timeout=4)[1:],('status','fixture-ready'))
        worker.request_stop();self.wait_for_exit(worker)
        self.assertEqual(worker.exitcode,0);self.assertFalse(worker.forced)
    def test_stuck_child_is_terminated_and_repeat_stop_keeps_deadline(self):
        worker=self.make_worker('hang');worker.events.get(timeout=4)
        started=time.monotonic();worker.request_stop();deadline=worker.deadline
        worker.request_stop();self.assertEqual(worker.deadline,deadline)
        self.wait_for_exit(worker);self.assertTrue(worker.forced)
        self.assertLess(time.monotonic()-started,3)
    def test_partial_pipe_message_does_not_block_shutdown(self):
        worker=self.make_worker('partial')
        worker.request_stop();self.wait_for_exit(worker)
        self.assertTrue(worker.forced)
    def test_process_exit_without_stopped_message_is_detected(self):
        worker=self.make_worker('exit');self.wait_for_exit(worker)
        self.assertEqual(worker.exitcode,7)
        self.assertTrue(worker.events.empty())
    def test_named_stop_signal_is_shared_with_child(self):
        from screen_overlay_win32 import StopSignal
        owner=StopSignal();reader=StopSignal(owner.name,open_existing=True)
        try:
            self.assertFalse(reader.is_set());owner.set();self.assertTrue(reader.wait(.1))
        finally:reader.close();owner.close()

class WindowCloseTests(unittest.TestCase):
    def test_close_hides_windows_immediately_while_detector_is_busy(self):
        app=App.__new__(App);app.closing=False;app.tk=SimpleNamespace(TclError=RuntimeError)
        app.root=Mock();app.worker=Mock();app.selector=Mock();app.fixture=Mock();app.overlay=Mock();app.hotkey=Mock();app.status=Mock();app.stop_button=Mock()
        selector,fixture,overlay=app.selector,app.fixture,app.overlay
        app.close();app.close()
        app.worker.request_stop.assert_called_once_with('window_close')
        app.root.withdraw.assert_called_once();app.root.destroy.assert_not_called()
        selector.destroy.assert_called_once();fixture.destroy.assert_called_once();overlay.destroy.assert_called_once()
    def test_poll_closes_when_child_exits_without_completion_event(self):
        app=App.__new__(App);app.events=queue.Queue();app.generation=1;app.worker=Mock();app.worker.is_alive.return_value=False
        app.closing=True;app.root=Mock()
        def finished():app.worker=None
        app.finish_worker=Mock(side_effect=finished)
        app.poll();app.finish_worker.assert_called_once();app.root.destroy.assert_called_once();app.root.after.assert_not_called()

if __name__=='__main__':unittest.main()
