import unittest
import StringIO
import time

from timer import *

class TimerTester(unittest.TestCase):
    def _wrapped_timing(self, stream):
        t = SimpleTimer(stream)
        time.sleep(0.1)
        
    def test_report_when_out_of_scope(self):
        stream = StringIO.StringIO()
        self._wrapped_timing(stream)
        output = stream.getvalue()
        self.assertIn("Started at", output)
        self.assertIn("Ended at", output)

        
if __name__ == "__main__":
    unittest.main()
