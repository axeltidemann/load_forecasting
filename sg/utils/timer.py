
import time
import sys
from datetime import timedelta as dt

class SimpleTimer():
    """Basic timer class, initial code by Lester.

    Basic usage:
    timer = SimpleTimer() 
    ...lost of slow code here...
    # Optional, will be called by destructor unless called manually:
    report = timer.end()
    print report

    Use cProfile for more in-depth profiling."""
    
    def __init__(self, output_stream=sys.stdout):
        """Start timing (may be restarted by explicit calls to start()). Output
        printed by start() and end() will be printed to output_stream unless
        this is Null."""
        self.times = []
        self.labels = []
        self._has_ended = False
        self._stream = output_stream
        self.start()

    def __del__(self):
        if (not self._has_ended) and self._stream is not None:
            print >>self._stream, self.end()
            
    def start(self):
        self.times = [time.time()]
        self.labels = ["start"]
        if self._stream is not None:
            print >>self._stream, "Started at", time.asctime()

    def end(self):
        self._has_ended = True
        self.lap("end")
        if self._stream is not None:
            print >>self._stream, "Ended at", time.asctime()
        return self.report()

    def lap(self,label):
        self.times.append(time.time())
        self.labels.append(label)

    @staticmethod
    def seconds_to_string(seconds):
        whole_secs = int(seconds)
        micros = int((seconds - whole_secs) * 1000000)
        delta = dt(seconds=whole_secs, microseconds=micros)
        if micros == 0:
            return str(delta)
        else:
            return str(delta)[:-4]
    
    @staticmethod
    def period_to_string(start_time, end_time):
        seconds = (end_time - start_time)
        return SimpleTimer.seconds_to_string(seconds)

    def report(self):
        s = "Finished in %s: " % self.period_to_string(self.times[0], 
                                                       self.times[-1])
        for i in range(1,len(self.labels)-1):
            s += "%s %s, " % (self.labels[i], 
                              self.period_to_string(self.times[i-1], 
                                                    self.times[i]))
        return s[:-2] + "."


if __name__ == "__main__":
    from unittest import main
    main(module="test_" + __file__[:-3])
    
