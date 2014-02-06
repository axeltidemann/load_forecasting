#!/usr/bin/python

import re
import sys

def rfilter(stream):
    messages = ["In log\(s2\) : NaNs produced",
                "^Warning message[s]*:",
                "[Ii]n arima\(x = loads, order = order, xreg = temp_hc\)",
                "non-stationary AR part from CSS",
                "possible convergence problem: optim gave code=",
                "Error in optim\(init\[mask\], armafn",
                "non-finite finite-difference value",
                "There were [0-9]* warnings \(use warnings\(\) to see them\)"]
    re_objs = [re.compile(msg) for msg in messages]
    for line in stream:
        do_filter = False
        for prog in re_objs:
            if prog.search(line) is not None:
                do_filter = True
        if not do_filter:
            print line[:-1]

if __name__ == "__main__":
    if len(sys.argv) == 1:
        rfilter(sys.stdin)
    else:
        for path in sys.argv[1:]:
            with open(path, "r") as f:
                rfilter(f)
    
