import os
import sys
import time

from sg.globals import SG_MODELS_PATH
from sg.utils.timer import SimpleTimer
from sg.globals import SG_SIM_PATH

# This function is defined in a separate file, so the main runner can reload it
# before each launch. This allows us to adjust parameters "on the fly".
def run_one(arg):
    """Run one evolution. Arg is a tuple containing user ID and run number."""
    user_id, run_number = arg
    # Note that the PID printed below is the PID in which this function is
    # running, which is different from the PID of the evolution.
    print "Launching evolution for user %d run %d (pid %d) at %s..." % \
      (user_id, run_number, os.getpid(), time.asctime())
    sys.stdout.flush()
      
    timer = SimpleTimer(output_stream=None)
    out_dir = os.path.join(SG_SIM_PATH, "id_%d" % user_id)
    model = os.path.join(SG_MODELS_PATH, "load_prediction.py")
    postfix = "run_%d" % run_number
    generations = 50
    pop_size = 400
    mutation = 0.05
    crossover = 0.5
    # NB Total-load sims:
    total = " --total-load"
    data_seed = 12
    
    stdout_path = os.path.join(out_dir, 
        "output_run_%d.txt" % run_number)
    os.system("test -d %s || mkdir -p %s" % (out_dir, out_dir))
    os.system("python %s " % model + \
              " --userid=%d" % user_id + \
              " --out-dir=%s --out-postfix=%s " % (out_dir, postfix) + \
              " --generations=%d --pop-size=%d " % (generations, pop_size) + \
              " --mutation=%f --crossover=%f " % (mutation, crossover) + \
              " --no-show-plot --save-plot " + \
              total + \
              " --data-seed=%d " % data_seed + \
              " >%s" % stdout_path)

    print "Evolution completed for user %d run %d. %s" \
      % (user_id, run_number, timer.end())
    sys.stdout.flush()
