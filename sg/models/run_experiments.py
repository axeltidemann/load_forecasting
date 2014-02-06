from multiprocessing import Pool
import os
import socket

from sg.data.sintef import tempfeeder_exp
from sg.utils.timer import SimpleTimer

import run_experiments_params as params

def run_one_wrapper(arg):
    reload(params)
    params.run_one(arg)
    
def make_runs(user_ids, num_runs):
    """Create a list of (user_id, run_number) pairs that can be sent via
    pool.map to the run_one function."""
    return [(user, run) for user in user_ids for run in range(num_runs)]

def run_simulations(runs):
    """Run all the simulations provided in runs by sending them on to the
    run_one function."""
    num_parallel_processes = 12
    pool = Pool(processes=num_parallel_processes)
    pool.map(run_one_wrapper, runs, chunksize=1)

if __name__ == "__main__":
    # if socket.gethostname() == "tanzenmusik.idi.ntnu.no":
    #     user_ids = tempfeeder_exp().user_ids[25:50]
    # else:
    #     user_ids = tempfeeder_exp().user_ids[0:25]

    user_ids = [tempfeeder_exp().user_ids[0]]
    num_runs = 12

    print "Master pid is %d " % os.getpid()
    timer = SimpleTimer(output_stream=None)
    tempfeeder_exp().close()
    runs = make_runs(user_ids, num_runs)
    run_simulations(runs)
    print "All simulations complete. %s" % timer.end()
    tempfeeder_exp().close()
