import subprocess
import time
import sys
import os

class JobSubmitter(object):
    def __init__(self, stream):
        self.batch_size = 1
        self.max_in_queue = 10
        self.wait_time_secs = 5
        self._jobs = [job[:-1] for job in stream]
        self._user = os.environ['USER']
        self._queue_status = ""
        self._slots_available = 0
        
    def _sys_cmd(self, cmd):
        #print "_sys_cmd with command:", cmd
        return subprocess.check_output(cmd, shell=True)

    def _log(self, *args):
        sys.stdout.write(*args)
        sys.stdout.write("\n")
        sys.stdout.flush()
        
    def _jobs_remaining(self):
        return len(self._jobs) > 0

    def _is_job_running(self, job):
        return self._queue_status.find(job) >= 0
    
    def _update_queue_status(self):
        self._queue_status = self._sys_cmd("qstat -f")

    def _resources_available(self):
        cmd = "qstat |grep %s |wc |awk '{print $1}'" % self._user
        in_queue = int(self._sys_cmd(cmd)[:-1])
        self._slots_available = self.max_in_queue - in_queue
        return self._slots_available > 0

    def _submit_more_jobs(self):
        self._update_queue_status()
        submitted = 0
        max_submissions = min(self._slots_available, self.batch_size)
        while self._jobs and submitted < max_submissions:
            job = self._jobs.pop(0)
            if self._is_job_running(job):
                self._log("Skipping job, already running: " + job)
            else:
                self._sys_cmd("qsub " + job)
                self._log("Submitted job at " + time.strftime("%b. %d, %X: ") \
                          + job)
                submitted += 1
        return submitted

    def _wait(self, brief=False):
        if brief:
            time.sleep(15)
        else:
            time.sleep(self.wait_time_secs)

    def submit_jobs(self):
        self._log("Queueing " + str(len(self._jobs)) + " jobs for submission in " + \
                  "batches of " + str(self.batch_size) + ". Polling the queue " + \
                  "for free space every " + str(self.wait_time_secs/60.0) + " minutes.")
        while self._jobs_remaining():
            brief_wait = False
            if self._resources_available():
                num_submitted = self._submit_more_jobs()
                brief_wait = num_submitted < self._slots_available
            self._wait(brief=brief_wait)
        self._log("All jobs submitted. Bye for now.")

def get_options():
    """Add prediction-related options to the parser. If no parser is provided, one
    will be created."""
    import optparse
    parser = optparse.OptionParser()
    parser.usage = "[options] [jobfile]"
    parser.description = "Send jobs in batches to the queueing system. The list of jobs can be sent to stdin or be stored in jobfile."
    parser.add_option("--wait", dest="wait", type="float", help="Wait time in minutes between each check of the queue", default=10)
    parser.add_option("--queued", dest="queued", type="long", help="Max number of jobs in the queue at once", default=10)
    parser.add_option("--batch", dest="batch", type="long", help="Number of jobs to submit each time", default=1)
    (options, args) = parser.parse_args()
    return options, args

def make_submitter(path=None):
    if path is None:
        if sys.stdin.isatty():
            sys.stderr.write("You must submit a path or cat jobs to stdin.")
            exit(1)
        print "Reading jobs from standard input."
        return JobSubmitter(sys.stdin)
    else:
        print "Reading jobs from " + path + "."
        with open(path, "r") as f:
            return JobSubmitter(f)
    
if __name__ == "__main__":
    options, args = get_options()
    submitter = make_submitter(args[0] if args else None)
    submitter.wait_time_secs = options.wait * 60
    submitter.batch_size = options.batch
    submitter.max_in_queue = options.queued
    submitter.submit_jobs()
