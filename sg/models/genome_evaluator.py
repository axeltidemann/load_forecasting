"""Use this program to evaluate one genome at a time, read from standard
input."""

import sys
import ast
import traceback
import random

import matplotlib.pyplot as plt

import sg.utils.pyevolve_utils as pu
import sg.utils
import ga
import sg.data.sintef.userloads as ul
import load_prediction as lp
from load_prediction_ar import *
from load_prediction_ar24 import *
from load_prediction_arima import *
from load_prediction_dshw import *
from load_prediction_esn import *
from load_prediction_esn24 import *
try:
    from load_prediction_CBR import *
    from load_prediction_wavelet import *
    from load_prediction_wavelet24 import *
except ImportError:
    print >>sys.stderr, "Genome evaluator can't import CBR/wavelet modules, probably some of the dependencies are not installed."

options = None
def get_options():
    global options
    parser = lp.prediction_options()
    parser = lp.ga_options(parser)
    parser = lp.data_options(parser)
    parser.add_option("--model", dest="model", help="The model class that the genomes instantiate", default=None)
    parser.add_option("--test-set", dest="test_set", action="store_true",
                      help="Test the genomes on the test set, rather than on the training set", default=False)
    parser.add_option("--plot", dest="plot", action="store_true",
                      help="Make a plot (in combination with --test-set)", default=False)
    (options, args) = parser.parse_args()
    lp.options = options
    if options.model is None:
        print >>sys.stderr, "Model argument is required."
        sys.exit(1)

def read_next_genome_list():
    print "Enter genome to be evaluated: "
    line = sys.stdin.readline()
    if line == "":
        print "End of input, exiting."
        sys.exit(0)
    return ast.literal_eval(line)

def next_indiv():
    gl = read_next_genome_list()
    genome = pu.AllelesGenome()
    genome.setInternalList(gl)
    genome.setParams(num_trials=options.num_trials)
    return genome
    
def gene_test_loop(model):
    while sys.stdin:
        ga._model = model
        indiv = next_indiv()
        if options.test_set:
            print "Evaluating genome on test set: ", indiv[:]
            sys.stdout.flush()
            try:
                (target, predictions) = lp.parallel_test_genome(indiv, model) if options.parallel else lp.test_genome(indiv, model)
            except Exception, e:
                print >>sys.stderr, "Exception raised, failed to evaluate genome."
                tb = "  " + traceback.format_exc(limit=50)[:-1]
                print >>sys.stderr, tb.replace("\n", "\n  ")
                continue
            error = sg.utils.concat_and_calc_error(predictions, target, model.error_func)
            print "Error on test phase: {}".format(error)
            if options.plot:
                sg.utils.plot_target_predictions(target, predictions)
                plt.show()
        else:
            print "Evaluating genome on training set: ", indiv[:]
            sys.stdout.flush()
            fitness = ga._fitness(indiv)
            print "Fitness:", fitness
            if fitness != 0:
                print "Error:", ga._fitness_to_error(fitness)
            else:
                print "Error not calculated for 0 fitness."

def run(): 
    """."""
    get_options()
    prev_handler = np.seterrcall(lp.float_err_handler)
    prev_err = np.seterr(all='call')
    np.seterr(under='ignore')
    random.seed(options.seed)
    np.random.seed(options.seed)
    model_creator = eval(options.model + "(options)")
    model = model_creator.get_model()
    lp._print_sim_context(model._dataset)
    print "Number of training sequences: %d" % options.num_trials 
    print "Start days of training sequences:", model._dataset.train_periods_desc
    gene_test_loop(model)
    ul.tempfeeder_exp().close()

if __name__ == "__main__":
    run()
    
