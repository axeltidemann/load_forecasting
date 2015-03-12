import random
from datetime import timedelta as dt
import sys
import os
import traceback
from multiprocessing import Lock
import time
import copy

import numpy as np
from pyevolve import G1DList
from pyevolve import GSimpleGA
from pyevolve import Initializators
from pyevolve import Selectors
from pyevolve import DBAdapters
from pyevolve import Consts
from pyevolve import Scaling
import pyevolve
import matplotlib.pyplot as plt

import sg.utils
import sg.utils.pyevolve_utils as pu
from sg.utils.cache import ATimeCache
import gui
from sg.utils.timer import SimpleTimer
from sg.utils.pyevolve_utils import G1DListMutatorRealGaussianFixed

# Set to True for more robust evolution, set to False for easier debugging of
# fitness evaluation.
_catch_exceptions_during_fitness_evaluation = True
_step_callbacks = []

_cache_mutex = Lock()
_cache = ATimeCache(1000)

def _error_to_fitness(error):
    return 1. / (1 + error)

def _fitness_to_error(fitness):
    if fitness == 0:
        return 1e9
    return 1. / fitness - 1

def _make_cache_key(genome, data_in, data_out):
    return tuple(sg.utils.safe_deep_flatten(
        genome + [data_in.index[0].value, data_in.index[-1].value,
                  data_out.index[0].value, data_out.index[-1].value]))

def process_data(data_in, genome, model, loci):
    data_now = copy.copy(data_in)
    for preproc in model.preprocessors:
        data_now = preproc(data_now, genome, loci, model.day)
    model_out = model.transformer(data_now, genome, loci, model.day)
    for postproc in reversed(model.postprocessors):
        model_out = postproc(model_out, genome, loci)
    return model_out
    
_model = None
# The model used to be part of the individual, but this means the entire
# dataset is pickled and sent over the network on every genome evaulation when
# running under MPI, which is actually a substantial overhead.
def _fitness(indiv):
    """Generic fitness function, calls preprocessors, transformer and
    postprocessors for each model, calculates error from error function."""
    global _model
    model = _model
    num_trials = indiv.getParam('num_trials')
    genome = pu.raw_genes(indiv, True)
    loci = model.loci

    def test_one(data_in, data_out):
        key = _make_cache_key(genome, data_in, data_out)
        _cache_mutex.acquire()
        try:
            error = _cache[key]
        except KeyError:
            _cache_mutex.release()
            model_out = process_data(data_in, genome, model, loci)
            
            assert(np.all(model_out.index == data_out.index))            
            error = model.error_func(model_out.values, data_out.values)
            _cache_mutex.acquire()
            _cache[key] = error
        _cache_mutex.release()
        return error
        
    def test_loop():
        error = 0
        #start = time.time()
        train_iter = model.dataset.train_data_iterator()
        for (data_in, data_out) in train_iter():
            error += test_one(data_in, data_out)
        #print "Testing completed in ", time.time() - start
        #sys.stdout.flush()
        return error

    global _catch_exceptions_during_fitness_evaluation
    if _catch_exceptions_during_fitness_evaluation:
        try:
            error = test_loop()
        except Exception, e:
            print >>sys.stderr, "Caught exception during fitness evaluation!"
            print >>sys.stderr, "  Offending genome was:", genome[:]
            tb = "  " + traceback.format_exc(limit=50)[:-1]
            print >>sys.stderr, tb.replace("\n", "\n  ")
            print >>sys.stderr, "  Setting fitness to 0 to avoid selection."
            return 0
    else:
        error = test_loop()
    error = float(error) / num_trials
    fitness = _error_to_fitness(error)
    if np.isnan(fitness):
        print >>sys.stderr, "Evaluation resulted in NaN fitness. Setting " \
            "to 0 to avoid propagation during fitness scaling."
        fitness = 0
    return fitness

def print_best_genome(ga_engine):
    best = ga_engine.bestIndividual()
    gen = ga_engine.getCurrentGeneration()
    raw_fitn = best.getRawScore()
    print "Best genome at generation %d had fitness %f (raw %f), error %f." % \
      (gen, best.getFitnessScore(), raw_fitn, _fitness_to_error(raw_fitn))
    print "Best genome at generation %d:" % gen, pu.raw_genes(best, True)
    sys.stdout.flush()

def print_population(ga_engine):
    for indiv in ga_engine.getPopulation():
        print "%9e, %9e" % (indiv.score, indiv.fitness), pu.raw_genes(indiv, True)
        
_lpgui = None
def update_gui(ga_engine):
    global _lpgui
    if _lpgui is None:
        _lpgui = gui.LoadPredictionGUI()
    _lpgui.update(ga_engine)
    
_time_of_prev_gen = [time.time(), time.time()]
def report_time_spent(ga_engine):
    global _time_of_prev_gen
    now = time.time()
    gen = ga_engine.getCurrentGeneration()
    print "Time spent on generation %s, total %s." % \
      (SimpleTimer.period_to_string(_time_of_prev_gen[1], now),
       SimpleTimer.period_to_string(_time_of_prev_gen[0], now))
    _time_of_prev_gen[1] = now
        
_live_fig = None
_generations = []
_min_raw = []
_avg_raw = []
_max_raw = []
_dev_raw = []
_min_fitn = []
_avg_fitn = []
_max_fitn = []
# No "fitness deviation" in Statistics class.
def plot_fitnesses(ga_engine):
    _generations.append(ga_engine.getCurrentGeneration())
    stats = ga_engine.getPopulation().stats
    _max_raw.append(stats["rawMax"])
    _min_raw.append(stats["rawMin"])
    _avg_raw.append(stats["rawAve"])
    _dev_raw.append(stats["rawDev"])
    _max_fitn.append(stats["fitMax"])
    _min_fitn.append(stats["fitMin"])
    _avg_fitn.append(stats["fitAve"])
    _live_fig.clear()
    ax_raw = _live_fig.add_subplot(211)
    ax_raw.plot(_generations, _max_raw, label="Maximum Score")
    ax_raw.plot(_generations, _avg_raw, label="Average Score")
    ax_raw.plot(_generations, _min_raw, label="Minimum Score")
    ax_raw.plot(_generations, _dev_raw, label="Deviation Score")
    ax_raw.set_ylabel('Raw Score')
    ax_raw.legend(loc=(0.2, 0.2))
    ax_fit = _live_fig.add_subplot(212)
    ax_fit.plot(_generations, _max_fitn, label="Maximum Fitness")
    ax_fit.plot(_generations, _avg_fitn, label="Average Fitness")
    ax_fit.plot(_generations, _min_fitn, label="Minimum Fitness")
    ax_fit.set_ylabel("Fitness (after scaling)")
    ax_fit.legend(loc=(0.2, 0.2))
    plt.draw()
    plt.show()
    

def make_dataset_stepper(model):
    def step_dataset(ga_engine):
        model.dataset.next_train_periods()
    return step_dataset

def step_generation(ga_engine):
    for callback in _step_callbacks:
        callback(ga_engine)

def _add_if_new(parser, option, **kwargs):
    if not parser.has_option(option):
        parser.add_option(option, **kwargs)

def ga_options(parser=None):
    """Add GA-related options to the parser. If no parser is provided, one
    will be created."""
    if parser is None:
            parser = optparse.OptionParser()
    _add_if_new(parser, "--pop-size", metavar="p", dest="pop_size", type="int", help="Population size", default=6)
    _add_if_new(parser, "--generations", metavar="g", dest="generations", type="int", help="Number of generations", default=4)
    _add_if_new(parser, "--termination", dest="term_fitness", type="float", help="Termination fitness", default=0)
    _add_if_new(parser, "--elite", dest="elite", type="int", help="Number of elite individuals", default=1)
    _add_if_new(parser, "--crossover", dest="crossover", type="float", help="Crossover rate", default=0.4)
    _add_if_new(parser, "--mutation", dest="mutation", type="float", help="Mutation rate", default=0.025)
    #_add_if_new(parser, "--mutation-mu", dest="mutation_mu", type="float", help="Mean of Gaussian mutation", default=0)
    #_add_if_new(parser, "--mutation-sigma", dest="mutation_sigma", type="float", help="Std dev of Gaussian mutation", default=1)
    _add_if_new(parser, "--num-trials", dest="num_trials", type="int", help="Number of trials (with different data and/or initial conditions) for each individual per fitness evaluation", default=5)
    _add_if_new(parser, "--parallel", dest="parallel", action="store_true", help="Enable parallel fitness evaluation", default=False)
    _add_if_new(parser, "--print-pop", dest="print_pop", action="store_true", help="Print the entire population at each generation (score, fitness, genes)", default=False)
    _add_if_new(parser, "--live-plot", dest="live_plot", action="store_true", help="Plot fitness per generation as the GA runs", default=False)
    _add_if_new(parser, "--gui", dest="gui", action="store_true", help="Show GUI while the GA runs", default=False)
    _add_if_new(parser, "--MPI", dest="MPI", action="store_true", help="Distribute evolution over MPI", default=False)
    _add_if_new(parser, "--initial-population", dest="initial_population", help="Path to file from which genes of initial population are read", default=None)
    return parser

def is_mpi_slave(options):
    if options.MPI:
        from mpi4py import MPI
        return MPI.COMM_WORLD.Get_rank() != 0
    else:
        return False

def _print_mkl_info():
    import os
    try:
        print "MKL_NUM_THREADS:", os.environ["MKL_NUM_THREADS"]
    except:
        print "not set"

def _print_mpi_info():
    import os
    try:
        with open(os.environ["PBS_NODEFILE"], "r") as f:
            units = [line[:-1] for line in f]
            print "Total %d PBS computation units, on %d nodes." \
              % (len(units), len(set(units)))
            #print units
    except:
        print "Failed to open PBS_NODEFILE for reading."
            
def run_GA(model, options=None):
    """Runs the genetic algorithm. See function ga_options for the contents of
    the options variable."""
    global _model
    _model = model
    genome = pu.make_alleles_genome(model.genes)
    genome.evaluator.set(_fitness)
    genome.setParams(bestrawscore=options.term_fitness,
                     num_trials=options.num_trials,
                     initial_population=options.initial_population)

    if options.MPI:
        from sg.utils.pyevolve_mpi import SimpleMPIGA
        ga = SimpleMPIGA(model, genome, seed=options.seed)
        if not is_mpi_slave(options):
            print "MPI-distributed evolution with %d hosts." % ga.nhosts
            _print_mpi_info()
    else:
        ga = pu.SimpleGAWithFixedElitism(genome, seed=options.seed)
    if not is_mpi_slave(options):
        _print_mkl_info()
    ga.setGenerations(options.generations)
    _step_callbacks.append(make_dataset_stepper(model))
    ga.stepCallback.set(step_generation)
    # Number of generations and dataset stepper must be set before starting eval_loop
    if is_mpi_slave(options):
        ga.eval_loop()
        return
    if options.initial_population is None:
        ga.setPopulationSize(options.pop_size)
    ga.setMutationRate(options.mutation)
    ga.setCrossoverRate(options.crossover)
    if options.elite > 0:
        ga.setElitism(True)
        ga.setElitismReplacement(options.elite)
    else:
        ga.setElitism(False)
    # DO NOT USE TOURNAMENT SELECTION WITH MINIMIZATION PROBLEMS (as of version
    # 0.6rc1 of Pyevolve), it selects max fitness irrespective of minimax mode.
    #ga.selector.set(Selectors.GTournamentSelector)
    ga.selector.set(Selectors.GRouletteWheel)
    
    #ga.terminationCriteria.set(GSimpleGA.RawScoreCriteria)
#    ga.getPopulation().scaleMethod.set(Scaling.SigmaTruncScaling)
    pop = ga.getPopulation()
    #pop.scaleMethod.set(pu.BoltzmannScalingFixed)
    pop.scaleMethod.set(pu.ExponentialScaler(options.generations))
    boltz_start = pop.getParam("boltz_temperature", Consts.CDefScaleBoltzStart)
    boltz_min = pop.getParam("boltz_min", Consts.CDefScaleBoltzMinTemp)
    pop.setParams(boltz_factor=(boltz_start - boltz_min) / (0.8 * options.generations))
    
    ga.setMinimax(Consts.minimaxType["maximize"])

    print "Evolution settings:"
    print "\tNumber of training sequences: %d" % options.num_trials 
    print "\tStart days of training sequences:", model.dataset.train_periods_desc
    print "\tTermination fitness: %f" % options.term_fitness
    print "\tRandom seed: %d" % options.seed
    print "\tPopulation size: %d" % ga.getPopulation().popSize
    print "\tNumber of generations: %d" % ga.getGenerations()
    print "\tNumber of elite indivs: %d" % options.elite
    print "\tCrossover rate: %f" % options.crossover
    # print "\tMutation rate (mu, sigma): %f (%f, %f)" % \
    #   (options.mutation, options.mutation_mu, options.mutation_sigma)
    print "\tMutation rate: %f" % options.mutation
    print "\tSelection mechanism(s): ", ga.selector[:]
    
    dbpath = sg.utils.get_path(options, "pyevolve", "db")
    sqlite_adapter = DBAdapters.DBSQLite(dbname=dbpath, identify="ex1", 
                                         resetDB=True, commit_freq=1)
    ga.setDBAdapter(sqlite_adapter)
    if options.parallel:
        ga.setMultiProcessing(True)
    if options.print_pop:
        _step_callbacks.append(print_population)
    if options.live_plot:
        global _live_fig
        _step_callbacks.append(plot_fitnesses)
        _live_fig = plt.figure()
        plt.ion()
    if options.gui:
        _step_callbacks.append(update_gui)
    _step_callbacks.append(print_best_genome)
    _step_callbacks.append(report_time_spent)
    ga.evolve(freq_stats=1)

    model.genome = ga.bestIndividual()
    return
