"""Initiates models, runs them through a genetic algorithm to find the
optimal parameters, and tests the models in a production setting."""

import random
from datetime import timedelta as dt
import itertools as it
import math
import sys
import optparse
import os
import time
import cPickle as pickle
import multiprocessing as mp
from functools import partial
from dateutil.relativedelta import relativedelta

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import scipy
import pandas as pd
import Oger

# Program options as global variable, to avoid passing all over the place.
options = None

import sg.utils
from sg.utils import concat_and_calc_error, plot_target_predictions
import sg.utils.pyevolve_utils as pu
from sg.utils.timer import SimpleTimer
from sg.utils import get_path
import sg.data.sintef.userloads as ul
import sg.data.bchydro as bc
import sg.data.gefcom2012 as gef12
import sg.data.gefcom2014 as gef14
import pattern_eliminators
from model import Model
from ga import run_GA, ga_options, is_mpi_slave, process_data
import esn
import sg.utils.pyevolve_utils as pu
import load_cleansing

def float_err_handler(type, flag):
    msg = "Floating point error (%s), with flag %s" % (type, flag)
    print >>sys.stderr, "==================== ERROR ====================="
    print >>sys.stderr, msg
    print >>sys.stderr, "==================== ERROR ====================="
    sys.stderr.flush()
    raise RuntimeError(msg)

class SingleUserDataset(object):
    def __init__(self, max_input_length):
        """Set up training and test data sets. max_input_length should
        correspond to the highest allowed value for the hindsight genome, or in
        case of R-tree, to the entire dataset."""
        if options.data_seed is None: # Control data seed by "global" random seed
            self._rng = np.random.RandomState(random.randrange(1, 2**16))
        else:
            self._rng = np.random.RandomState(options.data_seed)
        self.user_id = self._get_userid() # Slightly funky, since the other datasets inherit this
        (raw_train, raw_test) = self._prepare_dataset()
        # if options.contaminate:
        #     raw_train = self._contaminate_signal(raw_train)
        #     raw_test = self._contaminate_signal(raw_test)
        self.num_predictions = options.num_predictions        
        input_length = max_input_length + dt(hours=self.num_predictions*2)
        self._train_data = sg.data.Dataset(raw_train, input_length, dt(hours=24))
        self._test_data = sg.data.Dataset(raw_test, input_length, dt(hours=24))
        self._train_periods = self._rng.randint(0, self.num_train_periods,
                                               options.num_trials)
        self._env_replace = options.env_replace

    def train_data_iterator(self):
        """Returns an iterator that, when called, returns a tuple containing
        (training data input, correct output) for each training period. The
        input includes a 'num_predictions'-hour weather "forecast"."""
        def iterator():
            for period in self._train_periods:
                yield self.get_train_period(period)
        return iterator

    def test_data_iterator(self):
        """Returns an iterator that, when called, returns a tuple containing
        (test data input, correct output) for each day of the test set. The
        input includes a 'num_predictions'-hour weather "forecast"."""
        def iterator():
            for period in range(self._test_data.num_periods):
                yield self.get_test_period(period)
        return iterator

    def _get_period(self, period, dataset):
        data = dataset.get_period(period).copy()
        return self._split_input_output(data)

    def get_train_period(self, period):
        """Return the training instance with index 'period'."""
        return self._get_period(period, self._train_data)

    def get_test_period(self, period):
        """Return the training instance with index 'period'."""
        return self._get_period(period, self._test_data)

    def _split_input_output(self, data):
        """Prepare a dataset containing 'Load' and 'Temperature'. Move the last
        self.num_predictions timesteps of the load to a separate series. Return
        (data meant for input, correct output)."""
        data_out = data['Load'][-self.num_predictions:].copy()
        data['Load'][-self.num_predictions:] = np.nan
        return (data, data_out)

    def _prepare_dataset(self):
        """Return datasets for the experiment periods containing loads and
        temperatures for the given dataset."""
        loads = ul.tempfeeder_exp()[self.user_id]
        return [ul.add_temperatures(loads, period) 
                for period in ul.experiment_periods()]

    def _get_userid(self):
        if options.userid is None:
            # rng.choice introduced in Numpy 1.7.0
            return self._rng.permutation(ul.tempfeeder_exp().user_ids)[0]
        return options.userid

    def add_genes(self, alleles, loci_list):
        """Add dataset-specific genes to the genome and corresponding list of
        loci, when applicable."""
        pass

    def get_pre_post_processors(self):
        """Return dataset-specific pre- and post-processing functions, when
        applicable. Typical usage is to interpret a dataset-specific
        genome and transform the data accordingly."""
        return ([], [])

    @property
    def desc(self):
        """A short description of the data set."""
        return "single_user_id_%s" % str(self.user_id)

    def next_train_periods(self):
        """Update the start dates for training data."""
        num_replace = min(self._env_replace, len(self._train_periods))
        which_replace = self._rng.choice(np.arange(options.num_trials),
                                         size=num_replace, replace=False)
        self._train_periods[which_replace] = \
            self._rng.choice(np.arange(self._train_data.num_periods),
                             size=num_replace, replace=False)
        
    @property
    def train_periods_desc(self):
        """A description of the selected training data."""
        return str(self._train_periods) + ", replace " + \
          str(self._env_replace) + " each gen."

    @property
    def train_data(self):
        """Returns the entire training data set."""
        return self._train_data.series

    @property
    def test_data(self):
        """Returns the entire test data set."""
        return self._test_data.series

    @property
    def num_train_periods(self):
        return self._train_data.num_periods
    
    @property
    def num_test_periods(self):
        return self._test_data.num_periods


class TotalLoadDataset(SingleUserDataset):
    def _prepare_dataset(self):
        """Return datasets for the experiment periods containing the total load
        for all users in the experiment."""
        loads = pd.concat(ul.total_experiment_load())
        return [ul.add_temperatures(loads, period) 
                for period in ul.experiment_periods()]

    @property
    def desc(self):
        """A short description of the data set (e.g. user ID or "total")."""
        return "total_load"

class UserSubsetDataset(SingleUserDataset):
    def __init__(self, max_input_length):
        self._subset_size = options.user_subset
        self._subset_seed = options.data_seed
        SingleUserDataset.__init__(self, max_input_length)

    def _prepare_dataset(self):
        """Return datasets for the experiment periods containing the total load
        for a subset of users in the experiment."""
        loads = pd.concat(ul.mean_experiment_load_for_user_subset(self._subset_size,
                                                             self._subset_seed))
        return [ul.add_temperatures(loads, period) 
                for period in ul.experiment_periods()]

    @property
    def desc(self):
        """A short description of the data set (e.g. user ID or "total")."""
        return "subset_load"

class BCHydroDataset(SingleUserDataset):
    def _prepare_dataset(self):
        """Return datasets for the experiment periods containing the load and
        temperature for British Columbia."""
        data = pd.DataFrame({'Load': self._get_load(), 'Temperature': self._get_temperature()})
        test_period_starts = "2010-03-15 00:00:00"
        # Current version (0.14.0) of Pandas includes the end point when
        # slicing a range with dates, so drop the last element of the training period.
        (train, test) = (data[:test_period_starts][:-1], data[test_period_starts:])
        return (train, test)

    def _get_load(self):
        print "Removing zeros from BC dataset."
        return bc.load_remove_zeros()
        #return bc.load()

    def _get_temperature(self):
        return bc.temperature()

    @property
    def desc(self):
        """A short description of the data set."""
        return "bc_hydro_with_temperatures"

class BCHydroNoHolidaysDataset(BCHydroDataset):
    def _get_load(self):
        print "Removing zeros and holidays from BC dataset."
        data = bc.load_remove_zeros()
        bc.remove_holidays(data)
        return data
    
    @property
    def desc(self):
        """A short description of the data set."""
        return "bc-hydro-with-temperatures-no-zeros-no-holidays"

    
class GEFCOMSolvedNonNanMeanTempDataset(SingleUserDataset):
    def _prepare_dataset(self):
        """Return datasets for the experiment periods containing the total load and
        mean temperature from the GEFCOM 2012 dataset."""
        data = self._get_data()
        test_period_starts = data.index[-1] - relativedelta(years=1)
        (train, test) = (data[:test_period_starts], data[test_period_starts:])
        return (train, test)

    def _get_data(self):
        return pd.concat([self._get_load(), self._get_temp()], join='inner', axis=1)

    def _get_load(self):
        if options.gef_zone is None:
            load = gef12.load()
        else:
            load = gef12.load_solved_non_nan().load['zone_{}'.format(options.gef_zone)]
        return pd.DataFrame({'Load': load})
            
    def _get_temp(self):
        return pd.DataFrame({'Temperature': gef12.temperature()})
    
    @property
    def desc(self):
        """A short description of the data set."""
        if options.gef_zone is None:
            return "GEFCOM 2012 data with total load and mean temperatures."
        else:
            return "GEFCOM 2012 data for zone {} and mean temperatures.".format(options.gef_zone)

class GEFCOMSolvedNonNanMultiTempDataset(GEFCOMSolvedNonNanMeanTempDataset):
    def __init__(self, *args):
        self._temp = gef12.load_solved_non_nan().temp
        self._temp_columns = self._temp.columns
        GEFCOMSolvedNonNanMeanTempDataset.__init__(self, *args)
        
    def _get_temp(self):
        return self._temp
    
    @property
    def desc(self):
        if options.gef_zone is None:
            return "GEFCOM 2012 data with total load and separate " \
                "temperatures from each station."
        else:
            return "GEFCOM 2012 data for zone {} and separate " \
                "temperatures from each station.".format(options.gef_zone)

    def add_genes(self, alleles, loci_list):
        alleles.add(pu.make_real_gene(len(self._temp.columns), 0, 1, .1), weight=1)
        loci_list += ['temp_weights']

    def get_pre_post_processors(self):
        def gene_parser(data_now, genome, loci, prediction_steps):
            weights = np.array(genome[loci.temp_weights])
            temp = data_now[self._temp_columns] * weights
            temp = temp.sum(axis=1) / weights.sum()
            return pd.DataFrame({'Load': data_now['Load'], 'Temperature':temp})
        return ([gene_parser], [])

class GEF14Dataset(SingleUserDataset):
    def __init__(self, *args, **kwargs):
        self._temp_columns = gef14.temperature_columns()
        SingleUserDataset.__init__(self, *args, **kwargs)
    
    def _prepare_dataset(self):
        """Return datasets containing the load and temperatures for task 1 of
        the GEFCOM 2014 dataset.

        """
        data = gef14.load(task_no=1)
        data = data.iloc[np.where(np.logical_not(np.isnan(data['Load'])))[0]]
        test_period_starts = data.index[-1] - relativedelta(years=1)
        (train, test) = (data[:test_period_starts], data[test_period_starts:])
        return (train, test)

    @property
    def desc(self):
        """A short description of the data set."""
        return "GEFCOM 2014 data with total load and separate " \
               "temperatures from each station."

    def add_genes(self, alleles, loci_list):
        alleles.add(pu.make_real_gene(len(self._temp_columns), 0, 1, .1), weight=1)
        loci_list += ['temp_weights']

    def get_pre_post_processors(self):
        def gene_parser(data_now, genome, loci, prediction_steps):
            data = data_now[self._temp_columns]
            weights = np.array(genome[loci.temp_weights])
            temp = sg.utils.normalized_weighted_mean(data, weights)
            return pd.DataFrame({'Load': data_now['Load'], 'Temperature':temp})
        return ([gene_parser], [])


class ModelCreator(object):
    def __init__(self, options):
        self._hindsight_days = [1, 3, 7, 7*2, 7*4, 7*4*2, 7*4*3]
        # print >>sys.stderr, "Running with long hindsights only."
        # self._hindsight_days = [14, 28]
        self._max_hindsight_hours = self._hindsight_days[-1] * 24
        self._options = options
        max_input_length = dt(hours=self._max_hindsight_hours)
        if options.remove_holidays and not options.bc_data:
            raise RuntimeError("'remove-holidays' option only supported on BC Hydro data.")
        if options.gef_data:
            self._data = GEFCOMSolvedNonNanMeanTempDataset(max_input_length)
        elif options.gef_temp_data:
            self._data = GEFCOMSolvedNonNanMultiTempDataset(max_input_length)
        elif options.gef14_data:
            self._data = GEF14Dataset(max_input_length)
        elif options.bc_data:
            if options.remove_holidays:
                print "Removing holidays from dataset"
                self._data = BCHydroNoHolidaysDataset(max_input_length)
            else:
                self._data = BCHydroDataset(max_input_length)
        elif options.total_load:
            self._data = TotalLoadDataset(max_input_length)
        elif options.user_subset:
            self._data = UserSubsetDataset(max_input_length)
        else:
            self._data = SingleUserDataset(max_input_length)

    def _get_error_func(self):
        """Return error function (pointer) based on program options."""
        return eval(self._options.error_func)

    def get_model(self):
        """This is where the models are defined. The models are passed to the
        GA engine for evolution of the optimal set of parameters. Afterwards,
        the models are tested, and performance is measured."""
        self._alleles = pu.AllelesWithOperators()
        self._loci_list = []
        self._add_hindsight_genes()
        self._add_transform_genes()
        self._add_dataset_genes()
        self._add_cleaning_genes()
        model = Model(self.__class__.__name__, 
                      genes=self._alleles, 
                      error_func=self._get_error_func(),
                      transformer=self._get_transform(),
                      loci=sg.utils.Enum(*self._loci_list))
        model.dataset = self._data
        model.day = options.num_predictions
        model.preprocessors, model.postprocessors \
            = self._get_pre_post_processors()
        return model

    def _add_hindsight_genes(self):
        gene = pu.make_choice_gene(1, [i*24 for i in self._hindsight_days])
        self._alleles.add(gene, weight=1)
        self._loci_list += ['hindsight']
    
    def _add_cleaning_genes(self):
        def _smoothlist():
            return np.concatenate((np.arange(0.001, 0.01, 0.001),
                                   np.arange(0.01, 0.1, 0.01),
                                   np.arange(0.1, 1, 0.1),
                                   np.arange(1, 10, 1),
                                   np.arange(10, 100, 10),
                                   np.arange(100, 1000, 100)))
        if not self._options.no_cleaning:
            self._alleles.add(pu.make_choice_gene(1, _smoothlist()), weight=1)
            self._alleles.add(pu.make_choice_gene(1, _smoothlist()), weight=1)
            self._alleles.add(pu.make_choice_gene(1, np.arange(0.1, 3.05, 0.1)), weight=1)
            self._alleles.add(pu.make_choice_gene(1, np.arange(0.1, 3.05, 0.1)), weight=1)
            self._loci_list += ['t_smooth', 'l_smooth', 't_zscore', 'l_zscore']

    def _add_dataset_genes(self):
        self._data.add_genes(self._alleles, self._loci_list)

    def _pproc_append(self, pre, post, new_pre, new_post):
        pre.append(new_pre)
        post.append(new_post)
        
    def _get_pre_post_processors(self):
        pre = []
        post = []
        data_pre, data_post = self._data.get_pre_post_processors()
        pre += data_pre
        post += data_post
        # Make sure pre-/postprocessing happens in the right order
        # (e.g. difference before standardization)! Append to the end of
        # both lists; postprocessing operates on reversed(post).
        if not options.no_cleaning:
            if options.fast_cleaning:
                pre.append(load_cleansing.bspline_clean_dataset_fast)
            else:
                pre.append(load_cleansing.bspline_clean_dataset)
        if options.subtract_daily_pattern:
            self._pproc_append(pre, post,
                               *pattern_eliminators.make_daily_pattern_eliminator(self._data))
        if options.subtract_weekly_pattern:
            self._pproc_append(pre, post,
                               *pattern_eliminators.make_weekly_pattern_eliminator(self._data))
        if options.subtract_mixed_pattern:
            self._pproc_append(pre, post,
                               *pattern_eliminators.make_mixed_pattern_eliminator(self._data))
        for i in range(options.difference):
            self._pproc_append(pre, post,
                               *pattern_eliminators.make_diff_pipeliner(self._data))
        if options.standardize:
            self._pproc_append(pre, post,
                               *pattern_eliminators.make_standardizer_pipeliner(self._data))
        return (pre, post)

def data_options(parser=None):
    """Add data-related options to the parser. If no parser is provided, one
    will be created."""
    # optparse is deprecated and should be updated to argparse.
    if parser is None:
            parser = optparse.OptionParser()
    parser.add_option("--userid", dest="userid", type="long", help="User/meter ID", default=None)
    #parser.add_option("--random-user", dest="random_user", action="store_true", help="Ignore the userid argument, pick a random user ID instead", default=False)
    parser.add_option("--user-subset", dest="user_subset", type="long", help="Calculate the total load for a random subset of users. Argument gives the number of users in the subset", default=1)
    parser.add_option("--total-load", dest="total_load", action="store_true", help="Use total load for all meters", default=False)
    parser.add_option("--bc-data", dest="bc_data", action="store_true", help="Use BC Hydro data set with constant temperature", default=False)
    parser.add_option("--gef-data", dest="gef_data", action="store_true", help="Use GEFCOM2012 data set with total load and mean temperature", default=False)
    parser.add_option("--gef-temp-data", dest="gef_temp_data", action="store_true", help="Use GEFCOM2012 data set with total load and all temperatures", default=False)
    parser.add_option("--gef14-data", dest="gef14_data", action="store_true", help="Use GEFCOM2014 data set with multiple temperatures", default=False)
    parser.add_option("--gef-zone", dest="gef_zone", type="long", help="Choose a specific zone (1-20) from the GEFCOM2012 data set (use in combination with gef-data or gef-temp-data", default=None)
    parser.add_option("--remove-holidays", dest="remove_holidays", action="store_true", help="Remove holidays from the dataset.", default=False)
    parser.add_option("--data-seed", dest="data_seed", type="int", help="Random seed used for selecting training periods", default=None)
    parser.add_option("--env-replace", dest="env_replace", type="int", help="Number of training periods (environments) to replace at each generation", default=0)
    parser.add_option('--difference', dest='difference', type='int', help='Difference the data the given number of times', default=0)
    parser.add_option('--standardize', dest='standardize', action='store_true', help='Standardize the data', default=False)
    return parser

def prediction_options(parser=None):
    """Add prediction-related options to the parser. If no parser is provided, one
    will be created."""
    if parser is None:
            parser = optparse.OptionParser()
    parser.add_option("--seed", dest="seed", type="int", help="Evolution random seed", default=random.randrange(1, 2**16))
    parser.add_option("--out-dir", dest="out_dir", help="Output directory for log files etc", default=".")
    parser.add_option("--out-postfix", dest="out_postfix", help="Postfix for log files etc", default=str(os.getpid()))
    parser.add_option("--save-plot", dest="save_plot", action="store_true", help="Save the plot of testset prediction to PDF", default=False)
    parser.add_option("--no-show-plot", dest="no_show_plot", action="store_true", help="Create PDF plot of testset prediction, but don't show on screen", default=False)
    parser.add_option("--no-plot", dest="no_plot", action="store_true", help="Do not create plots whatsoever (suitable for running simulations without display)", default=False)
    parser.add_option("--no-cleaning", dest="no_cleaning", action="store_true", help="Disable smoothing and cleaning of input data (deprecated, use with-cleaning instead)", default=True)
    parser.add_option("--with-cleaning", dest="no_cleaning", action="store_false", help="Enable smoothing and cleaning of input data", default=True)
    parser.add_option("--fast-cleaning", dest="fast_cleaning", action="store_true", help="Enable fast (C++/MKL) cleaning of input data", default=False)
    parser.add_option("--subtract-daily-pattern", dest="subtract_daily_pattern", action="store_true", help="Subtract daily pattern from input data before cleaning/predicting", default=False)
    parser.add_option("--subtract-weekly-pattern", dest="subtract_weekly_pattern", action="store_true", help="Subtract weekly pattern from input data before cleaning/predicting", default=False)
    parser.add_option("--subtract-mixed-pattern", dest="subtract_mixed_pattern", action="store_true", help="Subtract weekly pattern from load, daily pattern from temperature", default=False)
    parser.add_option("--num-predictions", dest="num_predictions", type="int", help="Number of time steps to predict into the future", default=24)
    parser.add_option("--error-func", dest="error_func", type="string", help="Error function to be used, e.g. Oger.utils.rmse, sg.utils.relative_rmse, sg.utils.mape/mape_plus_one/mape_skip_zeros", default="Oger.utils.rmse")
    return parser

def get_options():
    global options
    parser = prediction_options()
    parser = ga_options(parser)
    parser = data_options(parser)
    (options, args) = parser.parse_args()
    
def test_genome(genome, model):
    """Run genome on test data. Return target values and a list of
    predictions."""
    loci = model.loci
    target = pd.TimeSeries()
    predictions = []
    test_iter = model.dataset.test_data_iterator()
    test_number = 1
    for (data_in, data_out) in test_iter():
        model_out = process_data(data_in, genome, model, loci)
        error = model.error_func(model_out.values, data_out.values)
        print "Error for test at %s: %f" % (str(model_out.index[0]), error)
        test_number += 1
        target = target.combine_first(data_out)
        predictions.append(model_out)
    return target, predictions

        
def parallel_test_genome(genome, model):
    """Run genome on test data. Return target values and a list of
    predictions. Runs in parallel."""
    loci = model.loci
    target = pd.TimeSeries()
    predictions = []
    test_iter = model.dataset.test_data_iterator()

    pool = mp.Pool()
    partial_process_data = partial(process_data, genome = genome, model = model, loci = loci)
    data_in, data_out = zip(*test_iter())
    performance = pool.map(partial_process_data, data_in)
    pool.close()
    performance.sort(key=lambda x: x.index[0])

    for data_out, model_out in zip(data_out, performance):
        error = model.error_func(model_out.values, data_out.values)
        print "Error for test at %s: %f" % (str(model_out.index[0]), error)
        target = target.combine_first(data_out)
        predictions.append(model_out)
        
    return target, predictions


def _save_test_prediction(target, 
                          predictions, data_desc):
    path = get_path(options, "test_series_%s" % data_desc, "pickle")
    with open(path, "a+") as f:
        pickle.dump((target, predictions), f)

def plot_test_prediction(target, predictions, 
                         model_name, data_desc, error_func):
    save_plot = options.save_plot
    if save_plot:
        pp_path = get_path(options, "prediction_%s" % data_desc, "pdf")
        pp = PdfPages(pp_path)
    error = concat_and_calc_error(predictions, target, error_func)
    if np.any(target == 0):
        mapestr = "no MAPE due to 0's in observations"
    else:
        mapestr = 'MAPE: %3.2f' % concat_and_calc_error(predictions, target, 
                                              sg.utils.mape)
    title = 'Test data prediction, %s, %s, %i days, model error: ' \
      '%3.2f, RMSE: %3.2f, %s' % \
      (model_name, data_desc, len(predictions), error, 
       concat_and_calc_error(predictions, target, Oger.utils.rmse), mapestr)
    print "%s." % title
    plt.figure()
    plt.title(title)
    plot_target_predictions(target, predictions)
    if save_plot:
        pp.savefig()
        pp.close()
    if not options.no_show_plot:
        plt.ioff() # In case --live-plot is chosen, since that sets ion().
        plt.show()

def _run_models(models, dataset):
    data_desc = dataset.desc
    for model in models:
        if not is_mpi_slave(options):
            print "Optimizing parameters for model", model.name, "."
        run_GA(model, options)
        if is_mpi_slave(options):
            continue
        raw_genes = pu.raw_genes(model.genome, strip=True)
        print "Best genes found during evolution: ", raw_genes
        target, predictions = test_genome(raw_genes, model)
        _save_test_prediction(target, predictions, data_desc)
        error = concat_and_calc_error(predictions, target, model.error_func)
        print "Error on test phase for best genome found, " \
          "%s, %i days: %5.4f" % (data_desc, len(predictions), error)
        if not options.no_plot:
            plot_test_prediction(target, predictions, 
                                 model.name, data_desc, model.error_func)

def _print_sim_context(dataset):
    """Print some info about the context in which the simulatotion was
    performed."""
    print "Command line was:", " ".join(sys.argv)
    print "Process ID is: %d" % os.getpid()
    sys.stdout.write("hg identify: ")
    sys.stdout.flush()
    os.system("hg identify")
    sys.stdout.write("hg status: ")
    sys.stdout.flush()
    os.system("hg status")
    print "\nUsing random seed %d" % options.seed
    print "Data description:", dataset.desc


def run(model_creator_class): 
    """Main entry point for specific models. model_creator is an instance of a
    class used to set up the model and the data."""
    get_options()
    if not is_mpi_slave(options):
        timer = SimpleTimer()
    prev_handler = np.seterrcall(float_err_handler)
    prev_err = np.seterr(all='call')
    np.seterr(under='ignore')
    random.seed(options.seed)
    np.random.seed(options.seed)
    model_creator = model_creator_class(options)
    model = model_creator.get_model()
    if not is_mpi_slave(options):
        _print_sim_context(model.dataset)
    _run_models([model], model.dataset)
    ul.tempfeeder_exp().close()
    
if __name__ == "__main__":
    print "You should run one of the model-specific programs (e.g. "\
      "load_prediction_esn) instead of this."

