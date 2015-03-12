"""Evolve a load predictor with BSpline data cleansing and a CBR retrieval mechanism."""

import random
from datetime import timedelta as dt

import numpy as np
from pyevolve import GAllele
import Oger
import pandas as pd

import sg.utils
import sg.utils.pyevolve_utils as pu
from model import Model
import wavelet
import load_cleansing
import load_prediction
from sg.data.sintef.create_full_temp_data import data as read_temperatures
import sg.data.sintef.userloads as ul
import sg.data.bchydro as bc

def _dummy_cleaner(d,g,l,p,m):
    return d

class RTreeDataset(object):
    def __init__(self, options, max_input_length):
        """Set up training and test data sets."""
        if options.data_seed is None: # Control data seed by "global" random seed
            self._rng = np.random.RandomState(random.randrange(1, 2**16))
        else:
            self._rng = np.random.RandomState(options.data_seed)
        self.user_id = self._get_userid(options)
        self._train_data, self._test_data = self._prepare_dataset()
        self.input_length = max_input_length + 2
        self._train_periods = self._rng.randint(self.input_length, 
                                               len(self._train_data)/24-1, 
                                               options.num_trials)
        self._env_replace = options.env_replace
        print "Data description:", self.desc

    def train_data_iterator(self):
        """Returns an iterator that, when called, returns a tuple containing
        (training data input, correct output) for each training period. The
        input includes a 24-hour weather "forecast"."""
        def iterator():
            for period in self._train_periods:
                data = self._train_data.copy()
                yield self._split_input_output(data, period*24)
        return iterator

    def test_data_iterator(self):
        """Returns an iterator that, when called, returns a tuple containing
        (test data input, correct output) for each day of the test set. The
        input includes a 24-hour weather "forecast"."""
        def iterator():
            for period in range(24, len(self._test_data), 24):
                data = pd.concat((self._train_data, self._test_data[:period])).copy()
                splitted = self._split_input_output(data, len(data)-24)
                yield splitted
        return iterator

    def _split_input_output(self, data, test_index, predict_steps=24):
        """Prepare a dataset containing 'Load' and 'Temperature'. Move the last
        predict_steps timesteps of the load to a separate series. Return (data
        meant for input, correct output)."""
        data_out = data['Load'][test_index:test_index+predict_steps].copy()
        data['Load'][test_index:test_index+predict_steps] = np.nan
        return (data, data_out)

    def _join_temp_and_load(self, temps, loads, period):
        """Given temp and load time series and a period, return a
        two-dimensional array containing temps and loads for the given
        period."""
        # Temperature and load readings may start and end at different times.
        (l, t) = loads.align(temps, join="inner", axis=0)
        frame = pd.concat((t, l['Load']), axis=1)
        frame = frame.rename(columns={0:"Temperature", 1:"Load"})
        return frame[period[0]:period[1]]

    def _prepare_dataset(self):
        """Return datasets for the experiment periods containing loads and
        temperatures for the given user, or for all users in the experiment if
        total_load is True."""
        temps = read_temperatures()
        loads = ul.tempfeeder_exp()[self.user_id]
        return [self._join_temp_and_load(temps, loads, period) 
                for period in ul.experiment_periods()]

    def _get_userid(self, options):
        if options.userid is None:
            return self._rng.permutation(ul.tempfeeder_exp().user_ids)[0]
        return options.userid

    @property
    def desc(self):
        """A short description of the data set."""
        return "rtree_user_%d" % self.user_id

    @property
    def train_data(self):
        """Returns the entire training data set."""
        return self._train_data

    @property
    def test_data(self):
        """Returns the entire test data set."""
        return self._test_data

    def next_train_periods(self):
        """Update the start dates for training data."""
        tp = self._train_periods
        self._rng.shuffle(tp)
        num_replace = min(self._env_replace, len(tp))
        # Slice with [:len(tp)-num_replace], not just [:-num_replace], this
        # fails when num_replace==0
        tp = np.concatenate((tp[:len(tp)-num_replace],
                             self._rng.randint(self.input_length, 
                                              len(self._train_data)/24-1, 
                                              num_replace)))
        self._train_periods = tp

    @property
    def train_periods_desc(self):
        """A description of the selected training data."""
        return str(self._train_periods) + ", replace " + \
          str(self._env_replace) + " each gen."

class RTreeTotalDataset(RTreeDataset):
    def _prepare_dataset(self):
        """Return datasets for the experiment periods containing the total load
        for all users in the experiment."""
        print "Using total load rather than per-user load."
        temps = read_temperatures()
        loads = pd.concat(ul.total_experiment_load())
        return [self._join_temp_and_load(temps, loads, period) 
                for period in ul.experiment_periods()]

    @property
    def desc(self):
        """A short description of the data set."""
        return "rtree_total_load"

class RTreeBCHydroDataset(RTreeDataset):
    def _prepare_dataset(self):
        """Return datasets for the experiment periods containing the total load
        for all users in the experiment."""
        load = bc.load()
        data = pd.DataFrame({'Load': load, 'Temperature': 0})
        test_period_starts = "2010-03-15 00:00:00"
        (train, test) = (data[:test_period_starts], data[test_period_starts:])
        return (train, test)

    @property
    def desc(self):
        """A short description of the data set."""
        return "rtree_bc_hydro_no_temperatures"

class WaveletCBRModelCreator(load_prediction.ModelCreator):
    def get_dataset(self, options):
        """This function should create the an instance of a dataset class
        according to the selected model and user options."""
        if options.bc_data:
            return RTreeBCHydroDataset(options, 31)
        elif options.total_load:
            return RTreeTotalDataset(options, 31)
        else:
            return RTreeDataset(options, 31)
    
    def _add_transform_genes(self):
        self._alleles.add(pu.make_int_gene(1, 2, 11, 1)) # Dimension
        # self._alleles.add(pu.make_real_gene(1, 0, 1, 0.05)) # Weight
        # self._alleles.add(pu.make_int_gene(1, 1024, 2048, 50)) # Boolean mask
        self._loci_list += ['dimension'] #, 'weight', 'mask', 
        if not self._options.no_cleaning:
            #wavelet.set_local_cleaning_func(self._get_cleansing_function(options), model)
            raise NotImplementedError("CBR with cleaning not updated to work with pipeline model for cleaning.")

    def _get_transform(self):
        return wavelet.retrieve

if __name__ == "__main__":
    load_prediction.run(WaveletCBRModelCreator)
