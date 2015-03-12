"""Code for experiments in 'Error functions' paper."""

from dateutil.relativedelta import relativedelta
import random
import itertools
import copy
import argparse

import pandas as pd
import numpy as np

import sg.data.bchydro as bchydro
import sg.utils
import pattern_eliminators as pe
import ga
import arima
import regul_ar
import load_prediction_regul_ar as lpra
import load_prediction as lp

def _process_data(model, data_in, genome, loci, prediction_steps=24):
    data_now = copy.copy(data_in)
    for preproc in model.preprocessors:
        data_now = preproc(data_now, genome, loci, prediction_steps)
    model_out = model.transformer(
        data_now, genome, loci, prediction_steps)
    for postproc in reversed(model.postprocessors):
        model_out = postproc(model_out, genome, loci)
    return model_out


class RegularizedARGridSearcher(object):
    def __init__(self, model):
        self._model = model
        self._N = 100 # Number of prediction days
        #num_lambdas = 50
        #self._lambda_conts = np.concatenate(([0], np.logspace(-3, 6, num_lambdas)))
        # _bc_mean_single_error = 0.2
        # _bc_mean_reg_penalty = 110
        # _lambda_equal = _bc_mean_single_error / _bc_mean_reg_penalty
        # _lambda_max = _lambda_equal * 10
        # self._lambda_conts = np.linspace(0, _lambda_max, num_lambdas)
        # self._lambda_conts \
        #     = np.array([0.0001, 0.001, 0.002, 0.005, 0.01, 0.02,
        #                 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100])
        # self._lambda_conts = np.array([0.1, 0.2, 0.3, 0.5, 0.8, 1.3, 2.2, 3.6, 6, 10.])            
        # self._lambda_conts = np.array([0.0001, 0.001, 0.01, 0.1, 1, 10, 100])
        self._lambda_conts = np.array([655128.55685955228])
        self._prediction_steps = 24
        # Using AR order for both endo and exo rather than separate EXO order.
        #self._AR_orders = np.array([1, 2, 6, 12, 24, 2*24, 4*24, 7*24])
        self._AR_orders = np.array([12])
        self._hindsight = 7*4*4*24 + self._model.day # 1344 + 24 hours
        self._data_periods = self._model.dataset.num_train_periods \
                             - np.arange(self._N) - 1
        self._data_idx_array = None

    def _init_grid(self):
        if self._data_idx_array is not None:
            raise RuntimeError('_init_grid already called. Something is wrong.')
        num_lambdas = len(self._lambda_conts)
        num_AR_orders = len(self._AR_orders)
        grid_shape = (self._N, num_lambdas, num_AR_orders)
        # ij indexing in meshgrid ensures that the data index runs
        # slowest. This is desirable, since processing time is fairly
        # constant for the different days, but may vary a lot for
        # different lambda values and AR orders.
        self._data_idx_array, self._lambda_idx, self._AR_idx = \
            np.meshgrid(*[np.arange(s) for s in grid_shape], indexing='ij')
        self._losses = np.empty(grid_shape)
        self._losses[:] = np.nan
        self._predictions = np.empty(grid_shape, dtype='object')
        self._parameters = np.empty(grid_shape,
                                    dtype=[('period_idx', 'i8'),
                                           ('lambda_cont','f8'),
                                           ('AR_order', 'i8'),
                                           ('start_date', 'a19'),
                                           ('end_date', 'a19')])
        print 'Regularized AR lambda grid searcher ready for evaluation of ' \
            '{} lambda values * {} AR/X orders on {} time series.' \
            .format(num_lambdas, num_AR_orders, self._N)

    @property
    def _data_idx(self):
        '''This property accesses the actual data_idx array, and at the same
        time allows for late creation of the grid search arrays, which
        in turn allows for modification of grid-search parameters such
        as lambda values and AR orders from code.

        '''
        if self._data_idx_array is None:
            self._init_grid()
        return self._data_idx_array
    
    def _get_params_as_genome(self, lambda_idx, AR_idx):
        genome = [self._AR_orders[AR_idx],
                  self._AR_orders[AR_idx],
                  self._hindsight,
                  self._lambda_conts[lambda_idx]]
        loci = sg.utils.Enum('AR_order', 'EXO_order', 'hindsight', 'lambda_cont')
        return genome, loci

    def _expand_index(self, i):
        return (self._data_idx.flat[i],
                self._lambda_idx.flat[i],
                self._AR_idx.flat[i])

    def evaluate_one(self, i):
        data_idx, lambda_idx, AR_idx = self._expand_index(i)
        print 'Training model for data instance number {}, lambda #{}={}, ' \
            'AR/X order #{}={}...'.format(
                data_idx, lambda_idx, self._lambda_conts[lambda_idx],
                AR_idx, self._AR_orders[AR_idx])
        data_in, data_out = self._model.dataset.get_train_period(
            self._data_periods[data_idx])
        genome, loci = self._get_params_as_genome(lambda_idx, AR_idx)
        prediction = _process_data(self._model, data_in, genome, loci)
        assert(np.all(prediction.index == data_out.index))
        error = self._model.error_func(prediction.values, data_out.values)
        return error, prediction

    def _save_results(self, save_path):
        np.save(save_path + '.losses', self._losses)
        np.save(save_path + '.forecasts', self._predictions)
        np.save(save_path + '.parameters', self._parameters)
        print 'Grid search finished, results saved to {}.losses/forecasts/parameters.'\
            .format(save_path)
        
    def _store_one_result(self, i, error, y_hat):
        idx = self._expand_index(i)
        self._losses[idx] = error
        self._predictions[idx] = y_hat
        self._parameters[idx] \
            = (self._data_periods[idx[0]],
               self._lambda_conts[idx[1]],
               self._AR_orders[idx[2]],
               str(y_hat.first_valid_index()),
               str(y_hat.last_valid_index()))
        
    def evaluate_all(self, save_path):
        for i in range(self._data_idx.size):
            error, y_hat = self.evaluate_one(i)
            self._store_one_result(i, error, y_hat)
        self._save_results(save_path)

    def mpi_evaluate_all(self, save_path):
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        nhosts = comm.Get_size()
        rank = comm.Get_rank()

        # import sys
        # import os
        # here = os.path.dirname(os.path.abspath(__file__))
        # path = os.path.join(here, '{}_stdout_{}.txt'.format(save_path, rank))
        # print 'Node {} redirecting terminal output to {}.'.format(rank, path)
        # sys.stdout = open(path, 'w')
        # sys.stderr = sys.stdout
        
        if rank == 0:
            all_indices = np.arange(self._data_idx.size)
            splits = np.linspace(0, self._data_idx.size, nhosts+1).astype('int')
            scattered = [all_indices[start:end] for (start, end) in \
                         zip(splits[:-1], splits[1:])]
        else:
            scattered = None
        my_indices = comm.scatter(scattered)
        errs_yhats = [self.evaluate_one(i) for i in my_indices]
        all_errs_yhats = comm.gather(errs_yhats)
        if rank == 0:
            # Remove empty responses in case there are more nodes than
            # tasks. Make sure to remove them in decreasing order, otherwise
            # indices will be invalid.
            if nhosts > self._data_idx.size:
                for i in np.where(np.logical_not(np.diff(splits)))[0][::-1]:
                    all_errs_yhats.pop(i)
            for i, (error, y_hat) in zip(range(self._data_idx.size),
                                         itertools.chain(*all_errs_yhats)):
                self._store_one_result(i, error, y_hat)
            self._save_results(save_path)


class RegularizedVanillaGridSearcher(object):
    def __init__(self, model):
        self._model = model
        self._N = 100 # Number of prediction days
        num_lambdas = 50
        self._lambda_conts = np.concatenate(([0], np.logspace(-3, 6, num_lambdas)))
        self._prediction_steps = 24
        self._idx_array = None

    def _init_grid(self):
        if self._idx_array is not None:
            raise RuntimeError('_init_grid already called. Something is wrong.')
        num_lambdas = len(self._lambda_conts)
        grid_shape = (num_lambdas, )
        self._idx_array = np.arange(num_lambdas)
        self._losses = np.empty(grid_shape, dtype=[('rmse', 'f8'),
                                                   ('mrmse', 'f8')])
        self._losses[:] = (np.nan, np.nan)
        self._predictions = np.empty(grid_shape, dtype='object')
        self._parameters = np.empty(
            grid_shape, dtype=[('lambda_cont','f8')])
        print 'Regularized vanilla grid searcher ready for evaluation of ' \
            '{} lambda values.'.format(num_lambdas)

    @property
    def _idx(self):
        '''This property accesses the actual data_idx array, and at the same
        time allows for late creation of the grid search arrays, which
        in turn allows for modification of grid-search parameters such
        as lambda values and AR orders from code.

        '''
        if self._idx_array is None:
            self._init_grid()
        return self._idx_array
    
    def _get_params_as_genome(self, lambda_idx):
        genome = [self._lambda_conts[lambda_idx]]
        loci = sg.utils.Enum('lambda_cont')
        return genome, loci

    def _expand_index(self, i):
        return (i, )

    def evaluate_one(self, i):
        lambda_idx = self._expand_index(i)[0]
        data_in = self._model.dataset.train_data.copy()
        day = self._model.day
        prediction_steps = self._N * day
        data_out = data_in['Load'][-prediction_steps:].copy()
        data_in['Load'][-prediction_steps:] = np.nan
        genome, loci = self._get_params_as_genome(lambda_idx)
        prediction = _process_data(self._model, data_in, genome, loci,
                                   prediction_steps=prediction_steps)
        assert(np.all(prediction.index == data_out.index))
        rmse = self._model.error_func(prediction.values, data_out.values)
        mrmse = np.mean([self._model.error_func(prediction.ix[i*day:(i+1)*day].values,
                                               data_out.ix[i*day:(i+1)*day].values)
                        for i in range(self._N)])
        print 'Model with lambda #{}={} got RMSE {} (MRMSE {}).'.format(
            lambda_idx, self._lambda_conts[lambda_idx],
            rmse, mrmse)
        return rmse, mrmse, prediction

    def _save_results(self, save_path):
        np.save(save_path + '.losses', self._losses)
        np.save(save_path + '.forecasts', self._predictions)
        np.save(save_path + '.parameters', self._parameters)
        print 'Grid search finished, results saved to {}.losses/forecasts/parameters.'\
            .format(save_path)
        
    def _store_one_result(self, i, errors, y_hat):
        idx = self._expand_index(i)
        self._losses[idx] = errors
        self._predictions[idx] = y_hat
        self._parameters[idx] = self._lambda_conts[idx[0]]
        
    def evaluate_all(self, save_path):
        for i in range(self._idx.size):
            rmse, mrmse, y_hat = self.evaluate_one(i)
            self._store_one_result(i, (rmse, mrmse), y_hat)
        self._save_results(save_path)

    def mpi_evaluate_all(self, save_path):
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        nhosts = comm.Get_size()
        rank = comm.Get_rank()

        if rank == 0:
            all_indices = np.arange(self._idx.size)
            splits = np.linspace(0, self._idx.size, nhosts+1).astype('int')
            scattered = [all_indices[start:end] for (start, end) in \
                         zip(splits[:-1], splits[1:])]
        else:
            scattered = None
        my_indices = comm.scatter(scattered)
        errs_yhats = [self.evaluate_one(i) for i in my_indices]
        all_errs_yhats = comm.gather(errs_yhats)
        if rank == 0:
            # Remove empty responses in case there are more nodes than
            # tasks. Make sure to remove them in decreasing order, otherwise
            # indices will be invalid.
            if nhosts > self._idx.size:
                for i in np.where(np.logical_not(np.diff(splits)))[0][::-1]:
                    all_errs_yhats.pop(i)
            for i, (rmse, mrmse, y_hat) in zip(range(self._idx.size),
                                               itertools.chain(*all_errs_yhats)):
                self._store_one_result(i, (rmse, mrmse), y_hat)
            self._save_results(save_path)


def parse_args(args=None):
    parser = argparse.ArgumentParser(description=
        'Estimate vector models with different levels of regularization.')
    parser.add_argument('--save-path', required=True,
                        help='path to file where results should be saved. '\
                        'Will be appended with ".losses.npy" and ".forecasts.npy".') 
    parser.add_argument(
        '--bc-data', action='store_const', const='bc-transmission',
        dest='dataset', help='Use transmission data from BC Columbia.')
    parser.add_argument(
        '--total-load', action='store_const', const='total-load',
        dest='dataset', help='Use distribution data from Norway.')
    parser.add_argument(
        '--gef-data', action='store_const', const='gefcom-2012',
        dest='dataset', help='Use GEFCom 2012 data.')
    parser.add_argument(
        '--vanilla', action='store_true', dest='vanilla',
        help='Use the vectorized vanilla model instead of ARX.')
    # Trying to load mpi4py causes segfault on some systems if the
    # program is not actually launched under MPI (i.e. via mpirun or
    # similar).
    parser.add_argument(
        '--no-mpi', action='store_true', dest='no_mpi', help='Do not load any MPI libraries.')
    
    # parser.add_argument('--total-load', action='store_true', help='Use distribution data.')
    # parser.add_argument('--gef-data', action='store_true', help='Use GEFCom 2012 data.')
    parsed_args = parser.parse_args(args)
    
    lp_cmdline = ['--data-seed=0',
                  '--standardize']
    if not parsed_args.vanilla:
        lp_cmdline += ['--subtract-weekly-pattern']
    if parsed_args.dataset is None:
        raise RuntimeError('A dataset argument must be supplied.')
    if parsed_args.dataset == 'bc-transmission':
        lp_cmdline += ['--bc-data',
                       '--remove-holidays']
    elif parsed_args.dataset == 'total-load':
        lp_cmdline += ['--total-load']
    elif parsed_args.dataset == 'gefcom-2012':
        lp_cmdline += ['--gef-data']
        raise RuntimeError(
            'GEFCom data is not supported yet, remains to assign a '
            'set of weights for the different temperature zones.')
    else:
        raise RuntimeError('Unrecognized dataset: {}'.format(parsed_args.dataset))
    parser = lp.prediction_options()
    parser = lp.data_options(parser)
    parser = ga.ga_options(parser)
    lp_options, _ = parser.parse_args(lp_cmdline)
    lp.options = lp_options
    return parsed_args, lp_options

def _cmdline_mpi_grid_search():
    import argparse
    args, lp_options = parse_args()
    if args.no_mpi:
        is_master = True
        using_mpi = False
    else:
        from mpi4py import MPI
        is_master = MPI.COMM_WORLD.Get_rank() == 0
        using_mpi = MPI.COMM_WORLD.Get_size() > 1
    if is_master:
        print 'Launched with dataset {}, saving results to {}.losses/.forecasts[.npy].'\
            .format(args.dataset, args.save_path)
    if args.vanilla:
        model = lpra.RegularizedVanillaModelCreator(lp_options).get_model()
        gs = RegularizedVanillaGridSearcher(model)
    else:
        model = lpra.LinearRegularizedVectorARModelCreator(lp_options).get_model()
        gs = RegularizedARGridSearcher(model)
    if using_mpi:
        gs.mpi_evaluate_all(args.save_path)
    else:
        gs.evaluate_all(args.save_path)

            
if __name__ == "__main__":
    _cmdline_mpi_grid_search()
        
