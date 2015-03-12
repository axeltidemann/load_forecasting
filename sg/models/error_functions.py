"""Code for experiments in 'Error functions' paper."""

from dateutil.relativedelta import relativedelta
import random
import itertools

import pandas as pd
import numpy as np
import Oger

import bfgs
import sg.data.gefcom2012 as gefcom
import sg.data.bchydro as bchydro
import sg.utils
import splines

def load_bc_dataset():
    return pd.DataFrame(
        {'Load': bchydro.load_remove_zeros(),
         'Temperature': bchydro.temperature()})

def load_gef_dataset():
    gefdata = gefcom.load_solved_non_nan()
    gef1 = pd.DataFrame({'Load': gefdata.load.zone_1,
                         'Temperature': gefdata.temp.station_2})
    gef3 = pd.DataFrame({'Load': gefdata.load.zone_3,
                         'Temperature': gefdata.temp.station_11})
    gef_sys = pd.DataFrame({'Load': gefdata.load.sum(axis=1),
                         'Temperature': gefdata.temp.mean(axis=1)})
    return gef1, gef3, gef_sys

def create_ann_dataset(data, res=24):
    """From a dataframe containing load and temperature timeseries, build a
    dataset containing all columns specified in the 'Error functions'
    paper.

    """
    ds = data.copy()
    # Load for the corresponding and two adjacent hours of the last two days
    for t in range(res-1, res+2) + range(2*res-1, 2*res+2):
        ds['Load_t-{}'.format(t)] = ds['Load'].shift(t)
    # Load for the corresponding hour of the five days before that
    for t in range(3, 8):
        ds['Load_t-{}*{}'.format(res, t)] = ds['Load'].shift(t*res)
    # The minimum and maximum loads and temperatures on the previous
    # day, and the average load and temperature for the last seven days
    for col in ('Load', 'Temperature'):
        # transform is desctructive, so make copies.
        grpcpy = data[col].copy()
        ds['{}_max_day'.format(col)] = \
            grpcpy.groupby(pd.Grouper(freq='D')).transform(lambda x: x.max()).shift(res)
        grpcpy = data[col].copy()
        ds['{}_min_day'.format(col)] = \
            grpcpy.groupby(pd.Grouper(freq='D')).transform(lambda x: x.min()).shift(res)
        grpcpy = data[col].copy()
        ds['{}_mean_week'.format(col)] = \
            grpcpy.groupby(pd.Grouper(freq='W')).transform(lambda x: x.mean()).shift(res*7)
    # Temperature for the last two hours and the current hour
    for t in range(1, 3): 
        ds['Temperature_t-{}'.format(t)] = ds['Temperature'].shift(t)
    # The mean temperature for the last 24 hours
    ds['Temperature_mean_24'] = pd.rolling_mean(ds['Temperature'], res).shift(1)
    return ds

def convert_to_vectormodel_xy_dataset(data, res=24):
    """From a dataset containing the current load in the column 'Load',
    return a list of <res> datasets with one observation per day,
    element i of this list contains the load for period i of the coming
    day.

    'data' shuld be a pd.DataFrame. Returns a list of pd.DataFrame
    tuples.

    """
    # Make sure all subsets are the same length (note the parenthesis
    # that ensures integer division happens before multiplication):
    last = res * (len(data) / res)
    vds = [(data.iloc[i:last:res].drop('Load', axis=1),
            data['Load'].iloc[i:last:res]) for i in range(res)]
    return vds

def remove_nans(data):
    """Remove all rows containing a NaN in any column."""
    return data[np.logical_not(np.any(np.isnan(data), axis=1))]

def split_train_test(data):
    """Split a dataset into training and test, where test is the last year of data."""
    test_period_starts = data.index[-1] - relativedelta(years=1)
    return data[:test_period_starts], data[test_period_starts:]

def vector_xy_dataset_split_indices(data, indices):
    """Given a dataset ((x0, y0), (x1, y1), ..., (xn, yn)), and a list of
    index vectors, split the data into 
      (((x0[indices[0]], y0[indices[0]]), ... (xn[indices[0]], yn[indices[0]])),
       ((x0[indices[1]], y0[indices[1]]), ... (xn[indices[1]], yn[indices[1]])),
       ...,
       ((x0[indices[-1]], y0[indices[-1]]), ... (xn[indices[-1]], yn[indices[-1]])))

    Expects and returns pd.Series or pd.DataFrame.
    """
    return [[(x.iloc[idx], y.iloc[idx]) for x, y in data] for idx in indices]


class ParameterGridSearcher(object):
    def __init__(self, dataset_rng=None):
        self._num_folds = 4
        self._epochs = 1500
        self._rule = 'mse'
        self._loss = Oger.utils.mse
        self._dataset_rng = dataset_rng
        max_layers = (1, 2, 3)
        self._nodes_per_layer = (2, 4, 8)
        self._net_topos = [[n] for n in self._nodes_per_layer]
        for num_layers in max_layers[1:]:
            self._net_topos += self._all_nlayer_nets(num_layers)
        self._hours = range(24)
        self._topo_idx, self._hr_idx = \
            np.meshgrid(np.arange(len(self._net_topos)),
                        np.arange(len(self._hours)))
        self._losses = np.empty(shape=(len(self._net_topos), len(self._hours)))
        self._losses[:] = np.nan
        # Late loading to avoid slow creation
        self._training_data = None
        print 'Grid-searcher ready for evaluation of {} different ' \
            'parameter sets (shape {}) '.format(self._losses.size, self._losses.shape)

    def _load_data(self):
        if self._training_data is None:
            # Load data once, rather than for every evaluation
            _, _, gef_sys = load_gef_dataset()
            ds = remove_nans(create_ann_dataset(gef_sys))
            self._training_data, _ = split_train_test(ds)

    def _all_nlayer_nets(self, num_layers):
        return zip(*map(lambda x: x.flatten(),
                        np.meshgrid(*[self._nodes_per_layer for _ in range(num_layers)])))

    def _expand_index(self, i):
        return self._topo_idx.flat[i], self._hr_idx.flat[i]
    
    def evaluate_one(self, i):
        self._load_data()
        topo_idx, hr_idx = self._expand_index(i)
        net_topo = self._net_topos[topo_idx]
        hour = self._hours[hr_idx]
        # Hack to check that indexing is correct
        # import functools
        # return functools.reduce(lambda x0, x: x0 * 100 + x, net_topo, 0) * 1000 + hour
        print 'Evaluation of parameter set {} started: network topography {}, hour {}'\
            .format(i, net_topo, hour)
        x, y = convert_to_vectormodel_xy_dataset(self._training_data)[hour]
        folds = sg.utils.n_fold_random(len(x), self._num_folds, self._dataset_rng)
        losses = np.zeros(self._num_folds)
        for fold in range(self._num_folds):
            fold_x_train = x.iloc[folds[0][fold]].values
            fold_y_train = np.atleast_2d(y.iloc[folds[0][fold]].values).T
            fold_x_test = x.iloc[folds[1][fold]]
            fold_y_test = np.atleast_2d(y.iloc[folds[1][fold]].values).T
            model = bfgs.BFGSPredictor(net_topo)
            model.train(self._rule, (fold_x_train, fold_y_train), self._epochs)
            yhat = model.execute(fold_x_test)
            losses[fold] = self._loss(yhat, fold_y_test)
            mape = sg.utils.mape_skip_zeros(yhat, fold_y_test)[0]
            print 'Error for model {}, fold {}: {} (MAPE {})'.format(i, fold, losses[fold], mape)
        return losses.mean()

    def mpi_evaluate_all(self, savepath):
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        nhosts = comm.Get_size()
        rank = comm.Get_rank()
        if rank == 0:
            all_indices = np.arange(self._topo_idx.size)
            splits = np.linspace(0, self._topo_idx.size, nhosts+1).astype('int')
            scattered = [all_indices[start:end] for (start, end) in \
                         zip(splits[:-1], splits[1:])]
        else:
            scattered = None
        my_indices = comm.scatter(scattered)
        losses = [self.evaluate_one(i) for i in my_indices]
        all_losses = comm.gather(losses)
        if rank == 0:
            all_losses = np.concatenate(all_losses)
            for i in range(len(all_losses)):
                self._losses[self._expand_index(i)] = all_losses[i]
            np.save(savepath, self._losses)
            print 'Master finished, losses saved to {}.'.format(savepath)


class ErrorFunctionEvaluator(object):
    def __init__(self, loss_func, inverted_estim=False):
        self._epochs = 1500
        self._repeats = range(100)
        self._dataset_names = ['bc', 'gef_1', 'gef_3', 'gef_sys']
        self._loss_func = loss_func
        self._test_loss_funcs = ['mse', 'mape', 'inverted mape']
        self._inverted_estim = inverted_estim
        self._hours = range(24)
        self._data_idx, self._rpt_idx = \
            np.meshgrid(np.arange(len(self._dataset_names)),
                        np.arange(len(self._repeats)))
        self._losses = np.empty(shape=(len(self._dataset_names),
                                       len(self._repeats),
                                       len(self._hours),
                                       len(self._test_loss_funcs)))
        self._losses[:] = np.nan
        self._forecasts = np.empty(shape=(len(self._dataset_names),
                                          len(self._repeats)),
                                   dtype=np.object)
        self._valid_test_indices = np.empty(shape=(len(self._dataset_names),
                                                   len(self._repeats), 2),
                                            dtype=np.object)
        # Late loading to avoid slow creation
        self._datasets = None
        self._datasets_inv = None
        self._envelopes = None
        print 'Error function evaluator ready for evaluation of {} different ' \
            'load predictors (shape {}).'.format(self._forecasts.size, self._forecasts.shape)

    def _load_data(self):
        if self._datasets is None:
            # Load data once, rather than for every evaluation
            self._datasets = dict()
            self._datasets_inv = dict()
            bc = load_bc_dataset()
            gef_1, gef_3, gef_sys = load_gef_dataset()
            raw_data = [bc, gef_1, gef_3, gef_sys]
            raw_data_inv = []
            self._envelopes = []
            for i in range(len(raw_data)):
                self._envelopes.append(splines.SplineEnvelope(raw_data[i]['Load'], 'D'))
                raw_data_inv.append(raw_data[i].copy())
                raw_data_inv[-1]['Load'] = self._envelopes[i].flip()
            for i in range(len(raw_data)):
                ds = remove_nans(create_ann_dataset(raw_data[i]))
                train, test = split_train_test(ds)
                self._datasets[self._dataset_names[i]] = (train, test)
            for i in range(len(raw_data_inv)):
                ds = remove_nans(create_ann_dataset(raw_data_inv[i]))
                train, test = split_train_test(ds)
                self._datasets_inv[self._dataset_names[i]] = (train, test)

    def copy_data(self, other):
        from copy import deepcopy
        self._datasets = deepcopy(other._datasets)
        self._datasets_inv = deepcopy(other._datasets_inv)
        self._envelopes = deepcopy(other._envelopes)
        
    def _expand_index(self, i):
        return (self._data_idx.flat[i],
                self._rpt_idx.flat[i])

    def evaluate_one(self, i):
        data_idx, rpt_idx = self._expand_index(i)
        print 'Training model for dataset {}, repetition {}...'.format(self._dataset_names[data_idx], rpt_idx)
        print 'Random state:'
        print np.random.get_state()

        # Hack to check that indexing is correct
        # import functools
        # test_errors = []
        # for hr in range(24):
        #     base = data_idx * 100000000 + rpt_idx * 100000 + hr * 100
        #     test_errors.append([base + i for i in range(3)])
        # return test_errors, None
        self._load_data()        
        if self._inverted_estim:
            train, test = self._datasets_inv[self._dataset_names[data_idx]]
        else:
            train, test = self._datasets[self._dataset_names[data_idx]]
        hourly_train = convert_to_vectormodel_xy_dataset(train)
        hourly_test = convert_to_vectormodel_xy_dataset(test)
        vi, ti = Oger.evaluation.train_test_only(len(hourly_test[0][0]), 0.5, random=True)
        self._valid_test_indices[data_idx, rpt_idx, :] = [vi, ti]
        hourly_valid, hourly_test = vector_xy_dataset_split_indices(hourly_test, (vi[0], ti[0]))
        model = bfgs.BFGSVectorModel([5])
        model.train(self._loss_func, hourly_train, self._epochs, hourly_valid)
        x_test, y_test = zip(*hourly_test)
        y_hat = model.execute(x_test)
        test_errors = []
        envelope = self._envelopes[data_idx]
        for y_hat_hr, y_test_hr in zip(y_hat, y_test):
            y_hat_hr_flip = envelope.flip(y_hat_hr)
            y_test_hr_flip = envelope.flip(y_test_hr)
            if self._inverted_estim:
                mse = Oger.utils.mse(y_hat_hr_flip.values, y_test_hr_flip.values)
                mape = sg.utils.mape_skip_zeros(y_hat_hr_flip.values, y_test_hr_flip.values)[0]
                imape = sg.utils.mape_skip_zeros(y_hat_hr.values, y_test_hr.values)[0]
            else:
                mse = Oger.utils.mse(y_hat_hr.values, y_test_hr.values)
                mape = sg.utils.mape_skip_zeros(y_hat_hr.values, y_test_hr.values)[0]
                imape = sg.utils.mape_skip_zeros(y_hat_hr_flip.values, y_test_hr_flip.values)[0]
            test_errors.append((mse, mape, imape))
        return test_errors, y_hat

    def mpi_evaluate_all(self, save_path):
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        nhosts = comm.Get_size()
        rank = comm.Get_rank()

        import sys
        import os
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(here, '{}_stdout_{}.txt'.format(save_path, rank))
        print 'Node {} redicting to {}.'.format(rank, path)
        sys.stdout = open(path, 'w')
        sys.stderr = sys.stdout
        
        if rank == 0:
            all_indices = np.arange(self._data_idx.size)
            splits = np.linspace(0, self._data_idx.size, nhosts+1).astype('int')
            scattered = [all_indices[start:end] for (start, end) in \
                         zip(splits[:-1], splits[1:])]
        else:
            scattered = None
        my_indices = comm.scatter(scattered)
        errs_hats = [self.evaluate_one(i) for i in my_indices]
        all_errs_hats = comm.gather(errs_hats)
        if rank == 0:
            # Remove empty responses in case there are more nodes than
            # tasks. Make sure to remove them in decreasing order, otherwise
            # indices will be invalid.
            if nhosts > self._data_idx.size:
                for i in np.where(np.logical_not(np.diff(splits)))[0][::-1]:
                    all_errs_hats.pop(i)
            for i, (test_errors, y_hat) in zip(range(self._data_idx.size),
                                               itertools.chain(*all_errs_hats)):
                data_idx, rpt_idx = self._expand_index(i)
                for hr_idx in range(24):
                    for err_idx in range(3):
                        self._losses[data_idx, rpt_idx, hr_idx, err_idx] = \
                            test_errors[hr_idx][err_idx]
                self._forecasts[data_idx, rpt_idx] = y_hat
            save_path_losses = save_path + '.losses'
            save_path_forecasts = save_path + '.forecasts'
            np.save(save_path_losses, self._losses)
            np.save(save_path_forecasts, self._forecasts)
            print 'Master finished, losses saved to {}, forecasts saved to {}.'\
                .format(save_path_losses, save_path_forecasts)


def _cmdline_mpi_grid_search():
    import argparse
    from mpi4py import MPI
    parser = argparse.ArgumentParser(description=
        'Perform a grid-search for BFGS ANN load prediction parameters.')
    parser.add_argument('--save-path', required=True,
                        help='path to file where results (losses) should be saved.') 
    parser.add_argument('--data-seed', type=int, required=True,
                        help='random seed for dataset (actually for generating '
                        'folds for n-fold cross-validation).')
    args = parser.parse_args()
    if MPI.COMM_WORLD.Get_rank() == 0:
        print 'Launched with random seed {}, saving results to {}.'\
            .format(args.data_seed, args.save_path)
    gs = ParameterGridSearcher(np.random.RandomState(args.data_seed))
    gs.mpi_evaluate_all(args.save_path)

def _cmdline_mpi_error_functions():
    import argparse
    from mpi4py import MPI
    parser = argparse.ArgumentParser(description=
        'Optimize load prediction models for given error function, evaluate on all error functions.')
    parser.add_argument('--save-path', required=True,
                        help='path to file where results should be saved. '\
                        'Will be appended with ".losses.npy" and ".forecasts.npy".') 
    parser.add_argument('--error-function', required=True,
                        help='Error function: mse, mape, imape, mix, imix.')
    args = parser.parse_args()
    if MPI.COMM_WORLD.Get_rank() == 0:
        print 'Launched with error function {}, saving results to {}.losses/.forecasts[.npy].'\
            .format(args.error_function, args.save_path)
    inverted_estim = False
    if args.error_function == 'imape':
        args.error_function = 'mape'
        inverted_estim = True
    if args.error_function == 'imix':
        args.error_function = 'mix'
        inverted_estim = True
    efeval = ErrorFunctionEvaluator(args.error_function, inverted_estim)
    efeval.mpi_evaluate_all(args.save_path)
            
if __name__ == "__main__":
    #_cmdline_mpi_grid_search()
    _cmdline_mpi_error_functions()    
        
