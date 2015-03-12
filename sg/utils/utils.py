"""Various utility functions."""

import itertools
import os
import collections
import sys

import numpy as np
from scipy.linalg import qr
from numpy.linalg import solve, inv
from numpy import dot
import matplotlib.pyplot as plt
import pandas as pd

def add_shadowscatter(axes, x, y, z, **kwargs):
    '''Add "shadow" projections on each plane to an existing scatter
    plot. Disables autoscaling, otherwise it is impossible to position
    the projections on each "wall" (the axis limits will be automatically
    readjusted).

    '''
    ones = np.ones(len(x))
    mrgx, mrgy, mrgz = axes.margins()
    side = 0 if axes.elev > 0 else 1
    offset = -1 if side == 0 else 1
    xy_plane = (ones + offset * mrgz) * axes.get_zlim()[side]

    side = 0 if 0 < axes.azim < 180 else 1
    offset = -1 if side == 0 else 1
    xz_plane = (ones + offset * mrgy) * axes.get_ylim()[side]

    side = 0 if abs(axes.azim) < 90 else 1
    offset = -1 if side == 0 else 1
    yz_plane = (ones + offset * mrgx) * axes.get_xlim()[side]

    axes.autoscale(enable=False)
    mykwargs = {'depthshade': False, 'alpha': 1, 'marker': '+', 'c':'0.25'}
    for key in mykwargs.viewkeys() - kwargs.viewkeys():
        kwargs[key] = mykwargs[key]
    axes.scatter3D(x, y, xy_plane, **kwargs)
    axes.scatter3D(x, xz_plane, z, **kwargs)
    axes.scatter3D(yz_plane, y, z, **kwargs)
    pkwargs = {'color':'0', 'linestyle':'dotted', 'alpha': 1}
    for i in range(len(x)):
        axes.plot([x[i], x[i]], [y[i], y[i]], [z[i], xy_plane[0]], **pkwargs)
        axes.plot([x[i], x[i]], [y[i], xz_plane[0]], [z[i], z[i]], **pkwargs)
        axes.plot([x[i], yz_plane[0]], [y[i], y[i]], [z[i], z[i]], **pkwargs)
    return axes


def fahrenheit_to_celsius(deg_f):
    return (deg_f - 32) * 5./9

def celsius_to_fahrenheit(deg_c):
    return deg_c * 9./5 + 32

def get_path(options, base, extension):
    """Return a path, using out_dir and out_postfix from options. The final
    path is thus:
      out_dir/{base}_{out_postfix}.{extension}
    """
    return os.path.join(options.out_dir, 
                        "%s_%s.%s" % (base, options.out_postfix, extension))

def redirect(filelike, path, append=False):
    """Redirect the output from a file/file-like object to the file identified
    in path. If append is True, open path with append flags, otherwise
    overwrite if file exists. Example usage that redirects all print commands
    to the given file:

    redirect(sys.stdout, "output_dump_%d.txt" % os.getpid())
    """
    flags = os.O_APPEND | os.O_CREAT if append else os.O_WRONLY | os.O_CREAT
    fd = os.open(path, flags)
    fd2 = filelike.fileno()
    os.dup2(fd, fd2)
 
def qr_solve(a, b, overwrite_a=False, lwork=None, mode='full', pivoting=False):
    """Similart to numpy.linalg.solve, but uses QR decomposition rather than LU
    decomposition. Slower but more numerically stable. The arguments following
    a and b are passed on to scipy.linalg.qr """
    Q,R = qr(a, overwrite_a=overwrite_a, lwork=lwork, mode=mode,
                       pivoting=pivoting)
    c = dot(Q.T,b)
    #x = np.linalg.solve(R,c) 
    x = dot(inv(R),c)
    return x

def plot_time_series(series, formats, labels, show=True):
    fig = plt.figure()
    if type(series) == type(formats) == type(labels) == list:
        for s,f,l in zip(series, formats, labels):
            s.plot(style=f, label=l)
    else:
        series.plot(style=formats, label=labels)
        
    plt.legend()
    if show:
        plt.show()

def flatten(*lists):
    """Return a flattened version of the inputs. Only one level of nesting,
    lists with sublists will not be flattened."""
    return list(itertools.chain(*lists))

def safe_deep_flatten(l):
    """Safe deep flatten: accepts both lists of sublists and non-iterable list
    elements. From
    http://stackoverflow.com/questions/2158395/flatten-an-irregular-list-of-lists-in-python"""
    for el in l:
        if isinstance(el, collections.Iterable) and not isinstance(el, basestring):
            for sub in safe_deep_flatten(el):
                yield sub
        else:
            yield el

def safe_shallow_flatten(l):
    """Safe shallow flatten: accepts lists containing both sublists and
    non-iterable elements. From
    http://stackoverflow.com/questions/2158395/flatten-an-irregular-list-of-lists-in-python"""
    for el in l:
        if isinstance(el, collections.Iterable) and not isinstance(el, basestring):
            for sub in flatten(el):
                yield sub
        else:
            yield el

def _slow_pinball_error(quantile_forecasts, observed):
    """This is the slow but readable version of the pinball error. Should
    give the same results as pinball_error below.

    """
    
    loss = 0
    for qf, y in zip(quantile_forecasts, observed):
        for a, qa in zip(np.linspace(0.01, 0.99, 99), qf):
            loss += (1 - a) * (qa - y) if qa > y else a * (y - qa)
    return loss / (len(observed) * 99)

def pinball_error(quantile_forecasts, observed):
    """Pinball error function for quantile forecasts, as defined here:
    http://robjhyndman.com/hyndsight/gefcom2014/

    """
    if len(observed.shape) == 1:
        observed = observed.copy()
        observed.shape += (1,)
    diff = quantile_forecasts - observed
    quantiles = np.linspace(0.01, 0.99, 99).reshape(1,99)
    above = diff * (1 - quantiles)
    below = -diff * quantiles
    results = np.empty(diff.shape)
    mask = diff > 0
    results[mask] = above[mask]
    not_mask = np.logical_not(mask)
    results[not_mask] = below[not_mask]
    return results.mean()

def mean_absolute_error(forecasted, observed):
    return np.mean(np.abs(forecasted.flatten() - observed.flatten()))

mae = mean_absolute_error

def mean_absolute_percent_error(forecasted, observed):
    """Mean absolute percentage error is an error measure typically used when
    comparing forecaster performance."""
    if np.any(observed == 0):
        raise ValueError("Don't know how to calculate MAPE when the " \
                         "observed value is 0.")
    all_errors = np.abs(observed.flatten() - forecasted.flatten()) / observed.flatten()
    return all_errors.mean() * 100

def mean_absolute_percent_error_skip_zeros(forecasted, observed):
    """Mean absolute percentage error is an error measure typically used when
    comparing forecaster performance. This variation calculates MAPE only for
    the entries where the observed value is non-zero. It returns a tuple (MAPE,
    fraction_nonzero_elements)"""
    fflat = forecasted.flatten()
    oflat = observed.flatten()
    nz = np.where(oflat != 0)[0]
    mapesz = mean_absolute_percent_error(fflat[nz], oflat[nz])
    return mapesz, float(len(nz)) / len(oflat)

def mean_absolute_percent_error_plus_one(forecasted, observed, divadd=1.0):
    """Mean absolute percentage error is an error measure typically used when
    comparing forecaster performance. This variation divides by (observed +
    divadd) rather than just 'observed', in order to avoid division by zero."""
    all_errors = abs((observed - forecasted) / (observed + divadd))
    return all_errors.sum() / len(observed) * 100

mape = mean_absolute_percent_error
mape_skip_zeros = mean_absolute_percent_error_skip_zeros
mape_plus_one = mean_absolute_percent_error_plus_one

def rmse(forecasted, observed):
    """Root mean square error (RMSE) of the input signal compared to target
    signal."""
    error = np.power(forecasted - observed, 2)
    return np.sqrt(error.mean())

def relative_rmse(forecasted, observed, alpha=1):
    """A variation of RMSE that multiplies the difference between forecast and
    observation with (1+observation) before squaring. This effectively
    penalizes errors in forecasting peaks harder than troughs."""
    error = abs(alpha * observed) * np.power((forecasted - observed), 2)
    return np.sqrt(error.mean())

def scale(data):
    """Scales data to the range [0,1] along the rows."""
    return np.transpose(np.array([ (data[:,i] - \
                                    min(data[:,i]))/(max(data[:,i]) - min(data[:,i])) \
                                    for i in range(data.shape[1]) ]))

# Based on Alex Thomas' suggestion on http://stackoverflow.com/questions/36932/
# (whats-the-best-way-to-implement-an-enum-in-python), which creates an Enum
# class instance on-the-fly. Contrary to the original suggestion, this
# implementation can be pickled.
class Enum(object):
    """Implements an enum in Python. Usage:
    numbers = enum('ZERO', 'ONE', 'TWO', NOT_THREE=4)"""

    def __init__(self, *sequential, **named):
        enums = dict(zip(sequential, range(len(sequential))), **named)
        for (name, value) in enums.iteritems():
            object.__setattr__(self, name, value)

def bound(lower, upper, current):
    """Bound current in [lower, upper]."""
    return max(lower, min(upper, current))

def indicer(*args):
    """Return a dict where 'args' are the keys and the values are integers
    representing the position of each argument in 'args'.

    Example:
    indicer('one', 'two', 'three')
    returns this dict:
    {'one': 0, 'two': 1, 'three': 2}

    The returned dict can be used for as a lookup for systematic manipulation
    of the elements of an array (e.g. a genome) given the name, but not the
    position, of each element."""
    return dict([(args[i], i) for i in range(len(args))])
 
class Normalizer(object):
    """Normalizes a dataset. Upon initialization, new data can be passed in and
    "normalized" or expanded using the scale and offset parameters of the
    original dataset.

    For two-dimensional data, the normalizer behaves similar to numpy.min: if
    no axis argument is given, flatten the array. Otherwise normalize along the
    given axis.

    An integer dataset must be cast to float before it can be normalized. This
    class will not do that for you."""


    def __init__(self, dataset, axis=None):
        """Initialize the Normalizer with the data in 'dataset'. 'dataset' must
        support matrix addition and division (e.g. numpy.array, but not Python
        lists). axis is the axis along which to normalize (None = flatten)"""
        self._raw_data = dataset
        # Try using the min() method if available, as this allows classes to
        # define their own sorting, and also works with scikits.timeseries.
        try:
            self._offset = dataset.min(axis=axis)
            self._range = dataset.max(axis=axis) - self._offset
        except AttributeError:
            self._offset = np.min(dataset, axis=axis)
            self._range = np.max(dataset, axis=axis) - self._offset
        self._empty_range()
        self._shape_params(axis)
        self._normalized = self.normalize(self._raw_data)

    def _shape_params(self, axis):
        """Explicitly set the shape of the parameters, otherwise matrix
        subtraction and division won't work as intended."""
        if axis is None or axis == 0:
            return
        elif axis == 1:
            self._offset.shape = (self._offset.shape[0], 1)
            self._range.shape = (self._range.shape[0], 1)
        else:
            raise ValueError("Normalizer can't handle axis '%s', it only " \
                             "knows how to handle axes 'None', '0' and '1'." % \
                             axis)

    def _empty_range(self):
        """Handle the case where all values in a dataset are the same, so the
        range is 0."""
        try:
            self._range[self._range == 0] = 1
        except TypeError:
            if self._range == 0:
                self._range = 1
                             
    @property
    def normalized(self):
        """The normalized version of the dataset passed to the __init__
        function. This is a read-only property."""
        return self._normalized

    @property
    def raw(self):
        """Returns the dataset passed in as an argument to the __init__
        function. This is a read-only property."""
        return self._raw_data
    
    def normalize(self, data):
        """Scale and offset 'data' according to the normalization parameters
        (range and offset) determined by the dataset passed as argument to the
        __init__ function. Use the 'normalized' property if you need the
        normalized version of the original dataset."""
        return (data - self._offset) / self._range

    def expand(self, data):
        """Expand 'data' by multiplying with the range and shifting with the
        offset determined by the dataset passed as argument to the __init__
        function. Use the 'raw' property if you want the original dataset."""
        return data * self._range + self._offset

def hook_chainer(old_hook, new_hook):
    """Returns a closure that chains hook functions for Oger
    reservoirs. Typical usage:

    reservoir._post_update_hook = hook_chainer(
      reservoir_post_update_hook, my_hook)
    """
    def chainer(states, input, timestep):
        old_hook(states, input, timestep)
        new_hook(states, input, timestep)
    return chainer

def concat_and_calc_error(predictions, target, error_func):
    all_preds = pd.concat(predictions)
    assert len(all_preds.index) == len(target.index),\
           'List of predictions did not match target. '\
           'Predictions missing or overlap?'
    assert(np.all(all_preds.index == target.index))
    return error_func(all_preds.values, target.values)

def calc_median_error(predictions, target, error_func):
    return np.median([error_func(p, target[p.index]) for p in predictions])

def plot_predictions(predictions, pred_fmt='r-', label='Prediction'):
    predictions[0].plot(style=pred_fmt, label=label)
    for i in range(1, len(predictions)):
        predictions[i].plot(style=pred_fmt, label='_nolegend_')

def plot_target_predictions(target, predictions, pred_fmt='r-'):
    target.plot(style='b', label='Target')
    plot_predictions(predictions, pred_fmt)
    plt.legend(loc=3)

def diffinv(x, n=1, xi=0):
    """An implementation of the R 'diffinv' function, using variable naming
    from numpy.diff. Computes the inverse function of the lagged differences
    function diff.

    x is the array of lagged differences (lag must be 1)
    n is the difference order
    xi is the array of initial values
    """
    restored = np.r_[0, xi, x]
    if len(restored) != len(x) + n + 1:
        raise ValueError("Length of xi must equal the difference order n.")
    for diff in range(n, 0, -1):
        restored[diff] -= restored[diff-1]
        for i in range(diff + 1, len(restored)):
            restored[i] += restored[i-1]
    return restored[1:]

def ask_user(prompt, default_yes=True):
    """Ask the user a yes/no question in the terminal, return True of False correspondingly.
    default_yes=True will yield:  "prompt [y]/n? "
    default_yes=False will yield: "prompt y/[n]? "
    default_yes=None will yield:  "prompt y/n? "
    """
    yes, no = ['Y', 'y'], ['N', 'n']
    if default_yes is None:
        options = "y/n"
    elif default_yes:
        options = "[y]/n"
        yes += ['']
    else:
        options = "y/[n]"
        no += ['']
    full_prompt = "%s %s? " % (prompt, options)
    answer = raw_input(full_prompt)
    while not answer in yes + no:
        answer = raw_input("Please answer y or n: %s" % full_prompt)
    if answer in yes:
        return True
    if answer in no:
        return False
    raise LogicError("Bug in code, you should not have come here.")


def n_fold_random(n_samples, n_folds, rng=None):
    '''Taken from Oger n_fold_random, just added support for supplying the
    random number generator as input argument.

    n_fold_random(n_samples, n_folds) -> train_indices, test_indices
    
    Return indices to do random n_fold cross_validation. Two lists are returned, with n_folds elements each.
        - train_indices contains the indices of the dataset used for training
        - test_indices contains the indices of the dataset used for testing

    '''
    import mdp
    
    if n_folds <= 1:
        raise Exception('Number of folds should be larger than one.')

    if n_folds > n_samples:
        raise Exception('Number of folds (%d) cannot be larger than the number of samples (%d).'%(n_folds, n_samples))

    if rng is None:
        randperm = mdp.numx.random.permutation(n_samples)
    else:
        randperm = rng.permutation(n_samples)
    train_indices, test_indices = [], []
    foldsize = int(mdp.numx.floor(float(n_samples) / n_folds))

    for fold in range(n_folds):
        # Select the sample indices used for testing
        test_indices.append(randperm[fold * foldsize:foldsize * (fold + 1)])
        # Select the rest for training
        train_indices.append(mdp.numx.array(mdp.numx.setdiff1d(randperm, test_indices[-1])))
    return train_indices, test_indices

def normalized_weighted_mean(data, weights):
    '''Normalize weights so they sum to one, then use the result to estimate
    the weighted mean of the data.

    '''
    nw = weights / weights.sum()
    return (data * nw).sum(axis=1)


if __name__ == "__main__":
    from unittest import main
    main(module="test_" + __file__[:-3])
    
