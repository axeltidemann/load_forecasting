"""
Train and predict function for wavelets, as described in
Wavelet-Based Combined Signal Filtering and
Prediction, Olivier Renaud, Jean-Luc Starck, and Fionn Murtagh

as well as

using RTrees based on wavelet decomposition, where the coefficients are the
indexes in the RTree.

Author: Axel
"""

from datetime import timedelta as dt
import glob
import sys
import math
from tempfile import NamedTemporaryFile
import os
import time

import numpy as np
import pandas as pd
import pywt
import mdp
import Oger
from rtree import index

import sg.utils
import load_cleansing

_local_cleaning_func = None
_model = None

def set_local_cleaning_func(clean_func, model):
    global _local_cleaning_func, _model
    _local_cleaning_func = clean_func
    _model = model
    
def _collect_coefficients(data, genome, loci):
    l = loci
    scale = genome[l.scale]
    (temps, loads) = (data['Temperature'], data['Load'])
    loads_coeffs = pywt.swt(loads, 'haar', level=scale)
    temps_coeffs = pywt.swt(temps, 'haar', level=scale)

    # The first 2^scale datapoints cannot be used to predict because of lack of
    # history. scale+1 because of the smooth array.
    a = np.zeros((len(loads) - 2**scale, 2*(scale+1)*genome[l.Aj]))

    # Collect coefficients for each scale + smooth array.
    for i in range(len(a)):
        row = []
        # cAn, the smoothest of the smooth arrays.
        for k in range(1, genome[l.Aj]+1):
            row.append(loads_coeffs[-1][0][2**scale + i - 2**scale*(k-1)])
            row.append(temps_coeffs[-1][0][2**scale + i - 2**scale*(k-1)])
        # cD, the details.
        for j in range(1, scale+1):
            for k in range(1, genome[l.Aj]+1):
                row.append(loads_coeffs[j-1][1][2**scale + i - 2**j*(k-1)])
                row.append(temps_coeffs[j-1][1][2**scale + i - 2**j*(k-1)])

        a[i] = np.array(row)

    return a

def hourbyhour_multiscale_prediction_ga(data, genome, loci, prediction_steps, spinup=0):
    prediction = [ multiscale_prediction(data.ix[i::prediction_steps], genome, loci, 1)
                   for i in range(prediction_steps) ]
    return pd.concat(prediction)

def multiscale_prediction(data, genome, loci, prediction_steps, spinup=0):
    hindsight = 2**math.floor(math.log(genome[loci.hindsight], 2))
    data = data[-hindsight-prediction_steps:]
    a = _collect_coefficients(data[:-prediction_steps], genome, loci)

    b = data['Load'][2**genome[loci.scale]+prediction_steps:-prediction_steps]

    (x, residuals, rank, s) = np.linalg.lstsq(a[:-prediction_steps],b)

    return pd.TimeSeries(data=np.dot(a,x)[-prediction_steps:], 
                         index=data.index[-prediction_steps:])

def _get_test_period(data):
    test_days = np.where(np.isnan(data['Load']))[0]
    return np.min(test_days), np.max(test_days)

def _split_to_periods(data, genome, loci, prediction_steps, test_starts, test_ends):
    window = genome[loci.hindsight]
    days_before = [ sg.utils.Normalizer(data[i-window:i], axis=0).normalized
                    for i in range(window, test_starts - prediction_steps, prediction_steps)]
    days_after = [ sg.utils.Normalizer(data[i-window:i], axis=0).normalized
                   for i in range(window+test_ends+1, len(data) - prediction_steps, prediction_steps)]
    days = days_before + days_after
    if _local_cleaning_func is not None:
        days = [_local_cleaning_func(
            day, genome, loci, prediction_steps, _model) for day in days]
    return days

def _coeffs(data, genome, loci, prediction_steps, test_starts, test_ends):
    days = _split_to_periods(
        data, genome, loci, prediction_steps, test_starts, test_ends)
    return [pywt.wavedec(day['Temperature'], 'haar') for day in days], \
      [pywt.wavedec(day['Load'], 'haar') for day in days], days

def _grow_tree(data, genome, loci, prediction_steps):
    test_starts, test_ends = _get_test_period(data)
    temps_coeffs, loads_coeffs, days = _coeffs(
        data, genome, loci, prediction_steps, test_starts, test_ends)
    p = index.Property()
    p.dimension = genome[loci.dimension]*2
    idx = index.Index(properties=p)
    i = 0
    for l_coeff, t_coeff in zip(loads_coeffs, temps_coeffs):
        key_load = [item for sublist in l_coeff for item in sublist][:p.dimension/2]
        key_temp = [item for sublist in t_coeff for item in sublist][:p.dimension/2]
        idx.insert(i, tuple(key_load + key_temp), obj=days[i].index[0])
        i+=1
    return idx

def _weighted_retrieve(data, genome, loci, prediction_steps, spinup, weight_func):
    l = loci
    (temps, loads) = (data['Temperature'], data['Load'])

    idx = _grow_tree(data, genome, loci, prediction_steps)
    window = genome[l.hindsight]
    test_starts, test_ends = _get_test_period(data)
    
    query_loads_norm = sg.utils.Normalizer(loads[test_starts-window:test_starts])
    query_loads = [item for sublist in pywt.wavedec(query_loads_norm.normalized, 'haar')
                   for item in sublist ][:idx.properties.dimension/2]

    query_temps_norm = sg.utils.Normalizer(temps[test_starts-window:test_starts])
    query_temps = [item for sublist in pywt.wavedec(query_temps_norm.normalized, 'haar')
                   for item in sublist ][:idx.properties.dimension/2]

    query = weight_func(query_loads) + weight_func(query_temps)
    # If we find matches where there is a gap in the timeseries (which will throw an exception), we look for the next best match.
    num_matches = 1
    while True:
        match_date = list(idx.nearest(tuple(query), num_matches, objects="raw"))[-1]
        end_date = match_date + dt(hours=genome[l.hindsight]+prediction_steps - 1)
        period = sg.utils.Normalizer(loads[match_date:end_date]).normalized
        prediction = query_loads_norm.expand(period[-prediction_steps:])
        try:
            result = pd.TimeSeries(data=prediction.values,
                                   index=data[test_starts:test_ends+1].index)
            idx.close()
            return result
        except:
            num_matches += 1
            print 'Time gap encountered, we will try match number', num_matches

def _cleansing_key(genome):
    try:
        return [genome[getattr(loci, attr)] for attr in ('t_smooth', 'l_smooth', 't_zscore', 'l_zscore')]
    except:
        return []
    
def retrieve(data, genome, loci, prediction_steps, spinup=0):
    return _weighted_retrieve(data, genome, loci, prediction_steps, spinup, lambda x: x)

def exp_decay_weight_retrieve(data, genome, loci, prediction_steps, spinup=0):
    def exp_decay(query):
        return np.array(query) * np.array([ genome[loci.weight]**i for i in range(len(query)) ])
    return _weighted_retrieve(data, genome, loci, prediction_steps, spinup, exp_decay)

def exp_increase_weight_retrieve(data, genome, loci, prediction_steps, spinup=0):
    def exp_increase(query):
        return np.array(query) * np.array([ genome[loci.weight]**(len(query) - i) for i in range(len(query)) ])
    return _weighted_retrieve(data, genome, loci, prediction_steps, spinup, exp_increase)

def mask_retrieve(data, genome, loci, prediction_steps, spinup=0):
    def mask(query):
        return np.array(query) * np.array(map(lambda x: int(x), bin(genome[loci.mask])[3:3+len(query)]))
    return _weighted_retrieve(data, genome, loci, prediction_steps, spinup, mask)
