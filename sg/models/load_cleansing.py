import hashlib
from multiprocessing import Lock
import sys
import time

import numpy as np
import pandas as pd

import sg.models.spclean as cln
from sg.utils.cache import ATimeCache
from sg.utils.timer import SimpleTimer

_smoother = None
_max_cache_size = 10000
_temp_mutex = Lock()
_load_mutex = Lock()
_temp_cache = ATimeCache(_max_cache_size)
_load_cache = ATimeCache(_max_cache_size)

def _get_dataset_hash(dataset):
    m = hashlib.md5()
    m.update(dataset)
    # Put the index in the hash, otherwise invalid datasets will be created
    # when we have no temperature data (different dates, same data ->
    # TimeSeries with lots of NaNs).
    m.update(str(dataset.index[0].value))
    m.update(str(dataset.index[-1].value))
    return m.digest()
    
def bspline_clean_dataset(dataset, genome, loci, prediction_steps):
    """Clean a dataset containing temperatures and loads using cleaning
    parameters from the genome. The dataset is expected to contain NaNs in the
    last *prediction_steps* elements of the Load series"""
    # Having the smoother as a global is not nice, but it speeds up things A
    # LOT, because pickling the smoother caches takes a long time for large
    # matrices (long time series).
    global _smoother, _temp_cache, _load_cache
    if _smoother is None:
        _smoother = cln.BSplineSmoother(dataset, smoothness=1)
    clean_data = dataset.copy()
    key = (_get_dataset_hash(dataset["Temperature"]), 
           genome[loci.t_smooth], genome[loci.t_zscore])
    try:
        _temp_mutex.acquire()
        clean_data['Temperature'] = _temp_cache[key].copy()
        # print "Got temp from cache: <dataset_hash>", key[1], key[2]
        # sys.stdout.flush()
    except KeyError:
        _temp_mutex.release()
        # print "Storing temp to cache: <dataset_hash>", key[1], key[2]
        # sys.stdout.flush()
        clean_data['Temperature'] = \
          cln.bspline_clean(dataset['Temperature'], 
                            genome[loci.t_smooth], 
                            genome[loci.t_zscore], _smoother)
        _temp_mutex.acquire()
        _temp_cache[key] = clean_data['Temperature'].copy()
    _temp_mutex.release()
    key = (_get_dataset_hash(dataset["Load"]), 
           genome[loci.l_smooth], genome[loci.l_zscore])
    try:
        _load_mutex.acquire()
        clean_data['Load'][:-prediction_steps] = _load_cache[key].copy()
        # print "Got load from cache: <dataset_hash>", key[1], key[2]
        # sys.stdout.flush()
    except KeyError:
        _load_mutex.release()
        # print "Storing load to cache: <dataset_hash>", key[1], key[2]
        # sys.stdout.flush()
        clean_data['Load'][:-prediction_steps] = \
          cln.bspline_clean(dataset['Load'][:-prediction_steps], 
                            genome[loci.l_smooth], 
                            genome[loci.l_zscore], _smoother)
        _load_mutex.acquire()
        _load_cache[key] = clean_data['Load'][:-prediction_steps].copy()
    _load_mutex.release()
    return clean_data

def bspline_clean_dataset_no_cache(dataset, genome, loci, prediction_steps):
    """Clean a dataset containing temperatures and loads using cleaning
    parameters from the genome. The dataset is expected to contain NaNs in the
    last *prediction_steps* elements of the Load series"""
    # Having the smoother as a global is not nice, but it speeds up things A
    # LOT, because pickling the smoother caches takes a long time for large
    # matrices (long time series).
    global _smoother
    if _smoother is None:
        _smoother = cln.BSplineSmoother(dataset, smoothness=1)
    clean_data = dataset.copy()
    clean_data['Temperature'] = cln.bspline_clean(dataset['Temperature'], 
                                                  genome[loci.t_smooth], 
                                                  genome[loci.t_zscore], _smoother)
    clean_data['Load'][:-prediction_steps] = \
      cln.bspline_clean(dataset['Load'][:-prediction_steps], 
                        genome[loci.l_smooth], 
                        genome[loci.l_zscore], _smoother)
    return clean_data

def bspline_clean_dataset_fast(dataset, genome, loci, prediction_steps):
    """Clean a dataset containing temperatures and loads using cleaning
    parameters from the genome. The dataset is expected to contain NaNs in the
    last *prediction_steps* elements of the Load series"""
    clean_data = dataset.copy()
    clean_data['Temperature'] = cln.bspline_clean_fast(
        dataset['Temperature'], genome[loci.t_smooth], genome[loci.t_zscore])
    clean_data['Load'][:-prediction_steps] = \
      cln.bspline_clean_fast(
        dataset['Load'][:-prediction_steps], 
        genome[loci.l_smooth], genome[loci.l_zscore])
    return clean_data
