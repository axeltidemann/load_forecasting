"""Demonstrate the cleansing algorithm on datasets of varying length."""

import sys
import time
from datetime import timedelta as dt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sg.data.sintef.userloads as ul
import spclean as cln
from sg.utils.timer import SimpleTimer
import splines as sp

import array
import ctypes

from ctypes import cdll
from ctypes import c_double

# Load a dataset containing power load history. This set is divided into
# training and test data, we only keep the traning part for now.

def _get_smoother():
    # Set slow_smoother to True in order to see the actual time consumed by the
    # B-spline smoothing operation. If set to False, will use the default
    # smoother where the roughness matrices are cached.
    slow_smoother = True
    if slow_smoother:
        #print "Using slow, analytic, non-caching smoother."
        return cln.BSplineAnalyticSmoother
    else:
        #print "Using not quite so slow, caching smoother."
        return cln.BSplineSmoother

ds_array = 0
kn_array = 0

class BsplineFastSmoother(object):
    def __init__(self, data, smoothness, zscore):
    	#create knot vector
    	knots = sp.get_uniform_knots_from_points(data, degree, knotrange=(0, len(data) - 1))

    	#determine datasize
    	n_data = len(data)
    	n_knot = len(knots)

    	#create a pointer to the dataset
    	ds = np.array(data)
    	ds_type = c_double*n_data
    	ds_array = ds_type(*ds)

    	#create a pointer to the knots
    	kn = np.array(knots)
    	kn_type = c_double*n_knot
    	kn_array = kn_type(*kn)
    
    	#number of threads
    	
    	self._lib = cdll.LoadLibrary('lib_mkl/libspclean.so')
        self.obj = self._lib.Smoother_new(ds_array, n_data, kn_array, n_knot, degree, c_double(smoothness), c_double(zscore))


    def __del__(self):
        self._lib.Smoother_delete(self.obj)
        
    def bsm_cleanData(self):
        return self._lib.bsm_cleanData(self.obj)

    def bsm_smoothedData(self):
        return self._lib.bsm_smoothedData(self.obj)
        

# load data
dataset, _ = ul.total_experiment_load()

# Set parameters for the B-spline smoother/cleanser
degree = 3
smoothness = 100.0
zscore = 1.0
    	
# Try smoothing/cleansing different time series lengths
for hindsight_days in [1]:
    # Select data
    num_hours = 24 * hindsight_days
    data = dataset["Load"][-num_hours:].copy()
    
    #determine datasize   
    n_data = len(data)

    # Some output and rough timing
    #print "Cleansing %d hours of data with smoothness %.2f, z-score %.2f..." % \
    #  (num_hours, smoothness, zscore)
    #sys.stdout.flush()
    start_time = time.time()    
    
    # This is the part that takes time    
    #smoother = _get_smoother()(data, smoothness)
    #cleaner = cln.RegressionCleaner(smoother, zscore)
    #cleaned, _ = cleaner.get_cleaned_data(method=cln.RegressionCleaner.replace_with_bound)

    #call cpp smpline object and get the result
    sm = BsplineFastSmoother(data, smoothness, zscore)
    res = sm.bsm_cleanData()

    # Wrap up and plot the result
    end_time = time.time()
    
    #convert the pointer to nparray
#    ArrayType = ctypes.c_double*n_data
#    array_pointer = ctypes.cast(res, ctypes.POINTER(ArrayType))
#    cleaned_data = np.frombuffer(array_pointer.contents, dtype=np.double)

#    print "Done in %s." % SimpleTimer.period_to_string(start_time, end_time)
#    sys.stdout.flush()

#    res = sm.bsm_smoothedData()
    
    #convert the pointer to nparray
#    ArrayType = ctypes.c_double*n_data
#    array_pointer = ctypes.cast(res, ctypes.POINTER(ArrayType))
#    print "Getting smoothed data..."
#    sys.stdout.flush()
#    smoothed_data = np.frombuffer(array_pointer.contents, dtype=np.double)
    print "Got smoothed data..."
#    sys.stdout.flush()

#    print data
#    print cleaned_data
#    print smoothed_data

#    plt.figure()
#    data.plot(style='b', label="Raw data") 
#    print "Creating time series from smoothed data..."
#    sys.stdout.flush()
#    smoothed_series = pd.TimeSeries(data=smoothed_data, index=data.index)
#    print "Plotting smoothed series..."
#    sys.stdout.flush()
#    smoothed_series.plot(style='r', label="Smoothed data")
#    print "Done plotting smoothed series."
#    sys.stdout.flush()
#    plt.legend()
#    plt.show()
#    data.plot(style='r', label='Raw load')
#    cleaned_data.plot(style='b', label='Cleaned load')
#    spline = pd.TimeSeries(data=smoother.splev(range(len(cleaned))), index=cleaned.index)
#    spline.plot(style='g', label='Smoothing spline')
#    plt.legend(loc=3)
    
#plt.savefig('cfig.pdf')
