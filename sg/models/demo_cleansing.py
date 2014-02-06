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

def _get_smoother():
    # Set slow_smoother to True in order to see the actual time consumed by the
    # B-spline smoothing operation. If set to False, will use the default
    # smoother where the roughness matrices are cached.
    slow_smoother = True
    if slow_smoother:
        print "Using slow, analytic, non-caching smoother."
        return cln.BSplineAnalyticSmoother
    else:
        print "Using not quite so slow, caching smoother."
        return cln.BSplineSmoother
        
# Load a dataset containing power load history. This set is divided into
# training and test data, we only keep the traning part for now.
dataset, _ = ul.total_experiment_load()

# Set parameters for the B-spline smoother/cleanser
smoothness = 10
zscore = 0.5
# Try smoothing/cleansing different time series lengths
for hindsight_days in [1]:
    # Select data
    num_hours = 24 * hindsight_days
    data = dataset["Load"][-num_hours:].copy()
    # Some output and rough timing
    print "Cleansing %d hours of data with smoothness %.2f, z-score %.2f..." % \
      (num_hours, smoothness, zscore)
    sys.stdout.flush()
    start_time = time.time()
    # This is the part that takes time    
    smoother = _get_smoother()(data, smoothness)
    cleaner = cln.RegressionCleaner(smoother, zscore)
    cleaned, _ = cleaner.get_cleaned_data(
        method=cln.RegressionCleaner.replace_with_bound)
    # Wrap up and plot the result
    end_time = time.time()
    print "Done in %s." % SimpleTimer.period_to_string(start_time, end_time)

    print cleaned
    sys.stdout.flush()
    plt.figure()
    data.plot(style='r', label='Raw load')

    spline = pd.TimeSeries(data=smoother.splev(range(len(cleaned))),
                           index=cleaned.index)
    spline.plot(style='g', label='Smoothing spline')

    # THE SAUSAGE!
    lower, upper = cleaner.get_confidence_interval()
    ax = plt.gca()
    ax.fill_between(cleaned.index, lower, upper, facecolor='g', alpha=0.1)

    cleaned.plot(style='b', label='Cleaned load')
    plt.legend(loc=3)

plt.show()
