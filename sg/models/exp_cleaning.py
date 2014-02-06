
from datetime import timedelta as dt

import matplotlib.pyplot as plt
import numpy as np

import sg.data.bchydro as bc
import spclean as cln

def clean_all_bc_data(period_days=7, step_days=6, 
                      smoothnesses=(0.1, 1, 3, 6, 10, 100)):
    """Clean the entire BC Hydro dataset period by period with the given
    smoothnesses. By default clean a week at a time with 1 day overlap (step
    length 6 days).

    Returns a dictionary keyed on the smoothness, where the values are lists of
    tuples, each tuple consisting of the period number and the outlier indices
    for all periods with outliers."""

    dataset = bc.Dataset(period=dt(days=period_days), 
                         step_length=dt(days=step_days))
    outliers_at = dict()
    for smoothness in smoothnesses:
        outliers_at[smoothness] = cln.clean_entire_dataset(dataset, smoothness)
        print "cleaned with smoothness", smoothness
    return (dataset, outliers_at)

def clean_and_process_bc_data(period_days=7, step_days=6, 
                      smoothnesses=(0.1, 1, 3, 6, 10, 100)):
    """Clean BC data using clean_all_bc_data. Then plot the data to show the
    distribution of outliers per period and smoothness."""
    data, outliers_at = clean_all_bc_data(period_days, step_days, smoothnesses)
    x = np.arange(data.num_periods)
    y_at = dict()
    for (smoothness, outliers) in outliers_at.iteritems():
        y = np.zeros(data.num_periods)
        for (period, outlier_indices) in outliers:
            y[period] = len(outlier_indices)
        y_at[smoothness] = y
    plt.figure()
    plt.hold(True)
    plt.title("Number of cleaned points for various smoothnesses")
    axes = plt.gcf().gca()
    for (smoothness, y) in y_at.iteritems():
        plt.figure()
        plt.plot(x, y, 'x')
        plt.title("Number of cleaned points for smoothness %.2f" % smoothness)
        axes.plot(x, y, 'x', label="Smoothness %.2f" % smoothness)
        plt.figure()
        plt.hist(y)
        plt.title("Histogram of number of cleaned points for " \
                      "smoothness %.2f" % smoothness)
    axes.legend()

def show_max_cleaning():
    week = 264
    dataset = bc.Dataset(period=dt(days=7), step_length=dt(days=6))
    period = dataset.get_period(week)
    smoother = cln.BSplineSmoother(period, smoothness=3)
    cleaner = cln.RegressionCleaner(smoother, zscore=0.67)
    (clean_data, outliers) = cleaner.get_cleaned_data(
        cln.RegressionCleaner.replace_with_estimate)
    plt.figure()
    plt.hold(True)
    n = len(smoother.dataset)
    knots = smoother.knots
    t = np.linspace(knots[0], knots[-1], n * 25)
    y = smoother.splev(t)
    plt.hold(True)
    plt.plot(t, y)
    x = np.linspace(knots[0], knots[-1], n)
    plt.plot(x, smoother.dataset, 'mx')
    (lower, upper) = cleaner.get_confidence_interval()

    plt.plot(lower, 'g-')
    plt.plot(upper, 'g-')
    if len(outliers) > 0:        
        print "Drawing %d outliers." % len(outliers)
        plt.plot(outliers, clean_data[outliers], 'r*', label="Cleaned data")
    else:
        print "No outliers!"
    
if __name__ == "__main__":
    show_max_cleaning()
    print "Done cleaning, showing plot"
    plt.show()
