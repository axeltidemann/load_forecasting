"""Miscellaneous utility functions for parsng and interpreting the output from
the load prediction programs."""

import cPickle as pickle
import math
import glob
import re
import os
import datetime

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import numpy as np
import Oger
import pandas as pd

from utils import calc_error, plot_target_predictions
import sg.models.load_prediction as load_prediction
import sg.models.load_prediction_esn as load_prediction_esn
import sg.models.esn as esn
from sg.globals import SG_SIM_PATH

def load_pickled_prediction(path):
    """Load a pickled prediction from the given path. The result is a "double
    tuple": (target, (prediction_day_1, ..., prediction_day_n))."""
    with open(path, "r") as f:
        return pickle.load(f)

def plot_pickled_prediction(path):
    plot_target_predictions(*load_pickled_prediction(path))

def split_dataset(dataset, split_point=0.5):
    """Split a (target, predictions) dataset at the indicated split point. Used
    for divinding up into validation and test sets. Assumes all prediction time
    series are of equal length (normally they should be 24 hours).

    Returns a tuple ((target_l, predictions_l), (target_r, predictions_r)).
    """
    (target, predictions) = dataset
    num_validation_days = int(math.floor(len(predictions) * split_point))
    num_validation_steps = num_validation_days * len(predictions[0])
    return ((target[:num_validation_steps], predictions[:num_validation_days]),
            (target[num_validation_steps:], predictions[num_validation_days:]))

def split_and_validate(datasets, split_point=0.5, 
                             error_func=Oger.utils.rmse):
    """Split each dataset at the indicated split point, then calculate the
    error on each part.

    Returns list of tuples, one element per dataset:
    [(error_l, error_r), ((target_l, predictions_l), (target_r, predictions_r))), ...].
    """
    split_sets = [split_dataset(dataset, split_point) for dataset in datasets]
    return [((calc_error(split[0][1], split[0][0], error_func), 
              calc_error(split[1][1], split[1][0], error_func)), 
              split) for split in split_sets]

def sort_data_by_validation_error(datasets, split_point=0.5, 
                             error_func=Oger.utils.rmse):
    """Split, validate and then sort by increasing validation error. 

    Returns the same list of tuples as split_and_validate:
    [(errors), ((target_l, predictions_l), (target_r, predictions_r))), ...].
    """
    with_error = split_and_validate(datasets, split_point, error_func)
    with_error.sort(cmp=lambda x,y: cmp(x[0][0], y[0][0]))
    return with_error

def sort_paths_by_validation_error(paths, split_point=0.5, 
                             error_func=Oger.utils.rmse):
    """Split, validate and then sort by increasing validation error. 

    Returns a list of tuples that contains the path to each dataset, in
    addition to errors and the split dataset:
    [(errors), ((target_l, predictions_l), (target_r, predictions_r)), path), ...].
    """
    datasets = [load_pickled_prediction(path) for path in paths]
    with_error = split_and_validate(datasets, split_point, error_func)
    with_paths = [(z[0][0], z[0][1], z[1]) for z in zip(with_error, paths)]
    with_paths.sort(cmp=lambda x,y: cmp(x[0][0], y[0][0]))
    return with_paths

def matching_paths(*wildcards):
    paths = glob.glob(wildcards[0])
    for wc in wildcards[1:]:
        paths = [path for path in paths if re.search(wc, path) is not None]
    return paths

def test_phase_errors_one(path):
    key = "Error for test at"
    if (os.path.splitext(path)[1] == ".bz2"):
        import bz2
        f = bz2.BZ2File(path, "r")
    else:
        f = open(path, "r")
    lines = [l for l in f]
    f.close()
    (date, time, errors) = zip(*[l[len(key):].split() for l in lines if l[:len(key)] == key])
    time = [t.strip(":") for t in time]
    index=[datetime.datetime.strptime(d + ' ' + t, "%Y-%m-%d %H:%M:%S")
           for (d, t) in zip(date, time)]
    return pd.TimeSeries(data=[float(error) for error in errors], index=index)

def test_phase_errors_many(paths):
    columns = dict()
    for path in paths:
        columns[path] = test_phase_errors_one(path)
    return pd.DataFrame(columns)

def best_genes(path):
    key = "Best genome found shown as alleles:"
    with open(path, "r") as f:
        lines = [l for l in f]
    matches = [l[len(key):].split() for l in lines if l[:len(key)] == key]
    assert(len(matches) == 1)
    match = matches[0]
    match = [float(m.strip('[],')) for m in match]
    return match
    
def best_genes_esn(path):
    genome = best_genes(path)
    for idx in (0, 1, 6):
        genome[idx] = int(genome[idx])
    return genome

def _get_errors_as_dataframe(*wildcards):
    if len(wildcards) == 0:
        paths = matching_paths("*esn*", "_run_[0-9]*_bc-data_no_clean.txt")
    else:
        paths = matching_paths(*wildcards)
    return test_phase_errors_many(paths)

def print_best_pred_per_day():
    runs = ["_run_[0-9]*.txt", 
            "_run_[0-9]*_no_clean.txt", 
            "_run_[0-9]*_total-load.txt", 
            "_run_[0-9]*_total-load_no_clean.txt",
            "_run_[0-9]*_bc-data.txt", 
            "_run_[0-9]*_bc-data_no_clean.txt"]
    for model in ["*esn*", "*arima*", "*wavelet*", "*output*"]:
        for run in runs:
            print "Model %s matching %s." % (model, run)
            frame = _get_errors_as_dataframe(model, run)
            day_by_day_best = np.mean([min(frame.ix[index]) for index in frame.index])
            best_single = np.min([frame[col].mean() for col in frame.columns])
            best_predictors = [np.argmin(frame.ix[day]) for day in frame.index]
            mean_yesterdays_best = np.mean([frame.icol(pred)[day] for (day, pred) in zip(frame.index[1:], best_predictors[:-1])])
            print "Mean of day-by-day best prediction:", day_by_day_best
            print "Overall best single prediction:", best_single
            print "Mean of means:", np.mean([frame[col].mean() for col in frame.columns])
            print "Mean of yesterday's best predictor:", mean_yesterdays_best
            print "---"
            print "Single best - yesterday's best:", best_single - mean_yesterdays_best
            print ""
            print ""

def plot_errors_per_day(*wildcards):
    matrix = _get_errors_as_dataframe(*wildcards).as_matrix().T
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for day in range(matrix.shape[1]):
        x = np.arange(matrix.shape[0])
        ax.bar(x, matrix[x, day], day, zdir='y')

    ax.set_xlabel('Predictor')
    ax.set_ylabel('Day')
    ax.set_zlabel('Error')
    #plt.show()

def plot_error_hist_per_model():
    fig = plt.figure()
    for (model, sub_idx) in zip(["*esn*", "*arima_r*", "*arima_ind*", "*wavelet*", "*CBR*"], 
                                (1, 2, 3, 4, 5)):
        ax = fig.add_subplot(3, 2, sub_idx, projection='3d')
        wildcards = ["_run_[0-9]*_bc-data.txt"]
        model_path = os.path.join(SG_SIM_PATH, "isgt-env-replace-3-of-7", model)
        matrix = _get_errors_as_dataframe(model_path, *wildcards).as_matrix()
        #bins = np.array(range(0, 11000, 100) + [35000])
        bins = np.array(range(0, 2100, 100))
        colors = [cm.flag(i) for i in np.linspace(0, 1, len(bins-1))]
        for pred in range(matrix.shape[1] - 1, -1, -1):
            (hist, _) = np.histogram(matrix[:,pred], bins=bins)
            x = np.array(range(len(bins[:-1])))
            x = np.array(bins[:-1])
            ax.bar(x, hist, pred, zdir='y', color=colors, edgecolor=colors)
            #ax.set_xticks(x + 0.5)
            #ax.set_xticklabels([str(bin) for bin in bins[:-1]])
        ax.set_xlabel('Error with model ' + model)
        ax.set_ylabel('Predictor number')
        ax.set_zlabel('Number of days with error')
    #plt.show()

activities = []
def _state_monitor(states, input, timestep):
    activities[-1].append(np.mean(np.abs(states[timestep+1,:])))
            
def _esn_feedback_with_hook(data, genome, loci, prediction_steps):
    return esn.feedback_with_external_input(
        data, genome, loci, prediction_steps, reservoir_hook=_state_monitor)

def test_genome_store_states(genome, model):
    global activities
    loci = model.loci
    target = pd.TimeSeries()
    predictions = []
    test_iter = model.dataset.test_data_iterator()
    test_number = 1
    i = 0
    activities = []
    for (data_in, data_out) in test_iter():
        cln_data = data_in if model.cleaning_disabled else \
          model.clean_func(data_in, genome, loci, model.day)
        activities.append([])
        model_out = esn.feedback_with_external_input(
            cln_data, genome, loci, model.day, reservoir_hook=_state_monitor)
        error = model.error_func(model_out, data_out)
        print "Error for test at %s: %f" % (str(model_out.index[0]), error)
        test_number += 1
        target = target.combine_first(data_out)
        predictions.append(model_out)
        i += 1
        if i == 30:
            break
    return target, predictions

def print_longterm_activity():
    path = os.path.join(
        SG_SIM_PATH, "isgt-env-replace-3-of-7", 
        "output_esn_run_0_bc-data_no_clean.txt")
    genome = best_genes_esn(path)
    options = load_prediction.get_options()
    options.num_predictions = 1 * 24
    options.bc_data = True
    mc = load_prediction_esn.ESNModelCreator()
    model = mc.get_model(options)
    model.dataset = mc.get_dataset(options)
    model.cleaning_disabled = True
    model.train_and_predict_func = _esn_feedback_with_hook
    model.day = options.num_predictions
    
    (target, predictions) = test_genome_store_states(genome, model)
    plt.figure()
    plot_target_predictions(target, predictions)
    plt.figure()
    for act in activities:
        plt.plot(act)
    plt.axvline(x=1342)
    plt.show()

#print_longterm_activity()

# plot_error_hist_per_model()
# plot_errors_per_day()
# plt.show()

# if __name__ == "__main__":
#     from unittest import main
#     main(module="test_" + __file__[:-3])
    
