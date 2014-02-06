from datetime import timedelta as dt
import math
import random
import os
import sys
import cPickle as pickle

import numpy as np
import Oger
import mdp, mdp.nodes
import matplotlib.pyplot as plt
import scikits.timeseries as ts

import esn
import sg.utils
from sg.data.sintef.create_full_temp_data import data as read_temperatures
import sg.data.sintef.userloads as ul
import load_prediction

def optimize(postfix):
    # sg.utils.redirect(sys.stdout, "gridopt_output_%s.txt" % postfix)
    
    user_id = 55864860
    (dataset, test) = load_prediction.prepare_datasets(user_id)
    
    day = 24
    freerun = day
    today = 4600
    
    # [len_data, res_size, leak, input, bias, spectral, 
    #  seed, ridge, tmp_sm, load_sm]
    train_hours = 336
    
    datas = \
        [sg.utils.Normalizer(dataset[today-train_hours:today+day-freerun,:], axis=0)
         for today in (1000, 2000, 3000, 4000)]
    
    input_data = []
    for data in datas:
        temps, loads = zip(*data.normalized)
        input_data.append([np.array((temps[24:], loads[:-24], loads[24:])).T])

    reservoir = Oger.nodes.LeakyReservoirNode(output_dim=400,
                                              leak_rate=1,
                                              input_scaling=0.5,
                                              bias_scaling=0.75,
                                              spectral_radius=1,
                                              reset_states=False)
    readout = Oger.nodes.RidgeRegressionNode(ridge_param = 0.001)
    flow = Oger.nodes.FreerunFlow(reservoir + readout,
                                  freerun_steps = freerun,
                                  external_input_range= \
                                  np.array([0, 1]))

    # gridsearch_parameters = {reservoir: {'_instance': range(5), 
    #                                      'spectral_radius': [0.6, 0.8, 1],
    #                                      'input_scaling': [0.1, 0.5, 0.9],
    #                                      'bias_scaling': [0.1, 0.5, 0.9],
    #                                      'leak_rate': [0.1, 0.5, 0.9]},
    #                          readout: {'_instance': range(5),
    #                                    'ridge_param': [0.1, 0.5, 0.9]}}
    
    gridsearch_parameters = {reservoir: {'_instance': range(20)},
                             readout: {'ridge_param': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}}

    print "gridsearch_parameters = " + str(gridsearch_parameters)
    optimizer = Oger.evaluation.Optimizer(gridsearch_parameters, 
                                          Oger.utils.nrmse)

    optimizer.grid_search([[], input_data], flow,
                          cross_validate_function=Oger.evaluation.leave_one_out)

    return (optimizer, reservoir)

def store_optimal_flow(optimizer, postfix):
    optflow = optimizer.get_optimal_flow(verbose=True)

    with open("gridopt_optimal_flow_%s.pickle" % postfix, "w") as f:
        pickle.dump(optflow, f)


if __name__ == "__main__":
    #postfix = str(os.getpid())
    postfix = "deleteme"
    optimizer, reservoir = optimize(postfix)
    store_optimal_flow(optimizer, postfix)
    optimizer.plot_results([(reservoir, '_instance')])

