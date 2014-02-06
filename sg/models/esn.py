import sys

import mdp, Oger
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sg.utils.timer import SimpleTimer
from sg.utils import Normalizer, hook_chainer

def hourbyhour_esn_feedback_with_external_input_ga(data, genome, loci, prediction_steps):
    prediction = []
    for i in range(prediction_steps):
        subset = data.ix[i::prediction_steps]
        ss1 = subset.ix[:-1]
        subset_rep = pd.concat([ss1.shift(periods=-len(ss1)*i, freq='D') for i in range(100,1,-1)] + [subset])
        prediction.append(feedback_with_external_input(subset_rep, genome, loci, 1, shift=1))
    return pd.concat(prediction)

def feedback_with_external_input(data, genome, loci, prediction_steps, 
                                 spinup=0, reservoir_hook=None, shift=24):
    l = loci
    # First SHIFT value not usable since we feed the ESN with yesterday's load.
    assert(len(data) >= genome[l.hindsight] + shift + prediction_steps)
    # A bit of work is needed to normalize an array that contains NaNs.
    loads = data['Load'].copy()
    loads_norm = Normalizer(data['Load'][:-prediction_steps])
    loads[:-prediction_steps] = loads_norm.normalized
    temps = Normalizer(data['Temperature']).normalized
    load_yesterday = np.roll(np.array(loads), shift=shift)
    input_data = np.array((temps, load_yesterday, loads)).T
    input_data = input_data[-(genome[l.hindsight] + prediction_steps):]
    mdp.numx.random.seed(genome[l.seed])
    reservoir = Oger.nodes.LeakyReservoirNode(output_dim=genome[l.size],
                                              leak_rate=genome[l.leak],
                                              input_scaling=genome[l.in_scale],
                                              bias_scaling=genome[l.bias_scale],
                                              spectral_radius=genome[l.spectral],
                                              reset_states=False)
    if reservoir_hook is not None:
        reservoir._post_update_hook = hook_chainer(
            reservoir._post_update_hook, reservoir_hook)
    readout = Oger.nodes.RidgeRegressionNode(
      ridge_param=genome[l.ridge]*genome[l.hindsight])
            
    # NB: 'n' freerun_steps will return a dataset where only the last 'n-1'
    # outputs are not teacher forced. The last freerun output will be contained
    # in flow.fb_value. In order to produce 24 predictions, one can specify 25
    # freerun steps. However, if the idea is to continue the flow execution in
    # subsequent steps (i.e. prediction_steps > 24), yesterday's prediction
    # must be manually copied in and used as the observed load when predicting
    # today. In this case, setting freerun_steps to 25 will not work.
    flow = Oger.nodes.FreerunFlow(reservoir + readout,
                                  freerun_steps=1,
                                  external_input_range=[0, 1])
    
    flow.train([[], [[input_data[:-prediction_steps]]]])

    idx = len(input_data) - prediction_steps
    flow(input_data[:idx]) # Warmup
    input_data[idx, 2] = flow.fb_value
    for i in range(prediction_steps - 1):
        input_data[idx+i, 1] = input_data[idx+i-shift, 2]
        flow(np.atleast_2d(input_data[idx+i, :]))
        input_data[idx+i+1, 2] = flow.fb_value

    prediction = loads_norm.expand(input_data[-prediction_steps:, -1])
    pred_as_series = pd.TimeSeries(data=prediction,
                                   index=data.index[-prediction_steps:])
    return pred_as_series

def feedback(data, genome, loci, prediction_steps, spinup=0):
    l = loci
    mdp.numx.random.seed(genome[l.seed])
    reservoir = Oger.nodes.LeakyReservoirNode(output_dim=genome[l.size],
                                              leak_rate=genome[l.leak],
                                              input_scaling=genome[l.in_scale],
                                              bias_scaling=genome[l.bias_scale],
                                              spectral_radius=genome[l.spectral],
                                              reset_states=False)
    readout = Oger.nodes.RidgeRegressionNode(ridge_param=genome[l.ridge]*genome[l.hindsight])
    flow = Oger.nodes.FreerunFlow(reservoir + readout,
                                  freerun_steps=prediction_steps)

    (temps, loads) = (data[:,0], data[:,1])
    input_data = np.array((temps[:-1], loads[:-1], loads[1:])).T
    flow.train([[], [[input_data[spinup:-prediction_steps]]]])
    return flow(input_data)[:,-1]


# For the time being, these conform to a different protocol, time-wise -
# commented out to highlight this issue.
    
# def feedforward(data, genome):
#     mdp.numx.random.seed(genome[6])
#     x = data[:-genome.getParam('day')]
#     y = data[genome.getParam('day'):]

#     reservoir = Oger.nodes.LeakyReservoirNode(output_dim=genome[1],
#                                               leak_rate=genome[2],
#                                               input_scaling=genome[3],
#                                               bias_scaling=genome[4],
#                                               spectral_radius=genome[5],
#                                               reset_states=False)
#     readout = Oger.nodes.RidgeRegressionNode(ridge_param=0.0001)
#     flow = mdp.hinet.FlowNode(reservoir + readout)
#     flow.train(x,y)
#     flow.stop_training()
#     return flow(y)

# def freerun(data, genome):
#     mdp.numx.random.seed(genome[6])
#     freerun = genome.getParam('day')
#     reservoir = Oger.nodes.LeakyReservoirNode(output_dim=genome[1],
#                                               leak_rate=genome[2],
#                                               input_scaling=genome[3],
#                                               bias_scaling=genome[4],
#                                               spectral_radius=genome[5],
#                                               reset_states=False)
#     readout = Oger.nodes.RidgeRegressionNode(ridge_param = 0.0001)
#     flow = Oger.nodes.FreerunFlow(reservoir + readout,
#                                   freerun_steps = freerun)
#     flow.train([[], [[data]]])

#     # We must add data points for the function to predict a day into
#     # the future - it accepts an input vector that includes the
#     # freerun period.
#     np.resize(data, data.shape[0] + freerun)
#     return flow(data)

if __name__ == "__main__":
    from unittest import main
    main(module="test_" + __file__[:-3])
    
