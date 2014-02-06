### Visualize matrices ###

from datetime import timedelta as dt
import math
import random

import ipdb
import numpy as np
import Oger, mdp
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.gridspec as gridspec
import pandas as pd

import sg.models.esn as esn
import sg.utils
from sg.data.sintef.create_full_temp_data import data as read_temperatures
import sg.data.sintef.userloads as ul
import sg.data.bchydro as bc
import sg.models.load_prediction as load_prediction

options = load_prediction.get_options()
dataset = load_prediction.BCHydroDataset(options, dt(hours=672))

# [len_data, res_size, leak, input, bias, spectral, 
#  seed, ridge, tmp_sm, load_sm]
genome = [672, 194, 0.9507914597451542, 0.23017393420143673, 0.18145624723908402, 1.1091372652108626, 53380, 1.4880952380952382e-07]

l = sg.utils.Enum('hindsight', 'size', 'leak', 'in_scale', 
                     'bias_scale', 'spectral', 'seed', 'ridge')#,
#'t_smooth', 'l_smooth', 't_zscore', 'l_zscore')


# A bit of work is needed to normalize an array that contains NaNs.
prediction_steps = 24
train_iter = dataset.train_data_iterator()
test_iter = dataset.test_data_iterator()

# Choose which day to show.
#(data, data_out) = list(test_iter())[6] # Mayhem ensues.
(data, data_out) = list(test_iter())[37] # Well behaved.
loads = data['Load'].copy()
loads_norm = sg.utils.Normalizer(data['Load'][:-prediction_steps])
loads[:-prediction_steps] = loads_norm.normalized
temps = sg.utils.Normalizer(data['Temperature']).normalized
load_yesterday = loads.copy()
for hr in range(24, len(load_yesterday)):
    load_yesterday[hr] = load_yesterday[hr - 24]
input_data = np.array((temps, load_yesterday, loads)).T
input_data = input_data[-(genome[l.hindsight] + prediction_steps):]
mdp.numx.random.seed(genome[l.seed])
reservoir = Oger.nodes.LeakyReservoirNode(output_dim=genome[l.size],
                                          leak_rate=genome[l.leak],
                                          input_scaling=genome[l.in_scale],
                                          bias_scaling=genome[l.bias_scale],
                                          spectral_radius=genome[l.spectral],
                                          reset_states=False)

collector = []
def my_hook(states, input, timestep):
    states[timestep + 1, :] = (1 - reservoir.leak_rate) * states[timestep, :] + reservoir.leak_rate * states[timestep + 1, :]
    global collector
    collector.append(states)

reservoir._post_update_hook = my_hook
readout = Oger.nodes.RidgeRegressionNode(
  ridge_param=genome[l.ridge]*genome[l.hindsight])
# NB: 'n' freerun_steps produces only 'n-1' outputs that are not teacher
# forced, we need prediction_steps+1 freerun steps.
flow = Oger.nodes.FreerunFlow(reservoir + readout,
                              freerun_steps=prediction_steps + 1, # NB!
                              external_input_range=[0, 1])
flow.train([[], [[input_data[:-prediction_steps]]]])
flow_output = flow(input_data)
prediction = loads_norm.expand(flow_output[-prediction_steps:, -1])

retrospect = 24*7
mega_collection = np.array([ collector[-26][i,:] for i in range(-retrospect,0) ] + [ collector[i][1,:] for i in range(-24,0) ]).T

cdict = { 'red': ((0.0, 0.0, 1.0),
                  (0.5, 1.0, 1.0),
                  (1.0, 0.0, 0.0)),
          'green': ((0.0, 0.0, 0.0),
                    (0.5, 1.0, 1.0),
                    (1.0, 0.0, 0.0)),
          'blue': ((0.0, 0.0, 0.0),
                   (0.5, 1.0, 1.0),
                   (1.0, 1.0, 0.0))}
          
my_cmap = matplotlib.colors.LinearSegmentedColormap('my_colormap',cdict,256)          

target = np.concatenate((loads_norm.expand(input_data[-retrospect-24:-24,-1]), data_out))
for i in range(mega_collection.shape[1]):
    fig = plt.figure()
    gs = gridspec.GridSpec(2,1,
                           width_ratios=[1,1],
                           height_ratios=[6,1])
    ax = plt.subplot(gs[0,0], polar=True)
    ax.set_ylim(0,1.1)

    for j in range(mega_collection.shape[0]):
        for k in range(mega_collection.shape[0]):
            connection = reservoir.w[j,k]*mega_collection[j,i]
            plt.polar([ 2.*np.pi*j/mega_collection.shape[0], 2.*np.pi*k/mega_collection.shape[0] ],
                      [ mega_collection[j,i], mega_collection[k,i] ],
                      alpha=abs(connection/np.max(np.abs(reservoir.w))),
                      linewidth=0.01,
                      color = 'r' if connection < 0 else 'b',
                      zorder=1)

    plt.scatter( [ 2.*np.pi*j/mega_collection.shape[0] for j in range(mega_collection.shape[0]) ], mega_collection[:,i],
                 zorder=2,
                 c = mega_collection[:,i],
                 cmap = my_cmap,
                 vmin = -1, vmax = 1)
    ax = plt.subplot(gs[1,0])
    beans = i - retrospect
    if beans >= 0:
        x = range(retrospect,retrospect + beans + 1)
        ax.plot(x, prediction[:beans+1], label='Prediction')
    ax.plot(target[:i+1], 'r', label='Target')
    ax.set_xlim(0,mega_collection.shape[1])
    ax.axvline(x=retrospect, color='k')
    #ax.get_yaxis().set_visible(False)
    yticks = ax.get_yticks()
    ax.set_yticks((yticks[0], yticks[-1]))
    ax.set_yticklabels(("% 1.2f" % (yticks[0]/10000), "% 1.2f" % (yticks[-1]/10000)))
    ax.set_xlabel('Day %d, hour %d' % (i/24 - mega_collection.shape[0]/24 + 1, i%24))
    plt.savefig('%03d_wellbehaved.png' % i, dpi=150, bbox_inches='tight')
    plt.close()
    print 'Plotted day', i

fig = plt.figure()
ax = fig.add_subplot(211)
im = ax.imshow(mega_collection, aspect='auto')
#plt.colorbar(im, use_gridspec=True)
ax = fig.add_subplot(212)
ax.plot(range(retrospect,retrospect + 24), prediction, label='Prediction')
ax.plot(target, 'r', label='Target')
ax.legend(loc=3)
ax.set_xlim(0,mega_collection.shape[1]-1)
ax.axvline(x=retrospect, color='k')
plt.show()

# plt.figure()
# plt.plot(test.normalized[:,1], label="Input loads")
# offset = len(test.raw) - genome[0] 
# plt.plot(range(offset, offset + len(ytest)), ytest, label="Prediction")
# plt.show()
