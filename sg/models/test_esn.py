"""Early attempt. Be patient."""

from datetime import timedelta as dt
import math
import random

import numpy as np
import Oger, mdp
import matplotlib.pyplot as plt
import scikits.timeseries as ts

import esn
import sg.utils
from sg.data.sintef.create_full_temp_data import data as read_temperatures
import sg.data.sintef.userloads as ul
import load_prediction

user_id = 55864860

(dataset, test) = load_prediction.prepare_datasets(user_id, True)

day = 24
today = random.randint(1000, dataset.shape[0]-day*2)
today = 4600


# [len_data, res_size, leak, input, bias, spectral, 
#  seed, ridge, tmp_sm, load_sm]
genome = [336, 500, 0.1, 0.5, 0.5, 0.9, 1000, 0.0001, 10, 10]
genome = [168, 360, 0.1370736370770198, 1.322886484520891, 0.3211445098985698,
          0.9492725784817237, 42979, 0.043436305850920925, 93, 52, 
          1.3053755202564812, 0.5905128791783507]

alleles.loci = sg.utils.enum('hindsight', 'size', 'leak', 'in_scale',
                             'bias_scale', 'spectral', 'seed', 'ridge',
                             't_smooth', 'l_smooth', 't_zscore', 'l_zscore')

test = sg.utils.Normalizer(dataset[today-genome[0]:today+day,:], axis=0)

ytest = esn.feedback_with_external_input(test.normalized, genome, day)

print Oger.utils.nrmse(ytest[-day:], test.normalized[-day:,1])

plt.figure()
plt.plot(test.normalized[:,1], label="Input loads")
offset = len(test.raw) - genome[0] 
plt.plot(range(offset, offset + len(ytest)), ytest, label="Prediction")
plt.show()

# ytest.shape = (len(ytest), 1)
# ytest = test.expand(np.concatenate((ytest, ytest), axis=1))[:,1]

# print sg.utils.mape(ytest[-day:], test.raw[-day:,1])

# out_series = ts.time_series(data=ytest, dates=loads[524:1000].dates)
# sg.utils.plot_time_series([loads[524:1000], out_series],
#                           ["r-", "g-"], ["Loads", "Prediction"])

