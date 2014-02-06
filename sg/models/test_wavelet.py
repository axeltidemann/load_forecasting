"""
Testing.
Author: Axel Tidemann
"""

from datetime import timedelta as dt
import math
import random

import numpy as np
import Oger, mdp
import matplotlib.pyplot as plt
import scikits.timeseries as ts

import pywt
import sg.utils
from sg.data.sintef.create_full_temp_data import data as read_temperatures
import sg.data.sintef.userloads as ul
import load_prediction

from static import StaticNode

user_id = 55864860

(dataset, test) = load_prediction.prepare_datasets(user_id, False)

#day = 24
#today = random.randint(1000, dataset.shape[0]-day*2)
#today = 4600

#See if we can predict 24 times based on instances, learned from the training set.

data_raw = sg.utils.Normalizer(dataset, axis=0)

data = data_raw.normalized[:2**14,1]

# One year is 365*24 = 8760 datapoints. If we round down to 8192, we will get
# the maximum amount of scales for the decomposition (13), i.e. math.pow(2,13)
# The number of levels/scales determine how far we look back.
level = 4

coeffs = pywt.swt(data, 'haar', level=level)

# Collect coeffecients for training. Aj = 2 is taken from the paper.

Aj = 2

# The first 2^level datapoints cannot be used to predict because of lack of history.
# level+1 because of the smooth array.
x = np.zeros((len(data) - 2**level, (level+1)*Aj))

for i in range(len(x)):
    row = []
    # Collect coefficients for each level. cAn, i.e. the smoothest array.
    for k in range(1, Aj+1):
        row.append(coeffs[-1][0][2**level + i - 2**level*(k-1)])
    # cD, the details.
    for j in range(1, level+1):
        for k in range(1, Aj+1):
            row.append(coeffs[j-1][1][2**level + i - 2**j*(k-1)])
    
    x[i] = np.array(row)

# Target
y = data_raw.normalized[2**level:,1]
y.shape = (len(y), 1)

# Split into train/test sets
x_train = x[:356*24]
y_train = y[:356*24]

print 'Start ESN training...'

# Do 24hr predictions based on single day instances
x_24 = x[::24]
y_24 = np.zeros((len(y)/24,24))
for i in range(len(y_24)):
    y_24[i] = np.transpose(y[i*24:i*24+24])
x_24_train = x_24[:365]
y_24_train = y_24[:365]

flow_24 = mdp.hinet.FlowNode(Oger.nodes.LeakyReservoirNode(input_dim = x_24.shape[1], output_dim = 100, spectral_radius = 0.9) + Oger.nodes.RidgeRegressionNode())
flow_24.train(x_24_train, y_24_train)
flow_24.stop_training()

x_24_test = x_24[365:-1] # There is one more element than the y target due to rounding.
y_24_target = y_24[365:]

y_24_test = flow_24(x_24_test)
print 'NRMSE 24hr:', Oger.utils.nrmse(np.ndarray.flatten(y_24_test), np.ndarray.flatten(y_24_target))

plt.figure()
plt.plot(np.ndarray.flatten(y_24_target), label='24 hr target')
plt.plot(np.ndarray.flatten(y_24_test), label='24 hr test')
plt.legend()

# Test with a classifier ESN
#reservoir = StaticNode(input_dim = x.shape[1], output_dim = 2000, spectral_radius = 0.9)
reservoir = Oger.nodes.LeakyReservoirNode(input_dim = x.shape[1], output_dim = 2000, spectral_radius = 0.9)
readout = Oger.nodes.RidgeRegressionNode()

flow = mdp.hinet.FlowNode(reservoir + readout)
flow.train(x_train, y_train)
flow.stop_training()

x_test = x[356*24:]
y_target = y[356*24:]

y_test = flow(x_test) 
print 'NRMSE:', Oger.utils.nrmse(y_test, y_target)
 
plt.figure()
plt.plot(data, label="Input loads")
plt.plot(coeffs[-1][0], label='Smooth array')
i = 1
for _,cD in coeffs:
    plt.plot(cD, label='cD%i'%i)
    i += 1
plt.legend()

plt.figure()
plt.plot(y_target, label='Target')
plt.plot(y_test, label='Prediction')
plt.legend()
plt.show()

