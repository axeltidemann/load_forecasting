# The dumbest form of similar sequence retrieval: sequential scan. To see if there actually
# are any similar sequences.


from datetime import timedelta as dt
import math
import random

import numpy as np
import Oger, mdp
import matplotlib.pyplot as plt
import scikits.timeseries as ts
from rtree import index

import pywt
import sg.utils
from sg.data.sintef.create_full_temp_data import data as read_temperatures
import sg.data.sintef.userloads as ul
import load_prediction

user_id = 55864860

(dataset, test) = load_prediction.prepare_datasets(user_id, False)

window = 256

candidate = sg.utils.Normalizer(dataset[:window+24,1]).normalized

sim = np.argmin([ Oger.utils.nrmse(sg.utils.Normalizer(test[i:i+window,1]).normalized, candidate[:-24]) for i in range(len(test)) if len(test)-i >= window ])
print 'Done.'
plt.plot(candidate, label='target')
most_similar = sg.utils.Normalizer(test[sim:sim+window+24,1]).normalized
plt.plot(most_similar, label='most similar, NRMSE %f' % Oger.utils.nrmse(most_similar[-24:], candidate[-24:]))
plt.legend()
plt.show()

