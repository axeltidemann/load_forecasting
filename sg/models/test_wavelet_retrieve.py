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
from rtree import index

import pywt
import sg.utils
from sg.data.sintef.create_full_temp_data import data as read_temperatures
import sg.data.sintef.userloads as ul
import load_prediction

user_id = 55864860

(dataset, test) = load_prediction.prepare_datasets(user_id, False)

#See if we can predict 24 times based on instances, learned from the training set.
window = 512

hours = [ sg.utils.Normalizer(dataset[i:i+window,1]).normalized for i in range(len(dataset)) if len(dataset)-i >= window+24 ]

coeffs = [ pywt.wavedec(segment,'haar') for segment in hours ]

# Grow tree
p = index.Property()
p.dimension = 20
idx = index.Index(properties=p)

i = 0
for c in coeffs:
    key = [item for sublist in c for item in sublist ][:p.dimension]
    idx.insert(i, tuple(key))
    i+=1

def retrieve(query):
    query_key = [item for sublist in pywt.wavedec(query[:-24],'haar') for item in sublist ][:p.dimension]
    results = list(idx.nearest(tuple(query_key), 3))
    print results
    plt.plot(query, label='query')
    for i in range(len(results)):
        candidate = sg.utils.Normalizer(dataset[results[i]:results[i]+window+24,1]).normalized
        plt.plot(candidate, label='candidate %i, NRMSE %f'%(i,Oger.utils.nrmse(candidate[-24:], query[-24:])))
    plt.axvline(x=window, color='r', linewidth=1)
    plt.legend()
    plt.show()
    

# Try to find 5 random points in the training dataset
for test_point in np.random.permutation(range(len(dataset) - window - 24))[:5]:
    retrieve(sg.utils.Normalizer(dataset[test_point:test_point+window+24,1]).normalized)

# Try 5 different random points in the test dataset
for test_point in np.random.permutation(range(len(test) - window - 24))[:5]:
    retrieve(sg.utils.Normalizer(test[test_point:test_point+window+24,1]).normalized)
