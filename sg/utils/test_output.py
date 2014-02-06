"""Test the output utilities."""

import os
import unittest
import copy

import numpy as np
import pandas as pd

import sg.utils.testutils as testutils
from sg.globals import SG_DATA_PATH

from output import * 

_PATH_TO_HERE = os.path.dirname(os.path.abspath(__file__))

class TestOutputMisc(testutils.ArrayTestCase):
    def setUp(self):
        test_file = "_test_series_bc_hydro_no_temperatures_esn_run_8_bc-data.pickle"
        test_path = os.path.join(_PATH_TO_HERE, test_file)
        self.dataset = load_pickled_prediction(test_path)
    
    def tearDown(self):
        pass
    
    def test_split_dataset(self):
        """Test splitting of the test dataset, which is known to have 262
        days."""
        left_lengths = [26, 52, 78, 104, 131, 157, 183, 209, 235]
        for splits in zip(np.arange(0.1, 1, 0.1), left_lengths):
            (left, right) = split_dataset(self.dataset, splits[0])
            self.assertEqual(len(left[1]), splits[1])
            self.assertEqual(len(left[0]), splits[1] * len(self.dataset[1][0]))
            self.assertEqual(len(right[1]), 262 - splits[1])
            self.assertArraysEqual(self.dataset[0], left[0].append(right[0]))
            for (whole_days, split_days) in zip(self.dataset[1], left[1] + right[1]):
                self.assertArraysEqual(whole_days, split_days)
        
    def test_sort_by_validation_error(self):
        """Test sorting by validation error by faking a number of datasets."""
        datasets = [self.dataset]
        # Incrementally append copies with modified target signal
        for i in range(10):
            next_set = copy.deepcopy(datasets[i])
            indices = np.random.random_integers(len(next_set), size=i+1)
            next_set[0][indices] *= 1.2
            datasets.append(next_set)
        # Permute to make sure they are not ordered on entry
        shuffled = [datasets[i] for i in np.random.permutation(len(datasets))]
        val_sorted = sort_data_by_validation_error(shuffled)
        def index_of(left, right):
            for i in range(len(datasets)):
                if np.all(datasets[i][0] == (left[0].append(right[0]))):
                    return i
        for i, (error, (left, right)) in zip(range(len(val_sorted)), val_sorted):
            self.assertEqual(i, index_of(left, right))

    def test_matching_paths(self):
        """Since the output of the matching_paths function depends on the
        contents of the working directory, the tests here may have to be
        updated when files are added to or removed from the relevant
        directory."""
        # Use full path to ensure that it works also when running unit tests
        # from another directory.
        here_wc = os.path.join(_PATH_TO_HERE, "*")
        wildcards = [here_wc, "test", "py$", "output"]
        self.assertEqual(matching_paths(wildcards), 
                         [os.path.join(_PATH_TO_HERE, "test_output.py")])
        wildcards = [here_wc, "__+", ".py$"]
        self.assertEqual(matching_paths(wildcards), 
                         [os.path.join(_PATH_TO_HERE, "__init__.py")])
        there_wc = os.path.join(SG_DATA_PATH, "bchydro", "*")
        wildcards = [there_wc, "area", "[89]"]
        targets = [os.path.join(SG_DATA_PATH, "bchydro", fname) for fname in \
                   ("2008controlareaload.csv", "jandec2009controlareaload.csv")]
        self.assertEqual(matching_paths(wildcards), targets)

        
        

if __name__ == '__main__':
    unittest.main()

