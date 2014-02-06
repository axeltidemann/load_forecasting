import unittest
import cPickle as pickle
import os
import tempfile

import numpy as np

import testutils as testutils
from utils import *

_PATH_TO_HERE = os.path.dirname(os.path.abspath(__file__))

class NormalizerTester(testutils.ArrayTestCase):
    def setUp(self):
        self._data = np.array([0., 1., 2., 3., 4., 5.])
        self._norm = np.array([0., 1./5, 2./5, 3./5, 4./5, 1.])
        self._shifted = np.array([-3., -2., -1., 0., 1., 2.])
        self._2d_data = np.array([[0., 1., 2.], [3., 4., 5.]])
        self._2d_norm = np.array([[0., 1./5, 2./5], [3./5, 4./5, 1.]])

    def test_normalize(self):
        normalizer = Normalizer(self._data)
        self.assertArraysEqual(normalizer.normalized, self._norm)
        self.assertArraysEqual(normalizer.normalize(self._data), self._norm)

    def test_normalize_other_data(self):
        normalizer = Normalizer(self._data)
        self.assertArraysAlmostEqual(normalizer.normalize(self._shifted),
                               self._norm - 3./5)

    def test_expand(self):
        normalizer = Normalizer(self._data)
        self.assertArraysEqual(normalizer.expand(self._norm), self._data)

    def test_expand_other_data(self):
        normalizer = Normalizer(self._data)
        shifted_norm = np.array([-0.6, -0.4, -0.2, 0., 0.2, 0.4])
        expanded = normalizer.expand(shifted_norm)
        self.assertArraysEqual(expanded, self._shifted)

    def test_twodim_flatten(self):
        normalizer = Normalizer(self._2d_data)
        self.assertArraysEqual(normalizer.normalized, self._2d_norm)
        self.assertArraysEqual(normalizer.normalize(self._data), self._norm)
                
    def test_twodim_axis_0(self):
        normalizer = Normalizer(self._2d_data, axis=0)
        self.assertArraysEqual(normalizer.normalized,
                               np.array([[0, 0, 0], [1, 1, 1]]))
        self.assertArraysEqual(normalizer.expand([[-1, 0, 2], [1, 0.5, 2]]),
                               np.array([[-3, 1, 8], [3, 2.5, 8]]))

    def test_twodim_axis_1(self):
        normalizer = Normalizer(self._2d_data, axis=1)
        self.assertArraysEqual(normalizer.normalized,
                               np.array([[0, 0.5, 1], [0, 0.5, 1]]))
        

class MiscTester(testutils.ArrayTestCase):
    def _test_enum_values(self, enum):
        self.assertEqual(enum.ZERO, 0)
        self.assertEqual(enum.ONE, 1)
        self.assertEqual(enum.TWO, 2)
        self.assertEqual(enum.NOT_THREE, 4)
        with self.assertRaises(AttributeError) as cm:
            x = enum.NONEXISTING

    def _make_enum(self):
        return Enum('ZERO', 'ONE', 'TWO', NOT_THREE=4)

    def test_enum_create(self):
        numbers = self._make_enum()
        self._test_enum_values(numbers)
        
    def test_pickle_enum(self):
        numbers = self._make_enum()
        storage = tempfile.NamedTemporaryFile(prefix='_test_utils_deleteme_', 
                                              dir=_PATH_TO_HERE)
        pickle.dump(numbers, storage)
        storage.flush()
        storage.seek(0)
        numbers2 = pickle.load(storage)
        self._test_enum_values(numbers2)
        
    def test_indicer_values(self):
        manual = dict((('one', 0),
                       ('two', 1),
                       ('three', 2)))
        indices = indicer('one', 'two', 'three')
        self.assertEqual(indices, manual)

    def test_bound(self):
        self.assertEqual(bound(0, 1, -1), 0)
        self.assertEqual(bound(0, 1, 2), 1)
        self.assertEqual(bound(0, 1, 0.5), 0.5)

    def test_flatten(self):
        lists = ((1, 2), (3, 4), (5, 6))
        flats = [1, 2, 3, 4, 5, 6]
        sublists = (((1, 2), (3, 4)), ((5, 6)))
        subflats = [(1, 2), (3, 4), 5, 6]
        self.assertEqual(flatten(*lists), flats)
        self.assertEqual(flatten(*sublists), subflats)

    def test_safe_flatten(self):
        l = [[1, 2, 3], 9, [[11, 12], [13, 14]], 22, 24]
        shallow = [1, 2, 3, 9, [11, 12], [13, 14], 22, 24]
        deep = [1, 2, 3, 9, 11, 12, 13, 14, 22, 24]
        self.assertEqual(list(safe_shallow_flatten(l)), shallow)
        self.assertEqual(list(safe_deep_flatten(l)), deep)

    def test_diffinv_determined(self):
        x = np.arange(10)
        diffed = np.diff(x)
        self.assertArraysEqual(diffed, np.ones(len(x) - 1))
        self.assertArraysEqual(diffinv(diffed, xi=0), x)
        diffed = np.diff(x, n=2)        
        self.assertArraysEqual(diffinv(diffed, n=2, xi=[0, 1]), x)
        # Difference increases by one each step
        x = np.array([1, 2, 4, 7, 11, 16, 22, 29, 37, 46])
        self.assertArraysEqual(np.diff(x), np.arange(1, len(x)))
        self.assertArraysEqual(
            diffinv(np.diff(x, n=2), n=2, xi=[1, 2]), x)
        
    def test_diffinv_roundtrip(self):
        diffed = np.arange(10)
        for diff_order in range(10):
            xi = np.arange(diff_order)
            x = diffinv(diffed, n=diff_order, xi=xi)
            re_diff = np.diff(x, n=diff_order)
            re_x = diffinv(re_diff, n=diff_order, xi=xi)
            self.assertArraysEqual(diffed, re_diff)
            self.assertArraysEqual(x, re_x)
            
if __name__ == "__main__":
    unittest.main()
