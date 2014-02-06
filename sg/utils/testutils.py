"""Unit testing utilities."""

import os
import unittest

import numpy as np

class ArrayTestCase(unittest.TestCase):
    """This class adds some extra assertions in order to simplify working with
    numpy matrices. Uses oddCamelCase in public method names to be consistent
    with the asserts in unittest.TestCase."""
        
    def assertArraysAlmostEqual(self, first, second,
                                places=7, msg=None, delta=None):
        """Assert that two or more numpy arrays have the same shape and contain
        elements that are almost equal."""
        self._generic_multi_array_assert(self.assertAlmostEqual, first, second,
                                         places=places, msg=msg, delta=delta)
        
    def assertArraysEqual(self, first, second, msg=None):
        """Assert that two or more numpy arrays have the same shape and contain
        elements that are equal."""
        self._generic_multi_array_assert(self.assertEqual, first, second,
                                         msg=msg)

    def assertNaNArraysEqual(self, first, second, msg=None):
        """Assert that two or more numpy arrays have the same shape and contain
        elements that are equal. This is the same as assertArraysEqual, but
        with the addition of NaN == NaN."""
        nans_first = np.isnan(first)
        nans_second = np.isnan(second)
        self.assertArraysEqual(nans_first, nans_second, msg=msg)
        self.assertArraysEqual(first[np.where(nans_first == False)[0]], 
                               second[np.where(nans_second == False)[0]], msg=msg)

    def _assert_are_arrays(self, *arrays):
        """Check that all the arrays passed in are actually numpy arrays."""
        for array in arrays:
            self.assertIsInstance(array, np.ndarray)

    def _assert_same_shape_arrays(self, *arrays):
        """Check that all the arrays passed in have the same shape."""
        self.assertGreater(len(arrays), 1)
        shape1 = arrays[0].shape
        for ar in arrays[1:]:
            self.assertEqual(shape1, ar.shape)

    def _assert_are_similar_arrays(self, *arrays):
        """Check that the arrays passed in are "similar": they are all numpy
        arrays with the same shape."""
        self._assert_are_arrays(*arrays)
        self._assert_same_shape_arrays(*arrays)

    def _generic_multi_array_assert(self, assertion, first, second, **kwargs):
        """Generic array assert for several arrays. Checks that the arrays are
        similar, then performs the assertion element-by-element."""
        self._assert_are_similar_arrays(first, second)
        flats = [array.flatten() for array in (first, second)]
        for elements in zip(*flats):
            assertion(*elements, **kwargs)
        
