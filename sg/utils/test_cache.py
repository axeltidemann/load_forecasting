"""Test cache class(es)."""

import os
import unittest

import sg.utils.testutils as testutils

from cache import * 

_PATH_TO_HERE = os.path.dirname(os.path.abspath(__file__))

class TestCache(testutils.ArrayTestCase):
    def setUp(self):
        self.size = 10
        self.cache = ATimeCache(max_entries=self.size)

    def _overfill_cache(self):
        for i in range(self.size*2):
            self.cache[i] = i
        
    def test_max_size(self):
        """Check that the cache size doesn't exceed the given size."""
        self._overfill_cache()
        self.assertEqual(self.size, len(self.cache))

    def test_resize(self):
        self._overfill_cache()
        new_size = self.size - 2
        self.cache.max_entries = new_size
        self.assertEqual(new_size, len(self.cache))
        self._overfill_cache()
        self.assertEqual(new_size, len(self.cache))
        
    def test_store_retrieve(self):
        """Check that storage and retrieval works both on empty and full
        caches."""
        self.cache[12] = 12
        self.assertEqual(self.cache[12], 12)
        self._overfill_cache()
        for i in range(self.size):
            self.cache[-i] = i*12
        for i in range(self.size):
            self.assertEqual(self.cache[-i], i*12)
        
    def test_retrieve_nonexisting(self):
        """Check that retrieval of a non-existing key fails."""
        with self.assertRaises(KeyError):
            x = self.cache[0]
        
    def test_read_refreshes(self):
        """Check that a read refreshes the cache status."""
        for i in range(100):
            self.cache[-12] = 1
            self.cache[i] = i
        with self.assertRaises(KeyError):
            x = self.cache[0]
        self.assertEqual(self.cache[-12], 1)
            
if __name__ == '__main__':
    unittest.main()

