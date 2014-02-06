"""This is a unit test skeleton, meant to be used as a template for the
boilerplate code when creating a new unit test file."""

import os
import sys
import unittest

import numpy as np

import sg.utils.testutils as testutils

from xxx import * 

_PATH_TO_HERE = os.path.dirname(os.path.abspath(__file__))

class Test(testutils.ArrayTestCase):
    def setUp(self):
        pass
    
    def tearDown(self):
        pass
    
    def test_(self):
        """."""
        pass
        
class Test(unittest.TestCase):
    def setUp(self):
        pass
    
    def tearDown(self):
        pass
    
    def test_(self):
        """."""
        pass


if __name__ == '__main__':
    unittest.main()

# if __name__ == "__main__":
#     from unittest import main
#     main(module="test_" + __file__[:-3])
    
