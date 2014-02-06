import unittest
import os

import numpy as np
import pandas as pd

import sg.utils.testutils as testutils

from userloads import * 

_PATH_TO_HERE = os.path.dirname(os.path.abspath(__file__))

class TestUserLoads(testutils.ArrayTestCase):
    def setUp(self):
        pass
    
    def tearDown(self):
        pass

    def test_invalid_user(self):
        self.assertRaises(KeyError, tempfeeder_dup().__getitem__, -1)
        
    def test_users_equal(self):
        users_dup = tempfeeder_dup().user_ids
        users_nodup = tempfeeder_nodup().user_ids
        self.assertEqual(users_dup, users_nodup)
        
    def test_num_users(self):
        users_dup = tempfeeder_dup().user_ids
        self.assertEqual(len(users_dup), 2416)

    def test_getitem(self):
        user_id = 83169400
        ul = UserLoads(tempfeeder_dup().path)
        self.assertNotIn(user_id, ul.loads)
        user_loads = ul[user_id]
        self.assertIn(user_id, ul.loads)
        self.assertEqual(len(user_loads), 36602)
        self.assertIs(type(user_loads), pd.DataFrame)
        self.assertNaNArraysEqual(user_loads.ix[14077], np.array([1., np.nan]))
        self.assertArraysEqual(user_loads.ix[-1], np.array([1., 0.]))
        
    def test_get_set_get(self):
        user_id = 29605779
        idx = 15689
        ul = UserLoads(tempfeeder_nodup().path)
        user_loads = ul[user_id]
        user_loads.ix[idx] = np.array([123, 12])
        user_loads.ix[-1] = np.array([124, 14])
        self.assertArraysEqual(ul[user_id].ix[idx], np.array([123, 12]))
        self.assertArraysEqual(ul[user_id].ix[-1], np.array([124, 14]))
        ul.read(user_id)
        self.assertNaNArraysEqual(user_loads.ix[14077], np.array([0., np.nan]))
        self.assertArraysEqual(ul[user_id].ix[-1], np.array([3., 0.]))

    def test_pop(self):
        user_id = 448601
        ul = tempfeeder_dup()
        self.assertNotIn(user_id, ul.loads)
        loads = ul[user_id]
        self.assertIn(user_id, ul.loads)
        ul.pop(user_id)
        self.assertNotIn(user_id, ul.loads)


if __name__ == '__main__':
    unittest.main()
    
