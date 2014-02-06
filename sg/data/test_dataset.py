
import datetime
import unittest

import numpy as np
import pandas as pd

import sg.utils.testutils as testutils
import sg.data.bchydro as bchydro
from dataset import *

class DatasetTester(testutils.ArrayTestCase):
    def setUp(self):
        month_index = pd.period_range(start='2005-01-01', periods=12, freq='M')
        day_index = pd.period_range(start='2005-01-01', periods=365, freq='D')
        hour_index = pd.period_range(start='2005-01-01', periods=365*24, freq='H')
        self.months = pd.Series(np.arange(12), index=month_index)
        self.days = pd.Series(np.arange(365), index=day_index)
        self.hours = pd.Series(np.arange(365*24), index=hour_index)
        self.period = datetime.timedelta(days = 9)
        self.month_data = Dataset(self.months, self.period)
        self.day_data = Dataset(self.days, self.period)
        self.hour_data = Dataset(self.hours, self.period,
                                 datetime.timedelta(hours = 12))

    def test_calculate_period(self):
        self.assertEqual(self.month_data._period_length, 1)
        self.assertEqual(self.day_data._period_length, 9)
        self.assertEqual(self.hour_data._period_length, 9 * 24)

    def test_number_of_periods(self):
        self.assertEqual(self.month_data.num_periods, 12)
        self.assertEqual(self.day_data.num_periods, 357)
        self.assertEqual(self.hour_data.num_periods, 356 * 2)

    def test_get_last_period(self):
        last_month = self.month_data.get_period(self.month_data.num_periods - 1)
        self.assertEqual(len(last_month), 1)
        self.assertEqual(last_month[0], 11)
        last_days = self.day_data.get_period(self.day_data.num_periods - 1)
        self.assertEqual(len(last_days), 9)
        self.assertArraysEqual(last_days, self.days[-9:])
        hour_periods = self.hour_data.num_periods
        last_hours = self.hour_data.get_period(hour_periods - 1)
        self.assertEqual(len(last_hours), 9 * 24)
        self.assertArraysEqual(last_hours, self.hours[-9*24-12:-12])

    def test_get_first_period(self):
        self.assertArraysEqual(self.month_data.get_period(0), self.months[0:1])
        self.assertArraysEqual(self.day_data.get_period(0), self.days[0:9])
        self.assertArraysEqual(self.hour_data.get_period(0), self.hours[0:9*24])
        
    def test_get_second_period(self):
        self.assertArraysEqual(self.month_data.get_period(1), self.months[1:2])
        self.assertArraysEqual(self.day_data.get_period(1), self.days[1:10])
        self.assertArraysEqual(self.hour_data.get_period(1),
                               self.hours[12:9*24+12])

    def test_get_random_period(self):
        for i in range(100):
            (ts, number) = self.month_data.get_random_period(True)
            self.assertArraysEqual(ts, self.months[number:number+1])
            (ts, number) = self.day_data.get_random_period(True)
            self.assertArraysEqual(ts, self.days[number:number+9])
            (ts, number) = self.hour_data.get_random_period(True)
            index = self.hour_data.index_of(number)
            self.assertEqual(index, number * 12)
            self.assertArraysEqual(ts, self.hours[index:index+9*24])

class MiscTester(testutils.ArrayTestCase):
    def test_remove_one_outlier(self):
        dataset = np.array([0, 1, 2, 0, 3, 4, 0, 5])
        remove_outlier_set_previous(dataset, outlier_val=0)
        self.assertArraysEqual(dataset, np.array([0, 1, 2, 2, 3, 4, 4, 5]))

    def test_remove_consecutive_outliers(self):
        dataset = np.array([0, 1, 0, 0, 0, 4, 0, 5])
        retset = remove_outlier_set_previous(dataset)
        self.assertArraysEqual(retset, np.array([0, 1, 1, 1, 1, 4, 4, 5]))

    def test_remove_other_outliers(self):
        dataset = np.array([0, 1, 2, 0, 3, 4, 0, 5])
        remove_outlier_set_previous(dataset, outlier_val=2)
        self.assertArraysEqual(dataset, np.array([0, 1, 1, 0, 3, 4, 0, 5]))

if __name__ == "__main__":
    unittest.main()
