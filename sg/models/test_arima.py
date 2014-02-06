import os
import unittest

import matplotlib.pyplot as plt
import numpy as np
import Oger
import pandas as pd

#import sg.data.bchydro as bc
import sg.data.sintef.userloads as ul

import sg.utils.testutils as testutils
from sg.utils import calc_error, plot_target_predictions, Enum

from arima import * 

_PATH_TO_HERE = os.path.dirname(os.path.abspath(__file__))

class Test(testutils.ArrayTestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = ul.add_temperatures(ul.total_experiment_load()[0],
                                       ul.experiment_periods()[0])
        
    def setUp(self):
        self.param_lags = np.arange(1, 7*24+1)
        self.periods = range(0, 50*24, 24)
        self.prediction_steps = 24
        self.hindsight = 1000 - 1000 % self.prediction_steps
        self.cos = np.cos

    @classmethod
    def tearDownClass(self):
        plt.show()

    def _get_target(self, data):
        target_start = self.hindsight - self.prediction_steps
        target_end = self.hindsight + self.periods[-1]
        return data[target_start:target_end]

    def _test_ar_data(self, data, lags_2d):
        predictions = []
        for period in self.periods:
            input_data = data[period:self.hindsight+period].copy()
            input_data['Load'][-self.prediction_steps:] = np.nan
            prediction, _ = ar(input_data.values, lags_2d, self.prediction_steps, 
                               out_cols=[input_data.columns.tolist().index('Load')])
            prediction = pd.Series(
                prediction[:,0], index=input_data.index[-self.prediction_steps:])
            predictions.append(prediction)
        target = self._get_target(data)
        rmse = calc_error(predictions, target['Load'], Oger.utils.rmse)
        return (predictions, target, rmse)
        
    def _test_ar_total_load(self, lags_2d):
        return self._test_ar_data(self.data, lags_2d)
    
    def _test_ar_fake(self, lags_2d, std):
        predictions = []
        x_max = max(self.periods) + self.hindsight + self.prediction_steps
        x = np.linspace(1, x_max, x_max)
        load = np.sin(x) + std * np.random.standard_normal(x.shape)
        temp = self.cos(x) + std * np.random.standard_normal(x.shape)
        data = pd.DataFrame({'Temperature':temp, 'Load':load})
        return self._test_ar_data(data, lags_2d)

    #@unittest.skip("")
    def test_plot_full_ar(self):
        for std in [0.1]: #[0.01, 0.1, 0.5]:
            (predictions, target, rmse) = self._test_ar_fake(
                [self.param_lags, []], std)
            plot_target_predictions(target, predictions)
            plt.title('Fake data prediction, sin input only, std=%f. RMSE %f' % (std, rmse))
            (predictions, target, rmse) = self._test_ar_fake(
                [[], self.param_lags], std)
            plot_target_predictions(target, predictions)
            plt.title('Fake data prediction, cos input only, std=%f. RMSE %f' % (std, rmse))
            (predictions, target, rmse) = self._test_ar_fake(
                [self.param_lags, self.param_lags], std)
            plot_target_predictions(target, predictions)
            plt.title('Fake data prediction, sin+cos inputs, std=%f. RMSE %f' % (std, rmse))
            self.cos = np.sin
            (predictions, target, rmse) = self._test_ar_fake(
                [self.param_lags, self.param_lags], std)
            plot_target_predictions(target, predictions)
            plt.title('Fake data prediction, sin+sin inputs, std=%f. RMSE %f' % (std, rmse))
            self.cos = np.cos
            self.hindsight = self.hindsight * 2
            (predictions, target, rmse) = self._test_ar_fake(
                [self.param_lags, []], std)
            plot_target_predictions(target, predictions)
            plt.title('Fake data prediction, double length sin, std=%f. RMSE %f' % (std, rmse))
            self.hindsight = self.hindsight / 2

        # (predictions, target, rmse) = self._test_ar_total_load(
        #     [[], self.param_lags])
        # plot_target_predictions(target, predictions)
        # plt.title("Total load prediction, not using weather. RMSE %f" % rmse)
        # (predictions, target, rmse) = self._test_ar_total_load(
        #     [np.arange(1, 24*3+1), self.param_lags])
        # plot_target_predictions(target, predictions)
        # plt.title("Total load prediction, using 3 days of weather. RMSE %f" % rmse)

        #@unittest.skip("")
    def test_ar_vs_ga(self):
        """Test genome-encoded AR settings vs explicitly specified lags, data
        reduced to hindsight period only, and a direct call to AR."""
        input_data = self.data[-self.hindsight-self.prediction_steps:].copy()
        pred1, _ = ar(input_data.values, 
                      [np.arange(1, 49), np.arange(1, 49)],
                      self.prediction_steps, 
                      out_cols=[input_data.columns.tolist().index('Load')])
        input_data = self.data.copy()
        input_data['Load'][-self.prediction_steps:] = np.nan
        pred2 = ar_ga(input_data,
                      genome=[self.hindsight, 48],
                      loci=Enum('hindsight', 'AR_order'),
                      prediction_steps=self.prediction_steps)
        pred1.shape = (pred1.shape[0], )
        self.assertArraysAlmostEqual(pred2.values, pred1)
        
    def test_hourbyhour_ar_ga(self):
        """Test 24-hour GA versus "manual" extraction of 24
        predictions. Indirectly tests stride functionality of AR."""
        input_data = self.data[-self.hindsight-self.prediction_steps:].copy()
        pred1 = input_data['Load'][-self.prediction_steps:].copy()
        pred1[:] = np.nan
        for hr in range(self.prediction_steps):
            pred1.ix[hr], _ = ar(input_data[hr:len(input_data)-self.prediction_steps+hr+1:self.prediction_steps].values,
                                lags_2d=[np.arange(1, 8), np.arange(1, 8)],
                                prediction_steps=1,
                                out_cols=[input_data.columns.tolist().index('Load')])
        input_data = self.data.copy()
        input_data['Load'][-self.prediction_steps:] = np.nan
        pred2 = hourbyhour_ar_ga(input_data,
                                 genome=[self.hindsight, 7],
                                 loci=Enum('hindsight', 'AR_order'),
                                 prediction_steps=self.prediction_steps)
        pred1.shape = (pred1.shape[0], )
        self.assertArraysAlmostEqual(pred2.values, pred1)
            
    def _test_ar_ga_with_predefined_lags(self, data, genome):
        loci = Enum('hindsight', 'AR_order')
        predictions = []
        for period in self.periods:
            input_data = data[period:self.hindsight+period].copy()
            input_data['Load'][-self.prediction_steps:] = np.nan
            predictions.append(
              ar_ga_with_predefined_lags(
                input_data, genome, loci, self.prediction_steps))
        target = self._get_target()
        rmse = calc_error(predictions, target, Oger.utils.rmse)
        return (predictions, target, rmse)

    @unittest.skip("")
    def test_plot_ar_ga_with_predefined_lags(self):
        data = pd.DataFrame({'Load': self.data, 'Temperature': 0})
        plt.figure()
        plt.title("Prediction with varying number of AR indices")
        for order, fmt in zip([1, 2, 3], ['r-', 'g-', 'c-']):
            (predictions, target, rmse) = \
              self._test_ar_ga_with_predefined_lags(data, [self.hindsight, order])
            plot_target_predictions(target, predictions, fmt)

if __name__ == '__main__':
    unittest.main()

