"""Evolve a load predictor with BSpline data cleansing and AR/ARIMA predictor."""

from pyevolve import GAllele
import Oger

import sg.utils
import sg.utils.pyevolve_utils as pu
from model import Model
import arima
import load_cleansing
import load_prediction
import load_prediction_ar

class ARHourByHourModelCreator(load_prediction_ar.ARModelCreator):
    def _get_transform(self):
        return arima.hourbyhour_ar_ga


class ARHourByHourBitmapModelCreator(load_prediction_ar.ARBitmapModelCreator):
    def _get_transform(self):
        return arima.bitmapped_hourbyhour_ar_ga


if __name__ == "__main__":
    load_prediction.run(ARHourByHourModelCreator)
    #load_prediction.run(ARHourByHourBitmapModelCreator())
