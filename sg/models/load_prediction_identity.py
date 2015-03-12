"""Evolve a load predictor with BSpline data cleansing and AR/ARIMA predictor."""

from pyevolve import GAllele
import Oger
import pandas as pd

import sg.utils
import sg.utils.pyevolve_utils as pu
from model import Model
import load_cleansing
import load_prediction

def identity_transformer(data, genome, loci, prediction_steps):
    """This prediction model assumes tomorrow will be the same as today."""
    return data["Load"][-prediction_steps*2:-prediction_steps].tshift(prediction_steps)

def null_transformer(data, genome, loci, prediction_steps):
    """This prediction model assumes tomorrow will be entirely flat."""
    return pd.TimeSeries(data=data["Load"][:-prediction_steps].mean(),
                         index=data.index[-prediction_steps:])

class IdentityModelCreator(load_prediction.ModelCreator):
    def _add_transform_genes(self):
        """Sets up for evolution of a system without transformer."""    
        pass

    def _get_transform(self):
        return identity_transformer


if __name__ == "__main__":
    load_prediction.run(IdentityModelCreator)
