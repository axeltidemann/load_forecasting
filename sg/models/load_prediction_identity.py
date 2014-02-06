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
    def get_model(self, options):
        """Sets up for evolution of a system without transformer."""    
        alleles = pu.AllelesWithOperators()
        self.add_hindsight(alleles)
        self.add_cleaning(options, alleles)        
        loci_list = ['hindsight']
        if not options.no_cleaning:
            loci_list += ['t_smooth', 'l_smooth', 't_zscore', 'l_zscore']
        loci = sg.utils.Enum(*loci_list)    
        return Model(self.__class__.__name__, 
                     genes=alleles, 
                     error_func=self._get_error_func(options),
                     transformer=identity_transformer,
                     loci=loci)


if __name__ == "__main__":
    load_prediction.run(IdentityModelCreator())
