"""Evolve a load predictor with BSpline data cleansing and predictor as daily or 24-hour averages."""

from pyevolve import GAllele
import Oger
import numpy as np
import pandas as pd

import sg.utils
import sg.utils.pyevolve_utils as pu
from model import Model
import load_prediction

def hourly_average(data, genome, loci, prediction_steps):
    assert(prediction_steps == 24)
    start = -prediction_steps - genome[loci.hindsight]
    end = -prediction_steps
    avg_data = pd.DataFrame({"Load": data["Load"][start:end].copy()})
    avg_data["Hour of day"] = [i.hour for i in avg_data.index]
    means = avg_data.groupby(["Hour of day"]).mean()["Load"]
    return pd.TimeSeries(data=means.values, 
                         index=data.index[-prediction_steps:])

class HourlyAverageModelCreator(load_prediction.ModelCreator):
    def _get_transformer(self):
        return hourly_average
        
    def get_model(self, options):
        """Sets up for evolution of the ARIMA model."""
        alleles = pu.AllelesWithOperators()
        self.add_hindsight(alleles)
        alleles.add(pu.make_real_gene(1, 0, 1, 0.1)) # Dummy to make 1D crossover work in Pyevolve
        self.add_cleaning(options, alleles)
        loci_list = ['hindsight', 'crossover_dummy']
        if not options.no_cleaning:
            loci_list += ['t_smooth', 'l_smooth', 't_zscore', 'l_zscore']
        loci = sg.utils.Enum(*loci_list)
        return Model(self.__class__.__name__, 
                     genes=alleles, 
                     error_func=self._get_error_func(options),
                     transformer=self._get_transformer(),
                     loci=loci)

if __name__ == "__main__":
    load_prediction.run(HourlyAverageModelCreator())
