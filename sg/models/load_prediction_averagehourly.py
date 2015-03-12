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
    def _add_transform_genes(self):
        """Sets up for evolution of the ARIMA model."""
        self._alleles.add(pu.make_real_gene(1, 0, 1, 0.1)) # Dummy to make 1D crossover work in Pyevolve
        self._loci_list += ['crossover_dummy']

    def _get_transform(self):
        return hourly_average


if __name__ == "__main__":
    load_prediction.run(HourlyAverageModelCreator)
