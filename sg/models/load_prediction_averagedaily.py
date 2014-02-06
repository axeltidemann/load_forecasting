"""Evolve a load predictor with BSpline data cleansing and predictor as daily or 24-hour averages."""

import numpy as np
import pandas as pd

import load_prediction
import load_prediction_averagehourly as lpah

def daily_average(data, genome, loci, prediction_steps):
    start = -prediction_steps - genome[loci.hindsight]
    end = -prediction_steps
    return pd.TimeSeries(data=data["Load"][start:end].mean(), 
                         index=data.index[-prediction_steps:])


class DailyAverageModelCreator(lpah.HourlyAverageModelCreator):
    def _get_transformer(self):
        return daily_average


if __name__ == "__main__":
    load_prediction.run(DailyAverageModelCreator())
