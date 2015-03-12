'''Evolve a load predictor with regularized vector AR predictor.'''

import functools

import numpy as np
import pandas as pd

import sg.utils.pyevolve_utils as pu
import load_prediction
import regul_ar
import arima


class LinearRegularizedVectorARModelCreator(load_prediction.ModelCreator):
    def _add_transform_genes(self):
        '''Sets up for evolution of the regularized vector AR model.'''    
        self._alleles.add(pu.make_int_gene(1, 1, 8*24, 5))
        self._alleles.add(pu.make_int_gene(1, 0, 8*24, 5))
        self._add_lambda_gene()
        self._loci_list += ['AR_order']
        self._loci_list += ['EXO_order']
        self._loci_list += ['lambda_cont']

    def _add_lambda_gene(self):
        self._alleles.add(pu.make_real_gene(1, 0, 9, 0.2))

    def _lambda_mapper(self, lc_gene_val):
        return lc_gene_val

    def _transform(self, data, genome, loci, prediction_steps):
        lags_2d = arima.lags_from_order_ga(data, genome, loci)
        lambda_cont = self._lambda_mapper(genome[loci.lambda_cont])
        x_start = max(-len(data), -genome[loci.hindsight] - prediction_steps)
        svp = regul_ar.SmoothVectorARPredictor(
            data[x_start:-prediction_steps].values,
            num_models=prediction_steps,
            lags_2d=lags_2d,
            relative_lags=True,
            add_bias=True,
            out_cols=[data.columns.tolist().index('Load')])
        svp.estimate(lambda_cont=lambda_cont)
        prediction = svp.predict(
            exo_series=np.atleast_2d(data['Temperature'].ix[-prediction_steps:].values).T,
            prediction_steps=prediction_steps)
        return pd.TimeSeries(data=prediction[:,0], index=data[-prediction_steps:].index)

    def _get_transform(self):
        return functools.partial(type(self)._transform, self)

    
class LogRegularizedVectorARModelCreator(LinearRegularizedVectorARModelCreator):
    def _add_lambda_gene(self):
        self._alleles.add(pu.make_int_gene(1, 0, 1e6, 100))

    def _lambda_mapper(self, lc_gene_val):
        return (np.power(10, lc_gene_val) - 1) / 1e3

    
class RegularizedVanillaModelCreator(load_prediction.ModelCreator):
    def __init__(self, *args, **kwargs):
        load_prediction.ModelCreator.__init__(self, *args, **kwargs)
        self._warning_printed = False
        
    def _add_transform_genes(self):
        '''Sets up for evolution of the regularized vanilla benchmark model.'''    
        self._alleles.add(pu.make_int_gene(1, 0, 1e6, 100))
        self._loci_list += ['lambda_cont']

    def _transform(self, data, genome, loci, prediction_steps):
        if not self._warning_printed:
            print 'Hindsight genome ignored, using all available data in Vanilla model.'
            self._warning_printed = True
        svp = regul_ar.VanillaVectorPredictor(data[:-prediction_steps])
        svp.estimate(lambda_cont=genome[loci.lambda_cont])
        return svp.predict(data[-prediction_steps:])
    
    def _get_transform(self):
        return functools.partial(type(self)._transform, self)


if __name__ == '__main__':
    load_prediction.run(LogRegularizedVectorARModelCreator)
