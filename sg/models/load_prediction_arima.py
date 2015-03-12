"""Evolve a load predictor with BSpline data cleansing and AR/ARIMA predictor."""

from pyevolve import GAllele
import Oger

import sg.utils
import sg.utils.pyevolve_utils as pu
from model import Model
import arima
import load_cleansing
import load_prediction

class ARIMAModelCreator(load_prediction.ModelCreator):
    def _add_transform_genes(self):
        """Sets up for evolution of the ARIMA model."""
        self._alleles.add(pu.make_int_gene(1, 1, 10, 1)) # 'AR' backshift (p)
        self._alleles.add(pu.make_choice_gene(1, [0, 1, 2])) # 'I' backshift (d) 
        self._alleles.add(pu.make_choice_gene(1, [1, 2, 3])) # 'MA' backshift (q)
        self._loci_list += ['AR_order', 'I_order', 'MA_order']

    def _get_transform(self):
        return arima.arima_with_weather


class SeasonalARIMAModelCreator(load_prediction.ModelCreator):
    def _add_transform_genes(self):
        """Sets up for evolution of a seasonal ARIMA model."""
        self._alleles.add(pu.make_int_gene(1, 1, 10, 1)) # 'AR' backshift (p)
        self._alleles.add(pu.make_choice_gene(1, [0, 1, 2])) # 'I' backshift (d) 
        self._alleles.add(pu.make_choice_gene(1, [1, 2, 3])) # 'MA' backshift (q)
        self._alleles.add(pu.make_int_gene(1, 1, 10, 1)) # Seasonal 'AR' backshift (p)
        self._alleles.add(pu.make_choice_gene(1, [0, 1, 2])) # Seasonal 'I' backshift (d) 
        self._alleles.add(pu.make_choice_gene(1, [1, 2, 3])) # Seasonal 'MA' backshift (q)
        self._loci_list += ['AR_order', 'I_order', 'MA_order',
                           'ssn_AR_order', 'ssn_I_order', 'ssn_MA_order']

    def _get_transform(self):
        return arima.seasonal_arima_with_weather


class AutoARIMAModelCreator(load_prediction.ModelCreator):
    def _add_transform_genes(self):
        """Sets up for evolution of the ARIMA model."""
        pass

    def _get_transform(self):
        return arima.auto_arima_with_weather


if __name__ == "__main__":
    load_prediction.run(ARIMAModelCreator)
