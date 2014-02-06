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
    def get_model(self, options):
        """Sets up for evolution of the ARIMA model."""
        alleles = pu.AllelesWithOperators()
        self.add_hindsight(alleles)
        alleles.add(pu.make_int_gene(1, 1, 10, 1)) # 'AR' backshift (p)
        alleles.add(pu.make_choice_gene(1, [0, 1, 2])) # 'I' backshift (d) 
        alleles.add(pu.make_choice_gene(1, [1, 2, 3])) # 'MA' backshift (q)
        self.add_cleaning(options, alleles)
        loci_list = ['hindsight', 'AR_order', 'I_order', 'MA_order']
        if not options.no_cleaning:
            loci_list += ['t_smooth', 'l_smooth', 't_zscore', 'l_zscore']
        loci = sg.utils.Enum(*loci_list)
        return Model(self.__class__.__name__, 
                     genes=alleles, 
                     error_func=self._get_error_func(options),
                     transformer=arima.arima_with_weather,
                     loci=loci)

class SeasonalARIMAModelCreator(load_prediction.ModelCreator):
    def get_model(self, options):
        """Sets up for evolution of a seasonal ARIMA model."""
        alleles = pu.AllelesWithOperators()
        self.add_hindsight(alleles)
        alleles.add(pu.make_int_gene(1, 1, 10, 1)) # 'AR' backshift (p)
        alleles.add(pu.make_choice_gene(1, [0, 1, 2])) # 'I' backshift (d) 
        alleles.add(pu.make_choice_gene(1, [1, 2, 3])) # 'MA' backshift (q)
        alleles.add(pu.make_int_gene(1, 1, 10, 1)) # Seasonal 'AR' backshift (p)
        alleles.add(pu.make_choice_gene(1, [0, 1, 2])) # Seasonal 'I' backshift (d) 
        alleles.add(pu.make_choice_gene(1, [1, 2, 3])) # Seasonal 'MA' backshift (q)
        self.add_cleaning(options, alleles)
        loci_list = ['hindsight', 'AR_order', 'I_order', 'MA_order',
                     'ssn_AR_order', 'ssn_I_order', 'ssn_MA_order']
        if not options.no_cleaning:
            loci_list += ['t_smooth', 'l_smooth', 't_zscore', 'l_zscore']
        loci = sg.utils.Enum(*loci_list)
        return Model(self.__class__.__name__, 
                     genes=alleles, 
                     error_func=self._get_error_func(options),
                     transformer=arima.seasonal_arima_with_weather,
                     loci=loci)

class AutoARIMAModelCreator(load_prediction.ModelCreator):
    def get_model(self, options):
        """Sets up for evolution of the ARIMA model."""
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
                     transformer=arima.auto_arima_with_weather,
                     loci=loci)

if __name__ == "__main__":
    load_prediction.run(ARIMAModelCreator())
