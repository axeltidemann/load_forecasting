"""Evolve a load predictor with BSpline data cleansing and AR/ARIMA predictor."""

from pyevolve import GAllele
import Oger

import sg.utils
import sg.utils.pyevolve_utils as pu
from model import Model
import arima
import load_cleansing
import load_prediction

class ARModelCreator(load_prediction.ModelCreator):
    def get_model(self, options):
        """Sets up for evolution of the ARIMA model."""    
        alleles = pu.AllelesWithOperators()
        alleles.add(pu.make_int_gene(1, 1, 8*24, 5)) # 'AR' backshift (p)
        self.add_hindsight(alleles)
        self.add_cleaning(options, alleles)        
        loci_list = ['AR_order', 'hindsight']
        if not options.no_cleaning:
            loci_list += ['t_smooth', 'l_smooth', 't_zscore', 'l_zscore']
        loci = sg.utils.Enum(*loci_list)    
        return Model(self.__class__.__name__, 
                     genes=alleles, 
                     error_func=self._get_error_func(options),
                     transformer=arima.ar_ga,
                     loci=loci)


class ARBitmapModelCreator(load_prediction.ModelCreator):
    def get_model(self, options):
        """Sets up for evolution of the ARIMA model."""    
        alleles = pu.AllelesWithOperators()
        alleles.add(pu.make_bitmap_gene(24*7)) # AR lags bitmap 
        alleles.add(pu.make_bitmap_gene(24*7)) # AR lags bitmap 
        self.add_hindsight(alleles)
        self.add_cleaning(options, alleles)        
        loci_list = ['lags_temp', 'lags_load', 'hindsight']
        if not options.no_cleaning:
            loci_list += ['t_smooth', 'l_smooth', 't_zscore', 'l_zscore']
        loci = sg.utils.Enum(*loci_list)    
        return Model(self.__class__.__name__, 
                     genes=alleles, 
                     error_func=self._get_error_func(options),
                     transformer=arima.bitmapped_ar_ga,
                     loci=loci)


if __name__ == "__main__":
    load_prediction.run(ARModelCreator())
    #load_prediction.run(ARBitmapModelCreator())
