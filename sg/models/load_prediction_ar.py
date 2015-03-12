'''Evolve a load predictor with BSpline data cleansing and AR/ARIMA predictor.'''

from pyevolve import GAllele
import Oger

import sg.utils
import sg.utils.pyevolve_utils as pu
from model import Model
import arima
import load_cleansing
import load_prediction

class ARModelCreator(load_prediction.ModelCreator):
    def _add_transform_genes(self):
        '''Sets up for evolution of the ARIMA model.'''    
        self._alleles.add(pu.make_int_gene(1, 1, 8*24, 5))
        self._alleles.add(pu.make_int_gene(1, 0, 8*24, 5))
        self._loci_list += ['AR_order']
        self._loci_list += ['EXO_order']

    def _get_transform(self):
        return arima.ar_ga


class ARBitmapModelCreator(load_prediction.ModelCreator):
    def _add_transform_genes(self):
        '''Sets up for evolution of the ARIMA model.'''    
        self._alleles.add(pu.make_bitmap_gene(24*8))
        self._alleles.add(pu.make_bitmap_gene(24*8))
        self._loci_list += ['AR_lags', 'EXO_lags']

    def _get_transform(self):
        return arima.bitmapped_ar_ga


if __name__ == '__main__':
    load_prediction.run(ARModelCreator)
    #load_prediction.run(ARBitmapModelCreator())
