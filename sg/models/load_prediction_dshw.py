"""Evolve a load predictor with BSpline data cleansing and double seasonal Holt
Winters predictor."""

from pyevolve import GAllele
import Oger

import sg.utils
import sg.utils.pyevolve_utils as pu
from model import Model
import arima
import load_cleansing
import load_prediction

class DSHWModelCreator(load_prediction.ModelCreator):
    def get_model(self, options):
        """Sets up for evolution of the DSHW model."""
        alleles = pu.AllelesWithOperators()
        self.add_hindsight(alleles)
        alleles.add(pu.make_real_gene(1, 0, 1, .1), weight=1) # alpha
        alleles.add(pu.make_real_gene(1, 0, 1, .1), weight=1) # beta
        alleles.add(pu.make_real_gene(1, 0, 1, .1), weight=1) # gamma
        alleles.add(pu.make_real_gene(1, 0, 1, .1), weight=1) # omega
        alleles.add(pu.make_real_gene(1, 0, 1, .1), weight=1) # phi
        self.add_cleaning(options, alleles)
        loci_list = ['hindsight', 'alpha', 'beta', 'gamma', 'omega', 'phi']
        if not options.no_cleaning:
            loci_list += ['t_smooth', 'l_smooth', 't_zscore', 'l_zscore']
        loci = sg.utils.Enum(*loci_list)
        transformer=arima.dshw
        return Model(self.__class__.__name__, 
                     genes=alleles, 
                     error_func=self._get_error_func(options),
                     transformer=transformer,
                     loci=loci)

class AutoDSHWModelCreator(load_prediction.ModelCreator):
    def get_model(self, options):
        """Sets up for evolution of the DSHW model."""
        alleles = pu.AllelesWithOperators()
        self.add_hindsight(alleles)
        self.add_cleaning(options, alleles)
        loci_list = ['hindsight']
        if not options.no_cleaning:
            loci_list += ['t_smooth', 'l_smooth', 't_zscore', 'l_zscore']
        loci = sg.utils.Enum(*loci_list)
        transformer=arima.auto_dshw
        return Model(self.__class__.__name__, 
                     genes=alleles, 
                     error_func=self._get_error_func(options),
                     transformer=transformer,
                     loci=loci)

if __name__ == "__main__":
    load_prediction.run(DSHWModelCreator())
    #load_prediction.run(AutoDSHWModelCreator())
