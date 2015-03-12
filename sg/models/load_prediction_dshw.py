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
    def _add_transform_genes(self):
        """Sets up for evolution of the DSHW model."""
        self._alleles.add(pu.make_real_gene(1, 0, 1, .1), weight=1) # alpha
        self._alleles.add(pu.make_real_gene(1, 0, 1, .1), weight=1) # beta
        self._alleles.add(pu.make_real_gene(1, 0, 1, .1), weight=1) # gamma
        self._alleles.add(pu.make_real_gene(1, 0, 1, .1), weight=1) # omega
        self._alleles.add(pu.make_real_gene(1, 0, 1, .1), weight=1) # phi
        self._loci_list += ['alpha', 'beta', 'gamma', 'omega', 'phi']

    def _get_transform(self):
        return arima.dshw


class AutoDSHWModelCreator(load_prediction.ModelCreator):
    def _add_transform_genes(self):
        """Sets up for evolution of the DSHW model."""
        pass

    def _get_transform(self):
        return arima.auto_dshw


if __name__ == "__main__":
    load_prediction.run(DSHWModelCreator)
    #load_prediction.run(AutoDSHWModelCreator())
