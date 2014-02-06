"""Evolve a load predictor with BSpline data cleansing and a wavelet predictor."""

import random

from pyevolve import GAllele
import Oger

import sg.utils
import sg.utils.pyevolve_utils as pu
from model import Model
import wavelet
import load_cleansing
import load_prediction

class WaveletHourByHourModelCreator(load_prediction.ModelCreator):
    def get_model(self, options):
        """This is where the models are defined. The models are passed to the
        GA engine for evolution of the optimal set of parameters. Afterwards,
        the models are tested, and performance is measured."""
        
        alleles = pu.AllelesWithOperators()
        alleles.add(pu.make_int_gene(1, 1, 10, 1)) # Scale
        alleles.add(pu.make_choice_gene(1, [2])) # Aj, in the paper 2 gives best results.
        gene = pu.make_choice_gene(1, [i for i in self._hindsight_days])
        alleles.add(gene, weight=1)
        self.add_cleaning(options, alleles)
        
        if options.no_cleaning:
            loci = sg.utils.Enum('scale', 'Aj', 'hindsight')
        else:
            loci = sg.utils.Enum('scale', 'Aj', 'hindsight', 't_smooth', 
                                 'l_smooth', 't_zscore', 'l_zscore')

        return Model(self.__class__.__name__, 
                     genes=alleles, 
                     error_func=self._get_error_func(options),
                     transformer=wavelet.hourbyhour_multiscale_prediction_ga,
                     loci=loci)

if __name__ == "__main__":
    load_prediction.run(WaveletHourByHourModelCreator())
