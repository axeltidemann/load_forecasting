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

class WaveletModelCreator(load_prediction.ModelCreator):
    def _add_transform_genes(self):
        """This is where the models are defined. The models are passed to the
        GA engine for evolution of the optimal set of parameters. Afterwards,
        the models are tested, and performance is measured."""
        self._alleles.add(pu.make_int_gene(1, 1, 10, 1)) # Scale
        self._alleles.add(pu.make_choice_gene(1, [2])) # Aj, in the paper 2 gives best results.
        self._loci_list += ['scale', 'Aj']

    def _get_transform(self):
        #return wavelet.linear_prediction
        #return wavelet.linear_vector
        #return wavelet.vector_multiscale_prediction
        #return wavelet.iterative_multiscale_prediction
        return wavelet.multiscale_prediction


if __name__ == "__main__":
    load_prediction.run(WaveletModelCreator)
