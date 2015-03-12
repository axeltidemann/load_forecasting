"""Evolve a load predictor with BSpline data cleansing and ESN predictor."""

import numpy as np
import Oger


import sg.utils
import sg.utils.pyevolve_utils as pu
from model import Model
import esn 
import load_cleansing
import load_prediction

class ESNModelCreator(load_prediction.ModelCreator):
    def _add_transform_genes(self):
        """Sets up for evolution of the ESN model."""
        self._alleles.add(pu.make_int_gene(1, 10, 500, 25), weight=1) # Network size
        self._alleles.add(pu.make_real_gene(1, 0, 1, 0.05), weight=1) # Leak rate
        self._alleles.add(pu.make_real_gene(1, 0.1, 0.75, 0.05), weight=1) # Input scaling
        self._alleles.add(pu.make_real_gene(1, 0, 1, 0.05), weight=1) # Bias scaling
        self._alleles.add(pu.make_real_gene(1, 0.5, 2, 0.05), weight=1) # Spectral radius
        # We don't want too many seeds per evolutions, but we don't want to
        # always evolve on the same 5 networks either:
        self._alleles.add(pu.make_choice_gene(
            1, np.random.random_integers(0, 2**16, 5)), weight=1) # Seed
        # Grid optimization showed that for a training length of 336, with
        # other params set based on previous gridopts and operating on the
        # total dataset rather than single AMS'es, optimal ridge was ~5. Scaled
        # thus 5/336=0.015.
        self._alleles.add(pu.make_choice_gene(
            1, [0.0001/self._max_hindsight_hours]), weight=1) # Scaled ridge
        self._loci_list += ['size', 'leak', 'in_scale', 
                      'bias_scale', 'spectral', 'seed', 'ridge' ]

    def _get_transform(self):
        return esn.feedback_with_external_input


if __name__ == "__main__":
    load_prediction.run(ESNModelCreator)
