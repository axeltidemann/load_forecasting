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
    def get_model(self, options):
        """Sets up for evolution of the ESN model."""
        alleles = pu.AllelesWithOperators()
        self.add_hindsight(alleles)
        alleles.add(pu.make_int_gene(1, 10, 500, 25), weight=1) # Network size
        alleles.add(pu.make_real_gene(1, 0, 1, 0.05), weight=1) # Leak rate
        alleles.add(pu.make_real_gene(1, 0.1, 0.75, 0.05), weight=1) # Input scaling
        alleles.add(pu.make_real_gene(1, 0, 1, 0.05), weight=1) # Bias scaling
        alleles.add(pu.make_real_gene(1, 0.5, 2, 0.05), weight=1) # Spectral radius
        # We don't want too many seeds per evolutions, but we don't want to
        # always evolve on the same 5 networks either:
        alleles.add(pu.make_choice_gene(
            1, np.random.random_integers(0, 2**16, 5)), weight=1) # Seed
        # Grid optimization showed that for a training length of 336, with
        # other params set based on previous gridopts and operating on the
        # total dataset rather than single AMS'es, optimal ridge was ~5. Scaled
        # thus 5/336=0.015.
        alleles.add(pu.make_choice_gene(
            1, [0.0001/self.max_hindsight_hours]), weight=1) # Scaled ridge
        self.add_cleaning(options, alleles)
        
        loci_list = [ 'hindsight', 'size', 'leak', 'in_scale', 
                      'bias_scale', 'spectral', 'seed', 'ridge' ]
        if not options.no_cleaning:
            loci_list += ['t_smooth', 'l_smooth', 't_zscore', 'l_zscore']
        loci = sg.utils.Enum(*loci_list)

        return Model(self.__class__.__name__, 
                     genes=alleles, 
                     error_func=self._get_error_func(options),
                     transformer=esn.feedback_with_external_input, 
                     loci=loci)

if __name__ == "__main__":
    load_prediction.run(ESNModelCreator())
