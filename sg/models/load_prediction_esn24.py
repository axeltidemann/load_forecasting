"""Evolve a load predictor with BSpline data cleansing and ESN predictor."""

import sg.utils.pyevolve_utils as pu
import esn 
import load_prediction_esn

class ESNHourByHourModelCreator(load_prediction_esn.ESNModelCreator):
    def _add_transform_genes(self):
        """Sets up for evolution of the ESN model."""
        # The 24 hour lags. 
        gene = pu.make_choice_gene(1, [i for i in self._hindsight_days])
        self._alleles.add(gene, weight=1)
        self._loci_list += ['lags']
        ESNModelCreator._add_transform_genes(self)

    def _get_transform(self):
        return esn.hourbyhour_esn_feedback_with_external_input_ga


if __name__ == "__main__":
    load_prediction.run(ESNHourByHourModelCreator)
