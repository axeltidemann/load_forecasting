"""The genemappers map a real-valued gene to an allele."""

import math

from pyevolve import GAllele, G1DList, Consts

class _AlleleMapper():
    def _get_normalized_gene(self, gene_val, gene_range):
        gene_norm = float(gene_val - gene_range[0]) / \
          (gene_range[1] - gene_range[0])
        if gene_norm < 0 or gene_norm > 1:
            raise ValueError("Gene value (%f) outside allowed range (%f - %f)." \
                             % (gene_val, gene_range[0], gene_range[1]))
        return gene_norm

class MappedAlleleRange(GAllele.GAlleleRange, _AlleleMapper): 
    """Subclass of GAllele.GAlleleRange that provides a way of mapping from a
    real-valued gene to a range allele gene."""

    def __init__(self, begin=Consts.CDefRangeMin,
                end=Consts.CDefRangeMax, real=False, scaling='linear'):
        """See superclass for begin, end and real args.  'scaling' scales the
        mapping, and can be linear or log. If scaling is log, then begin < end
        must hold."""
        GAllele.GAlleleRange.__init__(self, begin, end, real)
        self._scaling = scaling
        
    def map_to_allele(self, gene_val, gene_range):
        """Map a gene value in gene_range to the corresponding allele value."""
        if len(self.beginEnd) != 1:
            raise NotImplementedError("The mapper can currently only handle " \
                                      "alleles with a single range.")
        gene_norm = self._get_normalized_gene(gene_val, gene_range)
        beginEnd = self.beginEnd[0]
        to_range = (beginEnd[1] - beginEnd[0])
        if self._scaling == 'log':
            to_range = math.log(1 + to_range)
            mapped_val = beginEnd[0] + math.exp(gene_norm * to_range) - 1
        elif self._scaling == 'linear':
            mapped_val = beginEnd[0] + gene_norm * to_range
        else:
            raise ValueError("Unknown scaling method: %s" % self._scaling)
        if not self.real:
            return int(round(mapped_val))
        return max(beginEnd[0], min(beginEnd[1], mapped_val))

class MappedAlleleList(GAllele.GAlleleList, _AlleleMapper):
    """Subclass of GAllele.GAlleleList that provides a way of mapping from a
    real-valued gene to a list allele gene."""

    def map_to_allele(self, gene_val, gene_range):
        gene_norm = self._get_normalized_gene(gene_val, gene_range)
        to_idx = int(gene_norm * len(self.options))
        # In case gene_norm == 1:
        if to_idx == len(self.options):
            to_idx -= 1
        return self.options[to_idx]


def map_to_alleles(genome):
    """Maps from a (real-valued G1DList) genome to a list of allele genes."""
    alleles = genome.getParam("allele")
    genes = genome[:]
    gene_range = (genome.getParam("rangemin"), genome.getParam("rangemax"))
    return [alleles[i].map_to_allele(genes[i], gene_range)
            for i in range(len(genes))]


    
if __name__ == "__main__":
    from unittest import main
    main(module="test_" + __file__[:-3])
    
