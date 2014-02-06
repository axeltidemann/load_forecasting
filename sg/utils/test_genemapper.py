"""Testing the gene mapper classes."""

import os
import unittest

from pyevolve import GAllele, G1DList

from genemapper import * 

class TestGeneMapper(unittest.TestCase):
    def test_range_map(self):
        allele = MappedAlleleRange(10, 100)
        self.assertEqual(allele.map_to_allele(-1, (-1, 1)), 10)
        self.assertEqual(allele.map_to_allele(-0.5, (-1, 1)), 33)
        self.assertEqual(allele.map_to_allele(0, (-1, 1)), 55)
        self.assertEqual(allele.map_to_allele(1, (-1, 1)), 100)
        self.assertRaises(ValueError, allele.map_to_allele, -2, (-1, 1))
        self.assertRaises(ValueError, allele.map_to_allele, 1.1, (-1, 1))
        
    def test_float_range(self):
        allele = MappedAlleleRange(2, 4, real=True)
        self.assertEqual(allele.map_to_allele(-1, (-1, 1)), 2)
        self.assertEqual(allele.map_to_allele(-0.5, (-1, 1)), 2.5)
        self.assertEqual(allele.map_to_allele(0.4, (-1, 1)), 3.4)
        self.assertEqual(allele.map_to_allele(1, (-1, 1)), 4)
        self.assertRaises(ValueError, allele.map_to_allele, -2, (-1, 1))
        self.assertRaises(ValueError, allele.map_to_allele, 1.1, (-1, 1))

    def test_float_range_log(self):
        allele = MappedAlleleRange(2, 4, real=True, scaling='log')
        self.assertEqual(allele.map_to_allele(-1, (-1, 1)), 2)
        #self.assertEqual(allele.map_to_allele(-0.5, (-1, 1)), 2.5)
        #self.assertEqual(allele.map_to_allele(0.4, (-1, 1)), 3.4)
        self.assertEqual(allele.map_to_allele(1, (-1, 1)), 4)
        self.assertRaises(ValueError, allele.map_to_allele, -2, (-1, 1))
        self.assertRaises(ValueError, allele.map_to_allele, 1.1, (-1, 1))

    def test_long_list(self):
        allele = MappedAlleleList([2, 4, 12])
        self.assertEqual(allele.map_to_allele(-1, (-1, 1)), 2)
        self.assertEqual(allele.map_to_allele(-0.5, (-1, 1)), 2)
        self.assertEqual(allele.map_to_allele(0.3, (-1, 1)), 4)
        self.assertEqual(allele.map_to_allele(1, (-1, 1)), 12)
        self.assertRaises(ValueError, allele.map_to_allele, -2, (-1, 1))
        self.assertRaises(ValueError, allele.map_to_allele, 1.1, (-1, 1))
        
    def test_short_list(self):
        allele = MappedAlleleList([-12])
        self.assertEqual(allele.map_to_allele(-1, (-1, 1)), -12)
        self.assertEqual(allele.map_to_allele(-0.5, (-1, 1)), -12)
        self.assertEqual(allele.map_to_allele(0.3, (-1, 1)), -12)
        self.assertEqual(allele.map_to_allele(1, (-1, 1)), -12)
        self.assertRaises(ValueError, allele.map_to_allele, -2, (-1, 1))
        self.assertRaises(ValueError, allele.map_to_allele, 1.1, (-1, 1))

class TestGenomeMapper(unittest.TestCase):
    def setUp(self):
        self.alleles = GAllele.GAlleles()
        self.alleles.add(MappedAlleleRange(10, 100))
        self.alleles.add(MappedAlleleRange(0, 2, real=True))
        self.alleles.add(MappedAlleleList([2, 4, 12]))
        self.alleles.add(MappedAlleleList([-1]))
        self.genome = G1DList.G1DList(len(self.alleles))
        self.genome.setParams(allele=self.alleles, rangemin=-1, rangemax=1)

    def test_map_genome(self):
        self.genome[:] = [-1, -0.5, 0.4, 1]
        mapped_genome = map_to_alleles(self.genome)
        self.assertEqual(mapped_genome, [10, 0.5, 12, -1])

if __name__ == '__main__':
    unittest.main()

