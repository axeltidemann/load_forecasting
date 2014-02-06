import os
import sys
import unittest

import numpy as np
import matplotlib.pyplot as plt
from pyevolve import G1DList, G2DList, G1DBinaryString, GAllele
from pyevolve import GSimpleGA
from pyevolve import Initializators, Mutators, Crossovers
from pyevolve import Selectors
from pyevolve import Consts
from pyevolve import DBAdapters
from pyevolve import Scaling

import sg.utils.testutils as testutils

from pyevolve_utils import * 

_PATH_TO_HERE = os.path.dirname(os.path.abspath(__file__))

class OperatorWrapper(object):
    def __init__(self, operator, description, num_runs=1):
        self._operator = operator
        self._description = description
        self._num_runs = num_runs
        self._run_count = 0
        
    def __call__(self, genome, **args):
        if self._run_count < self._num_runs:
            print "============= Running wrapped operator: "
            print "-------------", self._description
            print "------------- print", self._run_count, "of", self._num_runs
            print "Genome ahead of call:"
            print genome
            print "args:"
            print args
        retval = self._operator(genome, **args)
        if self._run_count < self._num_runs:
            self._run_count += 1
            print "Genome after call:"
            print genome
            print "Returned:"
            print retval
        return retval

    @property
    def func_name(self):
        return self._description + " wrapper"

    @property
    def func_doc(self):
        return ''
    
def realval_fitness(target, actual):
    return 10 / (1 + ((target - actual)/target)**2)

#bitmap_target = [0, 1, 0, 0, 1, 1, 0, 1, 0, 0]
bitmap_target = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
chooser_keys = [12, 15, 16, 22, 'Bingo!', 'Jackpot!']
chooser_scores_1 = [1, 2, 3, 4, 900, 0]
chooser_scores_2 = [1, 2, 3, 4, 0, 900]
chooser_1 = dict(zip(chooser_keys, chooser_scores_1))
chooser_2 = dict(zip(chooser_keys, chooser_scores_2))

def eval_func(chromosome):
    score = 0.0
    set_to_1, bitmap, set_to_1000, chosen = chromosome
    score += realval_fitness(1, set_to_1[0])
    for (bit, target) in zip(bitmap, bitmap_target):
        score += 1 if bit == target else 0
    score += realval_fitness(1000, set_to_1000[0])
    score += chooser_1[chosen[0]]
    score += chooser_2[chosen[1]]
    return score

def verbose_eval(chromosome):
    set_to_1, bitmap, set_to_1000, chosen = chromosome
    score = realval_fitness(1, set_to_1[0])
    print "First gene: \t Value: %f, target %f \t Score: %f" \
      % (set_to_1[0], 1, score)
    score = 0
    for (bit, target) in zip(bitmap, bitmap_target):
        score += 1 if bit == target else 0
    print "Second gene: \t Value: %s, target %s \t Score: %f" \
      % (bitmap.getBinary(), str(bitmap_target), score)
    score = realval_fitness(1000, set_to_1000[0])
    print "Third gene: \t Value: %f, target %f \t Score: %f" \
      % (set_to_1000[0], 1000, score)
    score = chooser_1[chosen[0]] + chooser_2[chosen[1]]
    print "Fourth gene: \t Value:", chosen, "target:", \
      '(Bingo!, Jackpot!)', 'Score:', score


class Genome(object):
    def __init__(self):      
        self._alleles = AllelesWithOperators()
        self._alleles.add(make_real_gene(1, -10, 10, 0.01))
        self._alleles.add(make_bitmap_gene(10), weight=5)
        self._alleles.add(make_real_gene(1, -10000, 10000, 5))
        self._alleles.add(make_choice_gene(2, chooser_keys))
        self._genes = make_alleles_genome(self._alleles)
        self._genes.evaluator.set(eval_func)

    @property
    def alleles(self):
        return self._alleles

    @property
    def genes(self):
        return self._genes

def on_step_generation(ga_engine):
    best = ga_engine.bestIndividual()
    gen = ga_engine.getCurrentGeneration()
    raw_fitn = best.getRawScore()
    now_fitn = eval_func(best)
    print "Best genome at generation %d had raw fitness" % gen, raw_fitn, now_fitn
    print "Best genome at generation %d had genes:" % gen, \
      raw_genes(best, True)
    sys.stdout.flush()
    
class GA(object):
    def __init__(self, genome):
        self._ga = GSimpleGA.GSimpleGA(genome.genes)
        self._ga.setPopulationSize(500)
        self._ga.setGenerations(1000)
        self._ga.setMutationRate(0.1)
        self._ga.setCrossoverRate(0.5)
        self._ga.setMinimax(Consts.minimaxType["maximize"])
        #self._ga.selector.set(Selectors.GRouletteWheel)
        self._ga.selector.set(Selectors.GTournamentSelector)
        self._ga.getPopulation().scaleMethod.set(Scaling.SigmaTruncScaling)
        self._ga.stepCallback.set(on_step_generation)
        #self._ga.setElitism(True)
        #self._ga.setElitismReplacement(options.elite)
        #self._ga.setElitism(False)


    def run(self, freq_stats=50):
        self._ga.evolve(freq_stats=freq_stats)

    def attach_db(self, path):
        sqlite_adapter = DBAdapters.DBSQLite(dbname=path, identify="ex1", 
                                             resetDB=True, commit_freq=1)
        self._ga.setDBAdapter(sqlite_adapter)
        
    @property
    def ga(self):
        return self._ga


def run_ga():
    genome = Genome()
    ga = GA(genome)
    #path = "_results_test_pyevolve_utils.db"
    #ga.attach_db(path)
    ga.run()
    best = ga.ga.bestIndividual()    
    print best
    print "Best individual evaluation:"
    verbose_eval(best)
    #import plot_fitnesses as pf
    #plot_fitness

def plot_fitness_func():
    def one_plot(x, target):
        plt.figure()
        plt.title("Fitness of a value targetting %f" % target)
        plt.plot([target], [1], 'b.', label='Target')
        y = [realval_fitness(target, x0) for x0 in x]
        plt.plot(x, y, 'r-', label='Fitness')
        plt.legend()
    one_plot(np.linspace(-3, 5, 1000), 1)
    one_plot(np.linspace(-3000, 5000, 1000), 1000)
    plt.show()
    
# class Test(testutils.ArrayTestCase):
#     def setUp(self):
#         pass
    
#     def tearDown(self):
#         pass
    
#     def test_(self):
#         """."""
#         pass
        
class Test_raw_genes(unittest.TestCase):
    def test_1dlist(self):
        genome = G1DList.G1DList(3)
        target = [0, 1, 2, 'shalom!']
        genome[:] = target
        raw = raw_genes(genome)
        self.assertEqual(target, raw)

    def test_2dlist(self):
        genome = G2DList.G2DList(3, 3)
        target = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        for i in range(3):
            for j in range(3):
                genome[i][j] = target[i][j]
        raw = raw_genes(genome)
        self.assertEqual(target, raw)

    def test_binarystring(self):
        genome = G1DBinaryString.G1DBinaryString(10)
        target = [0, 1, 0, 0, 1, 1, 0, 1, 1, 0]
        genome.initializator.applyFunctions(genome).next()
        for i in range(len(target)):
            genome[i] = target[i]
        raw = raw_genes(genome)
        self.assertEqual(target, raw)
        
class Test_raw_genes_with_alleles(unittest.TestCase):
    def setUp(self):
        alleles = AllelesWithOperators()
        alleles.add(G2DList.G2DList(3, 3), weight=9)
        alleles.add(G1DBinaryString.G1DBinaryString(10), weight=10)
        alleles.add(G1DList.G1DList(1), weight=1)
        self.genome = G1DList.G1DList(3)
        self.genome.setParams(allele=alleles)
        allele_initializer(self.genome)
        
        self.target_2d = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        self.target_bitmap = [0, 1, 0, 0, 1, 1, 0, 1, 1, 0]
        self.target_1d = [13]
    
        for i in range(3):
            for j in range(3):
                self.genome[0][i][j] = self.target_2d[i][j]
        for i in range(len(self.target_bitmap)):
            self.genome[1][i] = self.target_bitmap[i]
        self.genome[2][:] = self.target_1d[:]

    def test_alleles_no_strip(self):
        target = [self.target_2d, self.target_bitmap, self.target_1d]
        raw = raw_genes(self.genome)
        self.assertEqual(target, raw)
        
    def test_alleles_with_strip(self):
        target = [self.target_2d, self.target_bitmap, self.target_1d[0]]
        raw = raw_genes(self.genome, True)
        self.assertEqual(target, raw)


if __name__ == '__main__':
    run_ga()
    unittest.main()


