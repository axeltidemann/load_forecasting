"""Various extensions and fixes to Pyevolve."""

import ast
import random
import copy
import sys
import math
import numpy as np
from types import BooleanType
import logging

import pyevolve.Util
import pyevolve.GSimpleGA
import pyevolve.G1DList, pyevolve.GenomeBase, pyevolve.G1DBinaryString
import pyevolve.GAllele
import pyevolve.Initializators, pyevolve.Mutators, pyevolve.Crossovers
from pyevolve.GPopulation import GPopulation
from pyevolve.FunctionSlot import FunctionSlot
from pyevolve.GenomeBase import GenomeBase
import pyevolve.Consts as Consts

_pyev_ver = '0.6rc1'
if pyevolve.__version__ != _pyev_ver:
    raise TypeError("Some of these extensions/workarounds only works with Pyevolve version %s" % _pyev_ver)

class AllelesWithOperators: 
    """This class holds a set of alleles along with separate operators for each
    allele. Each allele in this class can be of any gene type, not just the
    "Allele"-type genes in Pyevolve (GAlleleList and GAlleleRange)."""
    # This class reimplements the boilerplate rather than inherit from
    # GAlleles, in order to avoid inheriting unsupported operations (e.g. "add"
    # method with only one argument).

    def __init__(self):
        self._alleles = []
        self._weights = []
        self._total_weight = 0

    def add(self, allele, weight=None): 
        """Appends one allele to the alleles list, together with a
        corresponding mutator. If weight is not given, it will be set to the
        length of the gene created by the initializer."""
        self._alleles.append(allele)
        if weight is None:
            try:
                genome = allele.clone()
                genome.initializator.applyFunctions(genome).next()
                weight = len(genome)
                print >>sys.stderr, "No weight specified for allele, guessing %d." % weight
            except (TypeError, AttributeError):
                print >>sys.stderr, "Failed to get weight for allele, defaulting to 1."
                weight = 1
        self._total_weight += weight
        self._weights.append(weight)
    
    def get_weight(self, index):
         return self._weights[index]

    @property
    def total_weight(self):
        return self._total_weight
    
    def __iter__(self):
        """ Return the list iterator """
        return iter(self._alleles)

    def __len__(self):
        """ Returns the lenght of the alleles list """
        return len(self._alleles)


class AllelesGenome(pyevolve.G1DList.G1DList): 
    """This genome is intended to hold a list of sub-genes, each controlled by
    a gene in an AllelesWithOperators instance."""

    def _the_deep_copy(self, g):
        g.genomeList = copy.deepcopy(self.genomeList[:])

    def _the_hacky_copy(self, g):
        g.genomeSize = self.genomeSize
        g.genomeList = [gene.clone() for gene in self.genomeList]

    def copy(self, g):
        pyevolve.GenomeBase.GenomeBase.copy(self, g)
        #self._the_deep_copy(g)
        self._the_hacky_copy(g)

    def clone(self):
        newcopy = AllelesGenome(self.genomeSize, True)
        self.copy(newcopy)
        return newcopy
        
def _get_mu_sigma(genome):
    mu = genome.getParam("gauss_mu")
    sigma = genome.getParam("gauss_sigma")
    if mu is None or sigma is None:
        raise RuntimeError("mu and sigma should have been set!")
    return mu, sigma


def _noise_mutate_genome(genome, noise_func, **args):
    """Gauss mutation for reals and integers are identical but for the mutated
    value. This function performs the rest of the mutation logic, but gets the
    (normally distributed) noise values of the correct datatype from
    noise_func. Mutations with another distribution could use the same logic
    but supply a different noise function."""
    pmut = args["pmut"]
    if pmut <= 0.0: 
        return 0
    mutations = 0
    for it in range(len(genome)):
        if pyevolve.Util.randomFlipCoin(pmut):
            new_value = genome[it] + noise_func()
            new_value = min(new_value, genome.getParam(
                "rangemax", Consts.CDefRangeMax))
            new_value = max(new_value, genome.getParam(
                "rangemin", Consts.CDefRangeMax))
            genome[it] = new_value
            mutations += 1
    return mutations
 

def G1DListMutatorRealGaussianFixed(genome, **args):
    """ The mutator of G1DList, Gaussian Mutator

    Accepts the *rangemin* and *rangemax* genome parameters, both
    optional. Also accepts the parameter *gauss_mu* and the *gauss_sigma* which
    respectively represents the mean and the std. dev. of the random
    distribution.

    Modified to mutate each gene independently of the others, rather than the
    original implementation, where a deterministic number of genes are always
    mutated."""
    mu, sigma = _get_mu_sigma(genome)
    return _noise_mutate_genome(
        genome, lambda: random.gauss(mu, sigma), **args)


def G1DListMutatorIntegerGaussianFixed(genome, **args):
    """ A gaussian mutator for G1DList of Integers

    Accepts the *rangemin* and *rangemax* genome parameters, both optional. Also
    accepts the parameter *gauss_mu* and the *gauss_sigma* which respectively
    represents the mean and the std. dev. of the random distribution.

    Modified to mutate each gene independently of the others, rather than the
    original implementation, where a deterministic number of genes are always
    mutated."""
    mu, sigma = _get_mu_sigma(genome)
    return _noise_mutate_genome(
        genome, lambda: int(random.gauss(mu, sigma)), **args)


def G1DBinaryStringMutatorFlipFixed(genome, **args):
    """The classical flip mutator for binary strings.
    
    Modified to mutate each gene independently of the others, rather than the
    original implementation, where a deterministic number of genes are alwas
    mutated."""
    pmut = args["pmut"]
    if pmut <= 0.0: 
        return 0

    def flip(it):
        if genome[it] == 0: 
            genome[it] = 1
        else: 
            genome[it] = 0
        
    mutations = 0
    for it in range(len(genome)):
        if pyevolve.Util.randomFlipCoin(pmut):
            flip(it)
            mutations+=1
            
    return mutations

class ExponentialScaler(object):
    def __init__(self, generations, exp_min=0.1, exp_max=30, exp_range=0.1):
        self._exp_now = exp_min
        self._exp_range = exp_range
        self._factor = (exp_max - exp_min) / (0.8 * generations)

    def scale(self, pop):
        scores = np.array([indiv.score for indiv in pop])
        scores_min = scores.min()
        scores_range = scores.max() - scores_min
        exp_now = self._exp_now
        self._exp_now += self._factor
        if abs(scores_range) < 1e-9:
            for indiv in pop:
                indiv.fitness = 1
            return
        scores = np.exp(exp_now + exp_now * self._exp_range *
                        (scores - scores_min)/scores_range)
        scores = (scores - scores.min()) / (scores.max() - scores.min())
        for i in range(len(scores)):
            pop[i].fitness = scores[i]

    def __call__(self, pop):
        return self.scale(pop)

        
def BoltzmannScalingFixed(pop):
    """ Boltzmann scaling scheme. You can specify the **boltz_temperature** to
    the population parameters, this parameter will set the start
    temperature. You can specify the **boltz_factor** and the **boltz_min**
    parameters, the **boltz_factor** is the value that the temperature will be
    subtracted and the **boltz_min** is the mininum temperature of the scaling
    scheme.
    
    The Pyevolve version (0.6.1rc1) has a bug: it reads param
    boltz_temperature, but sets param boltzTemperature. Result:
    boltz_temperature never changes.
    """
    
    boltz_temperature = pop.getParam("boltz_temperature", Consts.CDefScaleBoltzStart)
    boltz_factor = pop.getParam("boltz_factor", Consts.CDefScaleBoltzFactor)
    boltz_min = pop.getParam("boltz_min", Consts.CDefScaleBoltzMinTemp)

    boltz_temperature-= boltz_factor
    boltz_temperature = max(boltz_temperature, boltz_min)
    pop.setParams(boltz_temperature=boltz_temperature)

    boltz_e = []
    avg = 0.0

    for i in xrange(len(pop)):
        val = math.exp(pop[i].score / boltz_temperature)
        boltz_e.append(val)
        avg += val
       
    # avg /= len(pop) # We believe this messes up the probability aspect.

    for i in xrange(len(pop)):
        pop[i].fitness = boltz_e[i] / avg

def allele_initializer(genome, **args):
    """Initializer for use with instances of the AllelesWithOperators class.

    To use this initializer, you must specify the *allele* genome parameter
    with the :class:`GAllele.GAlleles` instance.
    """
    alleles = genome.getParam("allele", None)
    if alleles is None:
        pyevolve.Util.raiseException("To use the AlleleInitializator, you "
                                     "must specify the 'allele' parameter", 
                                     TypeError)
    genome.genomeList = [allele.clone() for allele in alleles]
    for gene in genome.genomeList:
        gene.initializator.applyFunctions(gene).next()

    
def allele_mutator(genome, **args):
    """Mutator for use with instances of the AllelesWithOperators class.

    To use this mutator, you must specify the *allele* genome parameter with the
    :class:`GAllele.GAlleles` instance.

    The "standard" Pyevolve mutators (e.g. G1DBinaryStringMutatorFlip)
    interpret the 'pmut' argument as per-gene mutation probability, whereas the
    Pyevolve allele mutators (e.g. G1DListMutatorAllele) interpret the 'pmut'
    argument as per-allele mutation probability. This implementation interprets
    'pmut' as something in-between, taking into account the length of each
    allele when calculating per-allele probs unless the weight was set
    explicitly when the allele was added.
    """
    alleles = genome.getParam("allele", None)
    if alleles is None:
        pyevolve.Util.raiseException("To use the AlleleMutator, you must "
                            "specify the 'allele' parameter", TypeError)
    pmut = args["pmut"]
    num_mutations = 0
    for i in range(len(genome)):
        weight = alleles.get_weight(i)
        if pyevolve.Util.randomFlipCoin(pmut * weight):
            for mut in genome[i].mutator.applyFunctions(genome[i], pmut=pmut):
                num_mutations += mut
    return num_mutations

    
def allele_crossover(_, **args): 
    """Allele crossover. Modified version of single-point crossover: Selects
    the cross point based on the weights supplied to the genome when adding
    alleles. If a crossover function is supplied at the crossover point, this
    will be called prior to performing one-point crossover on the entire
    genome."""
    gMom = args["mom"]
    gDad = args["dad"]
    alleles = gMom.getParam("allele", None)
    if alleles is None:
        pyevolve.Util.raiseException("To use the AlleleCrossover, you must "
                            "specify the 'allele' parameter", TypeError)
    if not 1 <= args["count"] <= 2:
        pyevolve.Util.raiseException("This crossover assumes 1 <= count <= 2.",
                                     RuntimeError)

    inner_cut = random.randint(0, alleles.total_weight-1)
    weight_now = 0
    for outer_cut in range(len(gMom)):
        weight_now += alleles.get_weight(outer_cut)
        if weight_now > inner_cut:
            break
    sister = gMom.clone()
    sister.resetStats()
    brother = gDad.clone()
    brother.resetStats()
    # Swap between siblings via temporary since list elements are objects that
    # will be referenced, not copied, by the slice operation.
    temporary = sister[outer_cut:]
    sister[outer_cut:] = brother[outer_cut:]
    brother[outer_cut:] = temporary

    if not sister[outer_cut].crossover.isEmpty():
        for it in sister[outer_cut].crossover.applyFunctions(
                mom=sister[outer_cut], dad=brother[outer_cut], count=2):
            sister[outer_cut], brother[outer_cut] = it

    if args["count"] == 2:
        return (sister, brother)
    else:
        assert(args["count"] == 1)
        return (sister, None)

def _set_single_point_crossover(gene, length,
        cross_func=pyevolve.Crossovers.G1DListCrossoverSinglePoint):
    if length == 1:
        gene.crossover.clear()
    else:
        gene.crossover.set(cross_func)
    
def make_real_gene(length, minval, maxval, sigma):
    gene = pyevolve.G1DList.G1DList(length)
    gene.initializator.set(pyevolve.Initializators.G1DListInitializatorReal)
    gene.mutator.set(G1DListMutatorRealGaussianFixed)
    _set_single_point_crossover(gene, length)
    gene.setParams(rangemin=minval, rangemax=maxval, gauss_mu=0, gauss_sigma=sigma)
    return gene

def make_int_gene(length, minval, maxval, sigma):
    gene = pyevolve.G1DList.G1DList(length)
    gene.initializator.set(pyevolve.Initializators.G1DListInitializatorInteger)
    gene.mutator.set(G1DListMutatorIntegerGaussianFixed)
    _set_single_point_crossover(gene, length)
    gene.setParams(rangemin=minval, rangemax=maxval, gauss_mu=0, gauss_sigma=sigma)
    return gene

def make_choice_gene(length, choices):
    sub_allele = pyevolve.GAllele.GAlleleList(choices)
    sub_allele_set = pyevolve.GAllele.GAlleles(homogeneous=True)
    sub_allele_set.add(sub_allele)
    gene = pyevolve.G1DList.G1DList(length)
    gene.setParams(allele=sub_allele_set)
    gene.mutator.set(pyevolve.Mutators.G1DListMutatorAllele)
    gene.initializator.set(pyevolve.Initializators.G1DListInitializatorAllele)
    _set_single_point_crossover(gene, length)
    return gene

def make_bitmap_gene(length):
    gene = pyevolve.G1DBinaryString.G1DBinaryString(length)
    gene.initializator.set(pyevolve.Initializators.G1DBinaryStringInitializator)
    gene.mutator.set(G1DBinaryStringMutatorFlipFixed)
    _set_single_point_crossover(
        gene, length, pyevolve.Crossovers.G1DBinaryStringXSinglePoint)
    return gene

def make_alleles_genome(alleles):
    genome = AllelesGenome(len(alleles))
    genome.initializator.set(allele_initializer)
    genome.mutator.set(allele_mutator)
    genome.crossover.set(allele_crossover)
    genome.setParams(allele=alleles)
    return genome

def raw_genes(genome, strip=False):
    """Try to descend into the genome hierarchy and build a structure
    containing genes only, no other information such as operators, fitness,
    etc. If strip is True, remove brackets around one-element lists."""
    def recursion(genome):
        if type(genome) is str:
            return genome
        try:
            raw = [recursion(gene) for gene in genome[:]]
            return raw[0] if len(raw) == 1 and strip else raw
        # "Python" data types will raise TypeError when doing [:] and they're not
        # iterable, whereas numpy.int64 raises IndexError.
        except (TypeError, IndexError): 
            return genome
    # Special case: single-element genome. Make sure we return a one-element
    # list, not e.g. an int.
    if len(genome) == 1 and len(genome[0]) == 1:
        raw = [genome[0][0]]
    else:
        raw = recursion(genome)
    return raw


class SpecifiedPopulation(GPopulation):
    def __init__(self, genome):
        GPopulation.__init__(self, genome)
        self._init_path = genome.getParam("initial_population")

    def create(self, **args):
        if self._init_path is not None:
            nlines = 0
            with open(self._init_path, "r") as initial_pop_file:
                for lines in initial_pop_file:
                    nlines += 1
            self.setPopulationSize(nlines)
        GPopulation.create(self, **args)

    def initialize(self, **args):
        if self._init_path is None:
            GPopulation.initialize(self, **args)
        else:
            with open(self._init_path, "r") as initial_pop_file:
                for gen in self.internalPop:
                    gen.initialize(**args)
                    try:
                        line = initial_pop_file.next()
                    except StopIteration:
                        raise RuntimeError("Internal error, mismatch between "\
                                           "population size and initial "\
                                           "population file length")
                    gl = ast.literal_eval(line)
                    for i in range(len(gl)):
                        try:
                            gen[i][:] = gl[i][:]
                        except TypeError:
                            gen[i][0] = gl[i]

class SimpleGAWithFixedElitism(pyevolve.GSimpleGA.GSimpleGA):
    # "Reimplementation" of GSimpleGA with ability to create different
    # population types (used by SimpleMPIGA), and fixed elitism where elite
    # individuals also have their fitness evaluated at each generation.
    def __init__(self, genome, seed=None, interactiveMode=True):
        if seed: random.seed(seed)

        if type(interactiveMode) != BooleanType:
            pyevolve.Util.raiseException("Interactive Mode option must be True or False", TypeError)

        if not isinstance(genome, GenomeBase):
            pyevolve.Util.raiseException("The genome must be a GenomeBase subclass", TypeError)

        self.internalPop  = self.make_population(genome)
        self.nGenerations = Consts.CDefGAGenerations
        self.pMutation     = Consts.CDefGAMutationRate
        self.pCrossover    = Consts.CDefGACrossoverRate
        self.nElitismReplacement = Consts.CDefGAElitismReplacement
        self.setPopulationSize(Consts.CDefGAPopulationSize)
        self.minimax        = Consts.minimaxType["maximize"]
        self.elitism        = True

        # Adapters
        self.dbAdapter          = None
        self.migrationAdapter = None
        
        self.time_init         = None
        self.interactiveMode = interactiveMode
        self.interactiveGen  = -1
        self.GPMode = False

        self.selector                = FunctionSlot("Selector")
        self.stepCallback          = FunctionSlot("Generation Step Callback")
        self.terminationCriteria = FunctionSlot("Termination Criteria")
        self.selector.set(Consts.CDefGASelector)
        self.allSlots                = [ self.selector, self.stepCallback, self.terminationCriteria ]

        self.internalParams = {}

        self.currentGeneration = 0

        # GP Testing
        for classes in Consts.CDefGPGenomes:
            if  isinstance(self.internalPop.oneSelfGenome, classes):
                self.setGPMode(True)
                break
        
        logging.debug("A GA Engine was created, nGenerations=%d", self.nGenerations)

    def step(self):
        """ Just do one step in evolution, one generation """
        genomeMom = None
        genomeDad = None

        newPop = self.make_population(self.internalPop)
        logging.debug("Population was cloned.")
        
        size_iterate = len(self.internalPop)

        # Odd population size
        if size_iterate % 2 != 0: size_iterate -= 1

        crossover_empty = self.select(popID=self.currentGeneration).crossover.isEmpty()
        
        for i in xrange(0, size_iterate, 2):
            genomeMom = self.select(popID=self.currentGeneration)
            genomeDad = self.select(popID=self.currentGeneration)
            
            if not crossover_empty and self.pCrossover >= 1.0:
                for it in genomeMom.crossover.applyFunctions(mom=genomeMom, dad=genomeDad, count=2):
                    (sister, brother) = it
            else:
                if not crossover_empty and pyevolve.Util.randomFlipCoin(self.pCrossover):
                    for it in genomeMom.crossover.applyFunctions(mom=genomeMom, dad=genomeDad, count=2):
                        (sister, brother) = it
                else:
                    sister = genomeMom.clone()
                    brother = genomeDad.clone()

            sister.mutate(pmut=self.pMutation, ga_engine=self)
            brother.mutate(pmut=self.pMutation, ga_engine=self)

            newPop.internalPop.append(sister)
            newPop.internalPop.append(brother)

        if len(self.internalPop) % 2 != 0:
            genomeMom = self.select(popID=self.currentGeneration)
            genomeDad = self.select(popID=self.currentGeneration)

            if pyevolve.Util.randomFlipCoin(self.pCrossover):
                for it in genomeMom.crossover.applyFunctions(mom=genomeMom, dad=genomeDad, count=1):
                    (sister, brother) = it
            else:
                sister = random.choice([genomeMom, genomeDad])
                sister = sister.clone()
                sister.mutate(pmut=self.pMutation, ga_engine=self)

            newPop.internalPop.append(sister)

        #Niching methods- Petrowski's clearing
        self.clear()

        if self.elitism:
            logging.debug("Doing elitism.")
            if self.getMinimax() == Consts.minimaxType["maximize"]:
                for i in range(self.nElitismReplacement):
                        newPop[len(newPop)-1-i] = self.internalPop.bestRaw(i)
            elif self.getMinimax() == Consts.minimaxType["minimize"]:
                for i in range(self.nElitismReplacement):
                        newPop[len(newPop)-1-i] = self.internalPop.bestRaw(i)

        # Evalate after elitism, in order to re-evaluate elite individuals on
        # potentially changed environment.
        logging.debug("Evaluating the new created population.")
        newPop.evaluate()

        self.internalPop = newPop
        self.internalPop.sort()

        logging.debug("The generation %d was finished.", self.currentGeneration)

        self.currentGeneration += 1

        return (self.currentGeneration == self.nGenerations)

    def make_population(self, genome):
        return SpecifiedPopulation(genome)

    
if __name__ == "__main__":
    from unittest import main
    main(module="test_" + __file__[:-3])
    
