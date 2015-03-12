"""MPI variant of Pyevolve."""

import numpy as np
import random
import sys
import collections

from mpi4py import MPI
import pyevolve
from pyevolve.GPopulation import GPopulation, multiprocessing_eval
from pyevolve.GSimpleGA import GSimpleGA

import sg.utils
import pyevolve_utils as pu
from sg.utils.cache import ATimeCache

class MPIPopulation(pu.SpecifiedPopulation):
    def __init__(self, ga, genome):
        self._ga = ga
        if isinstance(genome, pu.SpecifiedPopulation):
            if not isinstance(genome, MPIPopulation):
                raise RuntimeError("A non-MPI population has crept into the system!")
        pu.SpecifiedPopulation.__init__(self, genome)        
        
    def _make_data_cache_key(self):
        key = []
        train_iter = self._ga.model.dataset.train_data_iterator()
        for (data_in, data_out) in train_iter():
            key += [data_in.index[0].value, data_in.index[-1].value,
                    data_out.index[0].value, data_out.index[-1].value]
        return tuple(key)
        
    def _make_cache_key(self, indiv):
        genome = pu.raw_genes(indiv, True)
        return tuple(sg.utils.safe_deep_flatten(genome))

    def evaluate(self, **args):        
        if self._ga.rank == 0:
            cache = self._ga.caches[self._make_data_cache_key()]
            keys = [self._make_cache_key(indiv) for indiv in self.internalPop]
            uncached = [key not in cache for key in keys]
            uncached_indices = np.where(uncached)[0]
            cached_indices = np.where(np.logical_not(uncached))[0]
            unevaled_pop = [self.internalPop[index] for index in uncached_indices]
            pop_size = len(unevaled_pop)
            print "Cache size is {}, unevaluated population size is {}. Now scattering"\
                .format(len(cache), pop_size)
            sys.stdout.flush()
            indices = np.linspace(0, pop_size, self._ga.nhosts+1).astype('int')
            scattered = [unevaled_pop[start:end] for (start, end) in \
                         zip(indices[:-1], indices[1:])]
        else:
            scattered = None
        indivs = self._ga.comm.scatter(scattered)
        fitnesses = np.array([multiprocessing_eval(indiv) for indiv in indivs])
#        print "Evaluation of {} indivs complete on host {}".format(len(indivs), self._ga.rank)
        sys.stdout.flush()
        all_fitnesses = self._ga.comm.gather(fitnesses)
        if self._ga.rank == 0:
            # Fetch from cache before adding newly evaluated genomes, as
            # these may otherwise delete old cached entries before their
            # values are retrieved.
            for index in cached_indices:
               self.internalPop[index].score = cache[keys[index]]
            for index, score in zip(uncached_indices, np.concatenate(all_fitnesses)):
               self.internalPop[index].score = score
               cache[keys[index]] = score
        self.clearFlags()

        
class SimpleMPIGA(pu.SimpleGAWithFixedElitism):
    def __init__(self, model, genome, seed=None, interactiveMode=True):
        self._init_MPI()
        self._model = model
        self._caches = collections.defaultdict(lambda: ATimeCache(1000))
        pu.SimpleGAWithFixedElitism.__init__(self, genome, seed, interactiveMode)

    def _init_MPI(self):
        self._comm = MPI.COMM_WORLD
        self._nhosts = self._comm.Get_size()
        self._rank = self._comm.Get_rank()

    def make_population(self, genome):
        return MPIPopulation(self, genome)

    def evolve(self, freq_stats=0):
        if not self.terminationCriteria.isEmpty():
            raise RuntimeError("Termination criteria other than number of generations unsupported under MPI.")
        if self._rank != 0:
            raise RuntimeError("Evolve should only be called on rank 0 process.")
        return pu.SimpleGAWithFixedElitism.evolve(self, freq_stats)

    def eval_loop(self):
        stopFlagCallback = False
        for gen in range(self.nGenerations + 1):
            self.internalPop.evaluate()
            if not self.stepCallback.isEmpty():
                for it in self.stepCallback.applyFunctions(self):
                    stopFlagCallback = it
            if stopFlagCallback:
                break

    @property
    def model(self):
        return self._model

    @property
    def caches(self):
        return self._caches
    
    @property
    def comm(self):
        return self._comm

    @property
    def rank(self):
        return self._rank

    @property
    def nhosts(self):
        return self._nhosts
