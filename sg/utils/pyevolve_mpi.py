"""MPI variant of Pyevolve."""

import numpy as np
import random
import sys

from mpi4py import MPI
import pyevolve
from pyevolve.GPopulation import GPopulation, multiprocessing_eval
from pyevolve.GSimpleGA import GSimpleGA

import pyevolve_utils as pu

class MPIPopulation(pu.SpecifiedPopulation):
    def __init__(self, genome):
        if isinstance(genome, pu.SpecifiedPopulation):
            if not isinstance(genome, MPIPopulation):
                raise RuntimeError("A non-MPI population has crept into the system!")
        pu.SpecifiedPopulation.__init__(self, genome)
        self._init_MPI()
        
    def _init_MPI(self):
        self._comm = MPI.COMM_WORLD
        self._nhosts = self._comm.Get_size()
        self._rank = self._comm.Get_rank()

    def evaluate(self, **args):
        if self._rank == 0:
            pop_size = len(self.internalPop)
            indices = np.linspace(0, pop_size, self._nhosts+1).astype('int')
            scattered = [self.internalPop[start:end] for (start, end) in \
                         zip(indices[:-1], indices[1:])]
        else:
            scattered = None
        indivs = self._comm.scatter(scattered)
        fitnesses = np.array([multiprocessing_eval(indiv) for indiv in indivs])
        all_fitnesses = self._comm.gather(fitnesses)
        if self._rank == 0:
            for individual, score in zip(self.internalPop, np.concatenate(all_fitnesses)):
               individual.score = score
        self.clearFlags()

        
class SimpleMPIGA(pu.SimpleGAWithFixedElitism):
    def __init__(self, genome, seed=None, interactiveMode=True):
        self._comm = MPI.COMM_WORLD
        self._rank = self._comm.Get_rank()
        self._nhosts = self._comm.Get_size()
        pu.SimpleGAWithFixedElitism.__init__(self, genome, seed, interactiveMode)

    def make_population(self, genome):
        return MPIPopulation(genome)

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
    def rank(self):
        return self._rank

    @property
    def nhosts(self):
        return self._nhosts
