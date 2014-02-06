import os
import random
import time

import numpy as np
from mpi4py import MPI

pop_size = 512*10
genome_length = 100
generations = 250
mutation_rate = 0.1

comm = MPI.COMM_WORLD
nhosts = comm.Get_size()
rank = comm.Get_rank()


def evolve():
    population = np.random.randint(2, size=(pop_size, genome_length)).astype('float')
    rest_time = None
    for gen in range(generations):
        eval_start = time.time()
        fitnesses = evaluate(population)
        eval_time = time.time() - eval_start
        rest_start = time.time()
        print_stats(gen, population, fitnesses, eval_time, rest_time)
        reproduce(population, fitnesses)
        rest_time = time.time() - rest_start

def eval_loop():
    for gen in range(generations):
        evaluate(None)

def eval_local(population):
    fitnesses = np.empty(len(population))
    target = np.arange(1, genome_length+1)
    for idx in range(population.shape[0]):
        fitnesses[idx] = -np.abs((population[idx,:] - target)).sum()
    return fitnesses
    
def evaluate_ndarray(population=None):
    indices = np.linspace(0, pop_size, nhosts+1).astype('int')
    displs = indices[:-1]
    sendcounts = indices[1:] - displs
    if rank == 0:
        sendbuf = (population, np.array(sendcounts) * genome_length, 
                   np.array(displs) * genome_length, MPI.DOUBLE)
        recvbuf = (np.empty(pop_size), sendcounts, displs, MPI.DOUBLE)
    else:
        sendbuf = None
        recvbuf = None

    indivs = np.empty(sendcounts[rank] * genome_length)
    fitnesses = np.empty(sendcounts[rank])
        
    comm.Scatterv(sendbuf,indivs)
    indivs.shape = (len(indivs)/genome_length, genome_length)

    fitnesses = eval_local(indivs)

    comm.Gatherv(fitnesses, recvbuf)
    if rank == 0:
        return recvbuf[0]

def evaluate_pickle(population=None):
    if rank == 0:
        indices = np.linspace(0, pop_size, nhosts+1).astype('int')
        starts = indices[:-1]
        ends = indices[1:]
        scattered = [population[s:e,:] for s,e in zip(starts, ends)]
    else:
        scattered = None
    indivs = comm.scatter(scattered)
    fitnesses = eval_local(indivs)
    all_fitnesses = comm.gather(fitnesses)
    if rank==0:
        return np.concatenate(all_fitnesses)

evaluate = evaluate_ndarray
#evaluate = evaluate_pickle

def print_stats(gen, population, fitnesses, eval_time, rest_time):
    if rest_time is None:
        timetxt = "%.4f" % eval_time
    else:
        timetxt = "%.4s/%.4s" % (eval_time, rest_time)
    print "Generation %d in %s: Fitnesses %.2f/%.2f/%.2f. Best indiv:" \
      % (gen, timetxt, fitnesses.min(), fitnesses.mean(), fitnesses.max())
    print population[fitnesses.argmax(),:]

def mutate(indiv):
    for i in range(len(indiv)):
        if random.random() < mutation_rate:
            indiv[i] = indiv[i] + 1 if random.random() < 0.5 else indiv[i] - 1

def reproduce(population, fitnesses):
    best = population[fitnesses.argmax(),:]
    for idx in range(pop_size):
        population[idx] = best
    mutations = np.where(np.random.random((pop_size, genome_length)) < mutation_rate)
    mutvals = np.random.randint(low=-1, high=2, size=len(mutations[0]))
    population[mutations] += mutvals
        #mutate(population[idx])
    
if __name__ == "__main__":
    if rank == 0:
        evolve()
    else:
        eval_loop()
        
