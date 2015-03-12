# Script to test a range of 'genomes', in order to substantiate the
# results (or lack of such) from the GA.

import optparse
import random
import copy
import sys

import numpy as np

import load_prediction as lp
import load_prediction_ar as lpar
import load_prediction_ar24 as lpar24
import load_prediction_identity as lpid

def process_data(data_in, genome, model, loci):
    data_now = copy.copy(data_in)
    for preproc in model.preprocessors:
        data_now = preproc(data_now, genome, loci, model.day)
    model_out = model.transformer(data_now, genome, loci, model.day)
    for postproc in reversed(model.postprocessors):
        model_out = postproc(model_out, genome, loci)
    return model_out

def test_one(genome, model, data_in, data_out):
    model_out = process_data(data_in, genome, model, model.loci)
    assert(np.all(model_out.index == data_out.index))            
    error = model.error_func(model_out.values, data_out.values)
    return error

model_creator_class = lpid.IdentityModelCreator
model_creator_class = lpar24.ARHourByHourModelCreator
model_creator_class = lpar.ARModelCreator

cmdline_base = ['--generations=100',
                '--pop-size=100',
                '--mutation=0.2',
                '--crossover=0.3',
                '--no-plot',
                '--elite=1',
                '--num-trials=100',
                '--env-replace=0',
                '--print-pop']

cmdline = cmdline_base + \
          ['--remove-holidays',
           '--data-seed=0',
           '--bc-data',
           '--standardize',
           '--difference=1',
           '--subtract-weekly-pattern']

cmdline = cmdline_base + \
          ['--data-seed=15',
           '--gef-data',
           '--subtract-weekly-pattern']

parser = lp.prediction_options()
parser = lp.ga_options(parser)
parser = lp.data_options(parser)
options, _ = parser.parse_args(cmdline)

lp.options = options

def reseed(seed=None):
    if seed is None:
        seed = options.seed
    random.seed(seed)
    np.random.seed(seed)

reseed()

def new_model():
    model_creator = model_creator_class(options)
    return model_creator.get_model()

def explore_gene_values():
    model = new_model()
    train_iter = model.dataset.train_data_iterator()

    genomes = {'ar': [[2016, ar_order, 0] for ar_order in range(1, 8*24+1)],
               'exo': [[2016, 168, exo_order] for exo_order in range(0, 8*24+1)],
               'hindsight': [[hindsight, 168, 168] for hindsight in range(8*24+1+24, 2016)]}

    # Initialize to -1 to 'see' indexing errors easily. Negative error
    # should never occur.
    errors = {'ar': np.zeros((len(genomes['ar']), options.num_trials)) - 1,
              'exo': np.zeros((len(genomes['exo']), options.num_trials)) - 1,
              'hindsight': np.zeros((len(genomes['hindsight']), options.num_trials)) - 1}

    for key in ['ar']:
        for g_idx in range(len(genomes[key])):
            genome = genomes[key][g_idx]
            print 'Testing', key, 'values, genome:', genome, '...', 
            sys.stdout.flush()
            for d_idx, (data_in, data_out) in zip(range(options.num_trials), train_iter()):
                errors[key][g_idx, d_idx] = test_one(genome, model, data_in, data_out)
            print 'Mean error: ', errors[key][g_idx,:].mean()
    print 'All done.'
    return genomes, errors


def test_fitness_variation(genomes=[[1344, 82, 68], [2016, 174, 104]],
                           evaluations=100,
                           trials=100,
                           MPI=False, seed=None):
    '''For each genome in 'genomes', perform 'evaluations' separate fitness
    evaluations of the genome, each using 'trials' separate samples from
    the dataset. The goal: get an idea of how robust/stable the fitness
    estimate is with the given number of trials.

    '''
    options.num_trials = trials
    errors = np.zeros((len(genomes), evaluations, trials)) - 1
    for g_idx, genome in enumerate(genomes):
        print 'Testing genome:', genome, 'on', evaluations, 'evaluations...'
        sys.stdout.flush()
        for evaluation in range(evaluations):
            options.data_seed = random.randrange(1, 2**32-1)
            print 'Data seed now', options.data_seed
            model = new_model()
            for d_idx, (data_in, data_out) in enumerate(model.dataset.train_data_iterator()()):
                errors[g_idx, evaluation, d_idx] = test_one(genome, model, data_in, data_out)
            print 'Mean error for genome {}, evaluation {}: {}'.format(
                genome, evaluation, errors[g_idx, evaluation, :].mean())
            sys.stdout.flush()
    return errors

def mpi_test_fitness_variation(genomes=[[1344, 82, 68], [2016, 174, 104]],
                           evaluations=100,
                           trials=100):
    '''For each genome in 'genomes', perform 'evaluations' separate fitness
    evaluations of the genome, each using 'trials' separate samples from
    the dataset. The goal: get an idea of how robust/stable the fitness
    estimate is with the given number of trials.

    '''
    from mpi4py import MPI
    mpi_comm = MPI.COMM_WORLD
    is_master = mpi_comm.Get_rank() == 0
    mpi_size = mpi_comm.Get_size()
    
    if is_master:
        print 'Reseeding all hosts with different seeds'
    reseed(mpi_comm.scatter(np.random.random_integers(2**32-1, size=mpi_size)))
        
    evaluations -= evaluations % mpi_size
    if is_master:
        print 'Setting number of evaluations to closest smaller multiple of hosts:', evaluations

    errors = mpi_comm.gather(test_fitness_variation(genomes, evaluations/mpi_size, trials))
    if is_master:
        return np.concatenate(errors, axis=1)

def mpi_test_save(path,
                  genomes=[[1344, 82, 68], [2016, 174, 104]],
                  evaluations=100,
                  trials=100):
    from mpi4py import MPI
    errors = mpi_test_fitness_variation(genomes, evaluations, trials)
    if MPI.COMM_WORLD.Get_rank() == 0:
        np.save(path, errors)
        print 'Master saved errors to', path

def load_and_plot_fitness_variations(path):
    import matplotlib.pyplot as plt
    errors = np.load(path)
    ax_mean = []
    ax_single = []
    for model in range(errors.shape[0]):
        plt.figure()
        plt.hist(errors[model,:,:].mean(axis=1), bins=50)
        plt.title('Model {}, mean error over {} evaluations'.format(model, errors.shape[2]))
        ax_mean += [plt.gca()]
        plt.figure()
        plt.hist(errors[model,:,:].reshape(
            (errors.shape[1]*10, errors.shape[2]/10)).mean(axis=1), bins=50)
        plt.title('Model {}, mean error over {} evaluations'.format(
            model, errors.shape[2]/10))
        ax_mean += [plt.gca()]
        plt.figure()
        plt.hist(errors[model,:,:].flatten(), bins=100)
        plt.title('Model {}, single evaluations'.format(
            model, errors.shape[2]/10))
        ax_single += [plt.gca()]
    xlim = (min([ax.get_xlim()[0] for ax in ax_mean]),
            max([ax.get_xlim()[1] for ax in ax_mean]))
    for ax in ax_mean:
        ax.set_xlim(xlim)
    xlim = (min([ax.get_xlim()[0] for ax in ax_single]),
            max([ax.get_xlim()[1] for ax in ax_single]))
    for ax in ax_single:
        ax.set_xlim(xlim)
    return errors, ax_mean, ax_single
        

def cross_check_winners():
    '''Run the best genomes from a series of evolutionary runs through the
    datasets that the other winners have seen.

    '''
    data_seeds = [0, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 1, 20, 21,
                  22, 23, 24, 25, 26, 27, 28, 29, 2, 3, 4, 5, 6, 7, 8, 9]
    genomes = np.array([[1344, 82, 68], [1344, 80, 82], [2016, 171, 55],
                        [1344, 96, 45], [2016, 174, 104], [1344, 102, 42],
                        [1344, 106, 21], [2016, 50, 17], [1344, 119, 91],
                        [2016, 96, 61], [1344, 73, 44], [1344, 74, 41],
                        [1344, 81, 46], [1344, 50, 67], [1344, 74, 95],
                        [1344, 146, 73], [2016, 176, 67], [1344, 101, 45],
                        [2016, 168, 42], [672, 48, 47], [2016, 168, 85],
                        [1344, 74, 43], [1344, 74, 113], [2016, 158, 87],
                        [1344, 168, 43], [672, 27, 20], [1344, 47, 74],
                        [2016, 175, 89], [2016, 124, 67], [1344, 93, 38]])
    winners = []
    errors = np.zeros((len(data_seeds), len(genomes), options.num_trials)) - 1
    for s_idx, seed in enumerate(data_seeds):
        options.data_seed = seed
        model = new_model()
        for g_idx, genome in enumerate(genomes):
            print 'Testing genome:', genome, 'on seed', seed
            sys.stdout.flush()
            for d_idx, (data_in, data_out) in \
                enumerate(model.dataset.train_data_iterator()()):
                errors[s_idx, g_idx, d_idx] = test_one(genome, model, data_in, data_out)
            print 'Mean error for genome {} on seed {}: {}'.format(
                genome, seed, errors[s_idx, g_idx, :].mean())
            sys.stdout.flush()
        seed_errors = errors[s_idx, :, :].mean(axis=1)
        winners.append(genomes[np.where(seed_errors == seed_errors.min())[0]])
    return winners, errors

if __name__ == '__main__':
    explore_gene_values()
