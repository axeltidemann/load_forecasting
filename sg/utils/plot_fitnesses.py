import sqlite3 as sql
import glob
import numpy as np
import sys
import optparse

import matplotlib.pyplot as plt

import sg.utils

def fetch_one(db_path, exp_id="ex1"):
    """Fetch one evolution, identified by exp_id, from the database given in
    db_path. Returns an array holding: generation, min/avg/max fitness, and
    fitness std dev."""
    with sql.connect(db_path) as conn:
        crs = conn.execute("select generation, rawMin, rawAve, rawMax, "
                           "rawDev from statistics")
        return np.array(zip(*crs.fetchall()))

def fetch_match(pattern, exp_id="ex1"):
    """Fetch all files matching pattern."""
    # While tempting to test for isinstance(pattern, collections.iterable) in
    # order to support multiple patterns (e.g. "fetch_match(sys.argv[1:])"), a
    # string (such as the pattern) will also pass this test.
    return [fetch_one(path, exp_id) for path in glob.glob(pattern)]

def _common_generations(evolutions):
    """Return the list of evolutions shortened to only common generations."""
    stats_lengths = [evo.shape[1] for evo in evolutions]
    last_common_idx = min(stats_lengths)
    first_evo = evolutions[0]
    last_common_gen = first_evo[0, last_common_idx - 1]
    for evo in evolutions[1:]:
        for gen in range(last_common_idx):
            if evo[0,gen] != first_evo[0,gen]:
                last_common_gen = min(last_common_gen, gen)
                last_common_idx = gen
                break

    max_gen = max([evo[0, -1] for evo in evolutions])
    print "max gen is", max_gen
    print "last_common_gen is", last_common_gen
    if last_common_gen < max_gen:
        print >>sys.stderr, "Some generations missing in at least one " \
          "simulation. Plotting only the first generations 0-%d." \
          % last_common_gen
        print >>sys.stderr, "Lengths of statistics for each evolution:"
        print >>sys.stderr, "\t", stats_lengths
        print >>sys.stderr, "Generation at last common index for each evolution:"
        print >>sys.stderr, "\t", [evo[0, last_common_idx-1] 
                                   for evo in evolutions]
    if last_common_idx <= 0:
        raise ValueError("No overlapping generations (one empty evolution?).")
    return [evo[:,:last_common_idx] for evo in evolutions]
    
def join(evolutions):
    """Join the output from several evolutions. Evolutions is a list where each
    evolution element is an array as returned from fetch_one.

    Return generation and averages.
    """
    evolutions = _common_generations(evolutions)
    return np.average(np.array(evolutions), axis=0)

def plot_evols(evolutions, generations=None, axes=None, **plt_kwargs):
    if axes is None:
        axes = plt.axes()
    col = sg.utils.Enum("gen", "min", "avg", "max", "dev")
    mg = -1 if generations is None else generations + 1
    axes.plot(evolutions[col.gen,0:mg], evolutions[col.min,0:mg],
              label="Minimum", **plt_kwargs)
    axes.plot(evolutions[col.gen,0:mg], evolutions[col.avg,0:mg], 
              label="Average", **plt_kwargs)
    axes.plot(evolutions[col.gen,0:mg], evolutions[col.max,0:mg], 
              label="Maximum", **plt_kwargs)
    axes.plot(evolutions[col.gen,0:mg], evolutions[col.dev,0:mg],
              label="Devation", **plt_kwargs)
    return axes

def _get_options():
    parser = optparse.OptionParser()
    parser.usage = "[options] path_to_pyevolve.db [more_pyevolve.dbs]"
    parser.description = "Plot fitness averaged over evolutions from multiple Pyevolve sqlite3 databases"
    parser.add_option("--title", dest="title", default=None, help="Title for the plot")
    parser.add_option("--exp", dest="exp_id", default="ex1", help="Name identifying experiment in database")
    parser.add_option("--generations", dest="generations", type="int", default=None, help="Max number of generations to plot")
    parser.add_option("--ymin", dest="ymin", type="float", default=None, help="Fix Y axis to given min value")
    parser.add_option("--ymax", dest="ymax", type="float", default=None, help="Fix Y axis to given max value")
    return parser.parse_args()

if __name__ == "__main__":
    options, args = _get_options()
    evolutions = [fetch_one(path, options.exp_id) for path in args]
    print "Plotting the average of %d evolutions." % len(evolutions)
    average = join(evolutions)
    plot_evols(average, generations=options.generations)
    plt.legend(loc=(0.2, 0.2))
    if options.ymin is not None:
        plt.ylim(ymin=options.ymin)
    if options.ymax is not None:
        plt.ylim(ymax=options.ymax)
    if options.title is not None:
        plt.title(options.title)
    plt.show()
