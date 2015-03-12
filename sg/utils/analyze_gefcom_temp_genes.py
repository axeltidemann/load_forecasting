"""Miscellaneous routines to import/extract/plot the evolution of the
temperature genes in evolved forecasters for the GEFCom 2012 dataset."""

import matplotlib.pyplot as plt
import pandas as pd

def _plot_on_axis(means, station, ax):
    m = means['temp_{}'.format(station)]
    for i in range(30):
        try:
            m.ix[i].plot(ax=ax, color='r', legend=False)
        except:
            print "Trouble plotting run {}, station {}. Missing data?".format(i, station)
    plt.title('Temp station {}'.format(station))
    
def multi(means):
    """Create one plot for each temperature station in 'means'. Draw the
    evolution in each run as a separate line."""
    for s in range(11):
        _plot_on_axis(means, s, plt.figure().gca())

def multi_sub(means, title=None):
    """Create one subplot for each temperature station in 'means'. Draw the
    evolution in each run as a separate line."""
    fig = plt.figure()
    if title is not None:
        plt.suptitle(title)
    for s in range(11):
        _plot_on_axis(means, s, fig.add_subplot(3, 4, s+1))
    
    # for s in range(11):
    #     ax = fig.add_subplot(3, 4, s)
    #     m = means['temp_{}'.format(s)]
    #     for i in range(30):
    #         m.ix[i].plot(ax=ax, color='b', alpha=0.3, legend=False)
    #     plt.title('Temp station {}'.format(s))

def multi_2(means, stations=range(11), fig=None):
    """All runs and (the given) stations on the same plot"""
    if fig is None:
        fig = plt.figure()
    ax = fig.gca()
    columns = ['temp_{}'.format(s) for s in stations]
    lbls = ['Temperature Station {}'.format(s+1) for s in stations]
    for i in range(30):
#        means[columns].ix[i].plot(ax=ax, colormap='jet', alpha=1, legend=False)
        means[columns].ix[i].plot(ax=ax, color=['b', 'c', 'm', 'g', 'y', 'r'], alpha=1, legend=False)
    plt.legend(lbls, loc='right')
        # m.ix[0].plot(ax=ax, colormap='jet', alpha=0.05, legend=False)
        # for i in range(1,30):
        #     m.ix[i].plot(ax=ax, colormap='jet', alpha=0.05, legend=False)
#        plt.title('Temp station {}'.format(t))

def import_from_csv(path):
    """Read the CSV file in 'path', output a pandas Dataframe with 11
    columns, one for each temperature gene, and 100 rows, one for each
    generation in each run. Each value is averaged across all
    individuals in all runs found in the CSV file. The CSV was typically
    made with a command similar to [...]/scripts/parse-logs-into-csv.sh output_*.txt."""
    all = pd.read_csv(open(path, 'r'))
    cols = ['file', 'gen', 'fitn1', 'fitn2', 'hindsight', 'AR_order']
    cols += [ 'temp_{}'.format(i) for i in range(11)]
    all.columns = cols
    grouped = all.groupby(['file', 'gen'], as_index='False')
    means = grouped.mean()
    stds = grouped.std()
    means = means.drop(['fitn1', 'fitn2', 'hindsight', 'AR_order'], axis=1)
    stds = stds.drop(['fitn1', 'fitn2', 'hindsight', 'AR_order'], axis=1)
    return means
