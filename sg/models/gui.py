"""Create a window layout containing six graphs as follows:
------------   ------------------------------------
- FITNESS --   ---------- PREDICTION --------------
------------   ------------------------------------
------------   ------------------------------------

------------   ------------------------------------
- GENOME ---   ---------- CLEANSED LOAD -----------
------------   ------------------------------------
------------   ------------------------------------

------------   ------------------------------------
------------   ---------- CLEANSED TEMP -----------
------------   ------------------------------------
------------   ------------------------------------
"""

import time
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sg.utils.pyevolve_utils as pu
import sg.utils
import sg.models.esn as esn

# Set to False for debugging
_catch_exceptions_during_update = True

class LoadPredictionGUI():
    def __init__(self):
        self.fig = plt.figure(figsize=(16,10))
        shape = (6, 9)
        self.plots = \
          {'fitness': plt.subplot2grid(shape, (0, 0), 3, 3),
           'prediction': plt.subplot2grid(shape, (0, 3), 2, 6),
           'genes': plt.subplot2grid(shape, (3, 0), 3, 3),
           'load': plt.subplot2grid(shape, (2, 3), 2, 6),
           'temp': plt.subplot2grid(shape, (4, 3), 2, 6)}
        self._set_axes_labels()
        self.generations = []
        self.fitnesses = ([], [], [])
        plt.ion()
        #plt.tight_layout()
        plt.show()
        manager = plt.get_current_fig_manager()
        manager.resize(1900, 1200)
        
    def _set_axes_labels(self):
        self.plots['fitness'].set_xlabel("Generation")
        self.plots['fitness'].set_ylabel("Fitness")
        self.plots['prediction'].set_ylabel("Predicted load (kWh)")
        self.plots['genes'].set_xlabel("Genes")
        self.plots['genes'].set_ylabel("Value")
        self.plots['load'].set_ylabel("Cleansed load (kWh)")
        self.plots['temp'].set_xlabel("Date")
        self.plots['temp'].set_ylabel("Cleansed temperature (deg. C)")
        
    def _update_fitnesses(self, ga_engine):
        self.generations.append(ga_engine.getCurrentGeneration())
        stats = ga_engine.getStatistics()
        self.fitnesses[0].append(stats["rawMin"])
        self.fitnesses[1].append(stats["rawAve"])
        self.fitnesses[2].append(stats["rawMax"])
        axes = self.plots['fitness']
        axes.clear()
        axes.plot(self.generations, self.fitnesses[0], label='Minimum')
        axes.plot(self.generations, self.fitnesses[1], label='Average')
        axes.plot(self.generations, self.fitnesses[2], label='Maximum')

    def _set_indiv(self, indiv):
        self.indiv = indiv
        self.model = indiv.getParam('model')
        self.day = self.model.day
        self.loci = self.model.loci
        self.genome = pu.raw_genes(indiv, True)
        train_iter = self.model.dataset.train_data_iterator()
        train_data = [t for t in train_iter()]
        (self.data_in, self.data_out) = random.choice(train_data)

    def _plot_prediction(self, model_out, axes):
        axes.clear()
        preview=24*2
        self.cln_data['Load'][-preview-self.day:-self.day].plot(
            ax=axes, style='b', linewidth=2, label='Cleansed historical load data')
        model_out.plot(ax=axes, style='r', linewidth=2, label='Model prediction')
        self.data_out.plot(ax=axes, style='g--', linewidth=2, label='Observed "future" load')
        axes.legend(loc=3)

    def _update_prediction(self):
        model_out = self.model.train_and_predict_func(
            self.cln_data, self.genome, self.loci, self.day)
        axes = self.plots['prediction']
        self._plot_prediction(model_out, axes)
                    
    def _update_cleansing(self):
        self.cln_data = self.data_in if self.model.cleaning_disabled else \
              self.model.clean_func(
                  self.data_in, self.genome, self.loci, self.day, self.model)
        axes = self.plots['temp']
        axes.clear()
        self.data_in['Temperature'].plot(ax=axes, style='g-', linewidth=1, label='_nolegend_')
        self.cln_data['Temperature'].plot(ax=axes, style='b', linewidth=2, label='Temperature')
        axes.legend(loc=3)
        axes = self.plots['load']
        axes.clear()
        self.data_in['Load'][:-self.day].plot(ax=axes, style='g-', linewidth=1, label='_nolegend_')
        self.cln_data['Load'][:-self.day].plot(ax=axes, style='b', linewidth=2, label='Load')
        axes.legend(loc=3)

    def _update_genes(self):
        axes = self.plots['genes']
        axes.clear()
        x = np.arange(len(self.indiv))
        axes.bar(x, self.indiv)
        loci_items = vars(self.loci).items()
        loci_items.sort(key=lambda x: x[1])
        labels = zip(*loci_items)[0]
        axes.set_xticks(x+0.5)
        axes.xaxis.set_ticklabels(labels)
        labels = axes.get_xticklabels()
        for label in labels:
            label.set_rotation(45)
            label.set_horizontalalignment('right')
        
    def update(self, ga_engine):
        best = ga_engine.bestIndividual()
        self._set_indiv(best)
        self._update_fitnesses(ga_engine)
        try:
            self._update_cleansing()
            self._update_prediction()
        except Exception, e:
            if _catch_exceptions_during_update:
                print >>sys.stderr, "Caught exception during GUI update, skipping."
            else:
                raise e
        self._update_genes()
        self.fig.canvas.draw()
        #self.fig.canvas.flush_events()

