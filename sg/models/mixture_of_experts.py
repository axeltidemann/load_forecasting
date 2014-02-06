"""Mixture of EXPERTS!"""

import os
import re
import math
import ipdb

import Oger
import numpy as np

import sg.utils
import sg.utils.output as output
from sg.globals import SG_SIM_PATH
import sg.data.bchydro as bc
import sg.data.sintef.userloads as ul

split_point = 0.5
dataset = "bc-data"
exp_path = os.path.join(SG_SIM_PATH, "convergence")

class SimpleMixtureOfExperts(object):
    def __init__(self, exp_path, dataset_name, pred_filter=None):
        self.error_func = Oger.utils.mse
        self._load_dataset(dataset_name)
        self.safety_cutoff = 2
        self.make_mixture(exp_path, dataset_name, pred_filter)

    def _load_dataset(self, dataset_name):
        if dataset_name == "bc-data":
            self._dataset = bc.load()
        elif dataset_name == "total-load":
            self._dataset = ul.total_experiment_load()[1]['Load']
        else:
            raise RuntimeError("Invalid dataset name: %s" % dataset_name)

    def _range(self, array): 
        return array.max() - array.min()
    
    def prediction_is_sane(self, prediction):
        return True
        reference = self._dataset[:prediction.index[0]][-24:]
        return self._range(prediction) < self.safety_cutoff * self._range(reference)

    def filtered_mse(self, target, predictions):
        """Return mean MSE over only those days where prediction is believed to
        be sane."""
        return np.mean([self.error_func(p, target[p.index]) for p in predictions \
                        if self.prediction_is_sane(p)])

    def _print_uniques_and_hits(self, alist):
        uniques = np.unique(alist)
        hit_list = [len(np.where(alist == u)[0]) for u in uniques]
        for (date, hits) in zip(uniques, hit_list):
            print "{}: {}".format(date, hits)
        
    def inner_join(self): 
        """Not all models start their testing at the same time. This function
        strips the lists of predictions, such that only the dates that all
        models have in common are kept."""
        all_start_dates = np.array([p[0].index[0] for p in self.predictions])
        start_date = all_start_dates.max()
        all_end_dates = np.array([p[-1].index[-1] for p in self.predictions])
        end_date = all_end_dates.min()
        print "Unique start dates and corresponding number of predictions: "
        self._print_uniques_and_hits(all_start_dates)
        print "Unique end dates and corresponding number of predictions: "
        self._print_uniques_and_hits(all_end_dates)
        print "Chosen (last) start date: ", start_date
        print "Chosen (first) end date: ", end_date
        for i in range(len(self.predictions)):
            prediction = self.predictions[i]
            for day in range(len(prediction)):
                if prediction[day].index[0] == start_date:
                    start_idx = day
                if prediction[day].index[-1] == end_date:
                    end_idx = day + 1
            self.predictions[i] = prediction[start_idx:end_idx]
        self.target = self.target[start_date:end_date]
        
    def make_mixture(self, exp_path, dataset_name, pred_filter):
        """Return target for prediction phase plus mixture of experts as list
        of (MSE, test_set_pred, model_path)."""
        wildcard = os.path.join(exp_path, "*.pickle")
        if pred_filter is None:
            self.paths = output.matching_paths(wildcard, dataset_name)
        else:
            self.paths = output.matching_paths(wildcard, dataset_name, *pred_filter)
        print "The following files are included in the mix: "
        for path in self. paths:
            print path
        datasets = [output.load_pickled_prediction(path) for path in self.paths]
        self.split_sets = [output.split_dataset(dataset, split_point) for dataset in datasets]
        # split_sets = [model_1, ..., model_p]
        # model_x = (validation, test)
        # validation/test = (target, [prediction_day_1, ... prediction_day_n])
        self.mix_mses = [self.filtered_mse(*valid) for (valid, _) in self.split_sets]
        self.predictions = [pred for (_, (_, pred)) in self.split_sets]
        self.target = self.split_sets[0][1][0]
        self.inner_join()

    def predict(self):
        predictions = []
        for day in range(len(self.predictions[0])):
            todays_predictions = [pred[day] for pred in self.predictions]

            pred_mse_used = [(pred, mse) for pred, mse in 
                             zip(todays_predictions, self.mix_mses) if 
                             self.prediction_is_sane(pred)]
            
            sum_inv_mse = sum([1./mse for (_, mse) in pred_mse_used])
            # weights = [(1./mse) / sum_inv_mse for (_, mse) in pred_mse_used]
            # import matplotlib.pyplot as plt
            # plt.clf()
            # plt.plot(weights)
            # plt.show()
            # import ipdb; ipdb.set_trace()
            weighted_preds = [(pred / mse) for pred, mse in pred_mse_used]
            # weighted_preds = [pred * weight for (pred, _), weight in zip(pred_mse_used, weights)]
            predictions.append(sum(weighted_preds)/sum_inv_mse)
        return predictions

class ComplexMixtureOfExperts(SimpleMixtureOfExperts):
    """Similar to the SimpleMixtureOfExperts, but rather than calculating a
    grand mean across all models and runs, prediction first calculates a
    weighted prediction and precision for each run, and subsequently calculates
    a weighted sum over all runs. Turns out to be the exact same formula, just
    estimated in a more difficult way."""
    def predict(self):
        predictions = []
        for day in range(len(self.predictions[0])):
            todays_predictions = [pred[day] for pred in self.predictions]
            preds_used = [(pred, 1./mse, path) for pred, mse, path in 
                                   zip(todays_predictions, self.mix_mses, self.paths) if 
                                   self.prediction_is_sane(pred)]
            run_predictions_precisions = []
            for run in range(30):
                preds, precs = zip(*[(pred, prec) for pred, prec, path in preds_used if
                                     re.search("run_%d_" % run, path)])
                sum_precs = sum(precs)
                weights = [prec/sum_precs for prec in precs]
                run_pred = sum([pred * w for pred, w in zip(preds, weights)])
                run_prec = 1./sum([w*w*(1./p) for w, p in zip(weights, precs)])
                run_predictions_precisions.append((run_pred, run_prec))
            preds, precs = zip(*run_predictions_precisions)
            sum_precs = sum([prec for prec in precs])
            weights = [prec/sum_precs for prec in precs]
            prediction = sum([pred * w for pred, w in zip(preds, weights)])
            predictions.append(prediction)
        return predictions


class PowerOfNMixtureOfExperts(SimpleMixtureOfExperts):
    """Yet another formula."""
    def predict(self):
        predictions = []
        N_validation = [len(valid[0]) for (valid, _) in self.split_sets]
        for day in range(len(self.predictions[0])):
            todays_predictions = [pred[day] for pred in self.predictions]
            pred_mse_used = [(pred, mse, N) for pred, mse, N in 
                             zip(todays_predictions, self.mix_mses, N_validation) if 
                             self.prediction_is_sane(pred)]
            msemin = min([mse for (_, mse, N) in pred_mse_used])
            pred_mse_used = [(pred, mse/msemin, N) for (pred, mse, N) in pred_mse_used]
            weights = [math.pow(math.sqrt(mse), -N) for (_, mse, N) in pred_mse_used]
            normalizer = sum(weights)
            # import matplotlib.pyplot as plt
            # import ipdb; ipdb.set_trace()
            # plt.clf()
            # plt.plot(weights)
            # plt.show()
            weighted_preds = [(weight * pred) \
                              for (pred, _, _), weight in \
                              zip(pred_mse_used, weights)]
            predictions.append(sum(weighted_preds) / normalizer)
        return predictions


def validation_vs_test_scatter(mixture, err_func=Oger.utils.rmse):
    import matplotlib.pyplot as plt
    x = []
    y = []
    for (validation, test) in mixture.split_sets:
        x.append(sg.utils.calc_error(validation[1], validation[0], err_func))
        y.append(sg.utils.calc_error(test[1], test[0], err_func))
    plt.scatter(x, y)
    plt.xlim(min(min(x), min(y)) * 0.95, max(max(x), max(y)) * 1.05)
    plt.ylim(min(min(x), min(y)) * 0.95, max(max(x), max(y)) * 1.05)
    plt.xlabel("Validation set error")
    plt.ylabel("Test set error")
    
if __name__ == "__main__":
    pass
