"""Routines for doing "regularized autoregression"."""

import sys

import numpy as np
import scipy
import scipy.optimize
import pandas as pd

def make_vector_ar_dataset(data, lags_2d, num_models, out_cols=[0],
                           relative_lags=True, add_bias=True):
    """Build dataset for an autoregressive vector model consisting of
    'num_models' consecutive sub-model, each with its own set of
    regression coefficients.
    
    relative_lags: If True, then the lags are interpreted as backshifts
       relative to each sub-model. Lag 0 indicates that the present time
       is used as input (only makes sense for non-output columns).
       
       If False, endogenuous lags are interpreted as relative to the
       first sub-model, and exogenuous lags as relative to the last
       sub-model (i.e. all sub-models see all the driver data for that
       period/season). As a result, all sub-models share the same design
       matrix.

    add_bias:
       If True, include a bias column of all 1's in the design matrix.

    Returns the design and observed matrices, as well as the lags for
    each sub-model, where all lags are relative to the last sub-model.

    """
    ar_order = int(np.concatenate(lags_2d).max())
    N = (data.shape[0] - ar_order) / num_models
    if N < 1:
        raise ValueError('Invalid input data: AR order ({}) too high for '
                         'available data ({} time steps, {} vectors in model).'.format(
                             ar_order, data.shape[0], num_models))
    p = sum([len(lags) for lags in lags_2d])
    if add_bias:
        design = np.empty((num_models, N, p+1))
        design[:] = np.nan
        design[...,p] = 1
    else:
        design = np.empty((num_models, N, p))
        design[:] = np.nan
    observed = np.empty((num_models, N, len(out_cols)))
    observed[:] = np.nan
    y0 = data.shape[0] - (N * num_models)
    # This way of constructing vector_lags_2d allows for lag lists of
    # different lengths for the different columns in the data set.
    vector_lags_2d = np.array([[np.array(lags) for lags in lags_2d] \
                               for _ in range(num_models)], copy=True)
    for model in range(num_models):
        if relative_lags:
            for col in range(data.shape[1]):
                vector_lags_2d[model,col] += num_models - model - 1
        else:
            for col in out_cols:
                vector_lags_2d[model,col] += num_models - 1

    for model in range(num_models):
        offset = 0
        for col, lags in enumerate(vector_lags_2d[model]):
            nlags = len(lags)
            if nlags > 0:
                # The range below starts at y0 + num_models since all
                # lags are now relative to the LAST sub-model.
                for (i, j) in enumerate(range(y0 + num_models - 1, data.shape[0], num_models)):
                    design[model, i, offset:offset+nlags] = data[(j - lags), col]
            offset += nlags
        observed[model, ...] = data[y0+model::num_models, out_cols]
    return design, observed, vector_lags_2d

def make_vanilla_vector_dataset(data):
    """First version of Tao Hongs vanilla benchmark in vector form."""
    temps = data['Temperature']
    num_params = 5 + 4*12 + 7
    print 'Created model with', num_params, 'parameters.'
    a = np.zeros((len(data), num_params))
    for i in range(a.shape[0]):
        day, month = temps.index[i].dayofweek, temps.index[i].month - 1
        tmp = temps[i]
        trend = i/24 + 1
        a[i, 0:5] = [ 1, trend, tmp, tmp**2, tmp**3]
        offset = 5
        a[i, offset + day] = 1
        offset += 7
        a[i, offset + month] = 1
        offset += 12
        a[i, offset + month] = tmp
        offset += 12
        a[i, offset + month] = tmp**2
        offset += 12
        a[i, offset + month] = tmp**3
        assert(offset + 12 == num_params)
    av = np.zeros((24, len(data)/24, num_params))
    lv = np.zeros((24, len(data)/24, 1))
    for i in range(24):
        av[i,...] = a[i::24,:]
        lv[i,...] = np.atleast_2d(data['Load'].ix[i::24]).T
    return av, lv

def make_vanilla_vector_dataset_2(data):
    """Second version of Tao Hongs vanilla benchmark in vector form."""
    temps = data['Temperature']
    num_params = 12
    print 'Created model with', num_params, 'parameters.'
    a = np.zeros((len(data), num_params))
    for i in range(a.shape[0]):
        day, month = temps.index[i].dayofweek, temps.index[i].month - 1
        tmp = temps[i]
        trend = i/24 + 1
        a[i, 0:5] = [ 1, trend, tmp, tmp**2, tmp**3]
        offset = 5
        a[i, offset + day] = 1
        assert(offset + 7 == num_params)
    raise NotImplementedError(
        'This is not finished. Tricky to make it work; since each submodel will get '
        'a different number of rows (different number of days in the different months).')
    
    
class SmoothVectorLinregEstimator(object):    
    """Estimate regression coefficients for a vector model consisting
    of 'num_models' consecutive linear sub-models, and continuity between
    the sub-models by lambda_cont. 

    Create, then call 'estimate' to do the actual work.

    """

    def __init__(self, design, observed, lambda_cont):
        self.design = design
        self.observed = observed
        self.lambda_cont = lambda_cont
        self.num_models, self.N, self.p = design.shape

    def error(self, coeffs):
        err = 0
        coeffs = coeffs.reshape(self.num_models, self.p)
        for mod_design, mod_coeffs, mod_observed in zip(self.design, coeffs, self.observed):
            err += np.sum(np.power(np.dot(mod_design, mod_coeffs) - mod_observed.flat, 2))
        return err
    
    def penalty(self, coeffs):
        pen = 0
        coeffs = coeffs.reshape(self.num_models, self.p)
        for coeffs1, coeffs2 in zip(coeffs[:-1,:], coeffs[1:,:]):
            pen += np.sum(np.power(coeffs1 - coeffs2, 2))
        return pen
    
    def objective(self, coeffs):
        error = self.error(coeffs)
        penalty = self.penalty(coeffs)
        obj_ret = error + self.lambda_cont * penalty
        # print 'Coeffs in [{:.3f}, {:.3f}], mean {:.3e}. ' \
        #     'Loss: {:.4f} + {:.4f} * {:.4f} = {:.3e}'.format(
        #     coeffs.min(), coeffs.max(), coeffs.mean(),
        #     error, self.lambda_cont, penalty, obj_ret)
        if obj_ret < 0:
            raise RuntimeError('Objective value < 0. Negative lambda?')
        self.last_error = error
        self.last_regul_penalty = penalty
        self.last_objective = obj_ret
        return obj_ret

    def gradient(self, coeffs):
        coeffs = coeffs.reshape(self.num_models, self.p)
        partials = np.zeros_like(coeffs)
        self.last_error_gradient = np.zeros_like(coeffs)
        self.last_regul_gradient = np.zeros_like(coeffs)
        for i in range(self.num_models):
            err_grad = np.dot(self.design[i].T, np.dot(self.design[i], coeffs[i]) - self.observed[i].flat)
            forward = coeffs[i] - coeffs[i+1] if i < self.num_models - 1 else 0
            backward = coeffs[i] - coeffs[i-1] if i > 0 else 0
            self.last_error_gradient[i] = err_grad
            self.last_regul_gradient[i] = self.lambda_cont * (forward + backward)
            partials[i] = err_grad + self.last_regul_gradient[i]
        return partials.flatten()

    def _get_initial_coeffs(self):
        return np.random.normal(size=self.num_models * self.p)
    
    def estimate(self, coeffs0=None, verbose=False, full_ret=False):
        """Estimate linear regression coefficients for each sub-model.

        coeffs0: Initial coefficients (guesses). Standard random normals
        are used if not provided.  

        verbose: If True, prints the enitre contents returned from
        optimization.

        if full_ret is True, returns the enitre contents returned from
        optimization. Else returns only the estimated regressors.

        """
        if coeffs0 is None:
            coeffs0 = self._get_initial_coeffs()
        optim = scipy.optimize.fmin_l_bfgs_b
        xopt = optim(self.objective, x0=coeffs0,
                     pgtol=1e-10, fprime=self.gradient)
        # optim = scipy.optimize.fmin_tnc
        # xopt = optim(self.objective, x0=coeffs0, fprime=self.gradient,
        #              epsilon=1e-08, messages=15, maxCGit=-1,
        #              maxfun=, eta=-1, stepmx=0, accuracy=0, fmin=0,
        #              ftol=-1, xtol=-1, pgtol=-1, rescale=-1, disp=None, callback=None)
        if verbose:
            print 'Optimization complete. On last iteration, error e={}, '\
                'regularization penalty r={}, objective e + {}*r={}'.format(
                    self.last_error, self.last_regul_penalty,
                    self.lambda_cont, self.last_objective)
            #print 'Return from optimization:', xopt
        if full_ret:
            return xopt
        else:
            return xopt[0].reshape(self.num_models, self.p)

class ClosedSmoothVectorLinregEstimator(SmoothVectorLinregEstimator):

    def __init__(self, *args, **kwargs):
        SmoothVectorLinregEstimator.__init__(self, *args, **kwargs)
        self.max_iterations = 10000
        self.min_obj_change = 0.00001

    def estimate(self, coeffs0=None, verbose=False, full_ret=False):
        """Iterative estimation with closed-form solution of each sub-model.

        """
        if coeffs0 is None:
            coeffs0 = self._get_initial_coeffs()
        coeffs = coeffs0.reshape(self.num_models, self.p).copy()
        a = [np.dot(d.T, d) + 2 * self.lambda_cont * np.identity(self.p) \
             for d in self.design]
        xty = [np.dot(d.T, o) for d, o in zip(self.design, self.observed)]
        prev_objective = None
        for iteration in range(self.max_iterations):
            for i in range(self.num_models):
                forward = np.atleast_2d(coeffs[i+1]).T if i < self.num_models - 1 else 0
                backward = np.atleast_2d(coeffs[i-1]).T if i > 0 else 0
                b = xty[i] + self.lambda_cont * (forward + backward)
                coeffs[i] = np.linalg.solve(a[i], b)[0].flatten()
                #coeffs[i] = np.dot(np.linalg.inv(a[i]), b).flatten()
            objective = self.objective(coeffs)
            if prev_objective is not None:
                try:
                    change = (prev_objective - objective) / objective
                except ArithmeticError:
                    change = 0
                print 'Iteration {}, objective value {:.3e}, change {:.4f}'.format(iteration, objective, change)
                if abs(change) < self.min_obj_change:
                    print 'Threshold for minimum change of objective value reached on iteration {}.'.format(iteration)
                    break
            else:
                print 'Iteration {}, objective value {:.3e}'.format(iteration, objective)
            prev_objective = objective
        if verbose:
            print 'Optimization complete. On last iteration, error e={}, '\
                'regularization penalty r={}, objective e + {}*r={}'.format(
                    self.error(coeffs), self.penalty(coeffs),
                    self.lambda_cont, self.objective(coeffs))
        if full_ret:
            print 'full_ret argument ignored in this subclass.'
        return coeffs
    
    
def ridge_lstsq(x, y, ridge=1e-6):
    return np.linalg.solve(
        np.dot(x.T, x) + ridge * np.identity(x.shape[1]),
        np.dot(x.T, y))

def get_vector_lstsq_coeffs(design, observed):
    '''Get the least squares solution for each submodel independently of the
    others.

    '''
    # Note: ridge_lstsq does NOT return a tuple, as linalg.lstsq does.
    # coeffs0 = [np.linalg.lstsq(d, o)[0] for d, o in \
    coeffs0 = [ridge_lstsq(d, o) for d, o in \
                   zip(design, observed)]
    return np.array([a.flatten() for a in coeffs0])

def get_single_model_lstsq_coeffs(design, observed):
    '''Get the least squares solution when all submodels have the same
    parameters. This code only makes sense when using relative lags, as
    the vector model is not an extension of the single-' instance model
    when using absolute lags.

    '''
    # Note: ridge_lstsq does NOT return a tuple, as linalg.lstsq does.
    #coeffs0 = np.linalg.lstsq(np.vstack(design), np.vstack(observed))[0]
    coeffs0 = ridge_lstsq(np.vstack(design), np.vstack(observed))
    num_models = design.shape[0]
    return np.tile(coeffs0.flatten(), (num_models, 1))

def get_lstsq_initial_coeffs(design, observed, estimator):
    coeffs0_vec = get_vector_lstsq_coeffs(design, observed)
    coeffs0_one = get_single_model_lstsq_coeffs(design, observed)
    obj_vec = estimator.objective(coeffs0_vec)
    obj_one = estimator.objective(coeffs0_one)
    coeffs0 = coeffs0_one if obj_one < obj_vec else coeffs0_vec
    print 'Vector-model LS loss: {}, single-model LS loss: {}. '\
        'Using {} model, lambda={}'.format(
            obj_vec, obj_one, 'single' if obj_one < obj_vec else 'vector',
            estimator.lambda_cont)
    return coeffs0
    # print 'Vector-model LS loss: {}, single-model LS loss: {}. '\
    #     'Always using {} model, lambda={}'.format(
    #         obj_vec, obj_one, 'single',
    #         estimator.lambda_cont)
    # return coeffs0_one

    
class SmoothVectorARPredictor(object):
    """Vector AR model estimation and prediction for time
    series. A regularization parameter controls the similarity in
    regressors between sub-models.

    """
    
    def __init__(self, series, num_models, lags_2d,
                 out_cols=[0], relative_lags=True, add_bias=True):
        self.series = series.copy()
        self.relative_lags = relative_lags
        self.add_bias = add_bias
        self.out_cols = out_cols
        self.lags_2d = lags_2d
        self.out_cols = out_cols
        self.exo_cols = np.delete(np.arange(series.shape[1]), out_cols)
        self.num_models = num_models
        self.estimator_class = SmoothVectorLinregEstimator
        #self.estimator_class = ClosedSmoothVectorLinregEstimator
        self.design, self.observed, self.vlags = make_vector_ar_dataset(
            series, lags_2d=lags_2d, num_models=num_models,
            out_cols=out_cols, relative_lags=relative_lags, add_bias=add_bias)
        self._next_model = 0

    def _get_initial_coeffs_closed_iterative(self):
        print 'Initiating with all zeros. This should amount to starting out with ridge regression.'
        nm, N, p = self.design.shape
        return np.zeros(nm * p)
    
    def _get_initial_coeffs(self):
        return get_lstsq_initial_coeffs(self.design, self.observed, self.estim)
        #return self._get_initial_coeffs_closed_iterative()
    
    def estimate(self, lambda_cont):
        """Estimate AR parameters for the given regularization parameter
        'lambda_cont'. Returns the estimated parameters."""
        self.estim = self.estimator_class(
            self.design, self.observed, lambda_cont=lambda_cont)
        coeffs0 = self._get_initial_coeffs()
        # print 'Skipping optimization, using coeffs0'
        # self.ar_params = coeffs0.reshape(self.estim.num_models, self.estim.p)
        self.ar_params = self.estim.estimate(coeffs0=coeffs0, verbose=True)
        return self.ar_params

    def _check_pred_inputs(self, exo_series, prediction_steps):
        """Check that one of the two are not None. Return inferred or given
        prediction steps."""
        num_exo = self.series.shape[1] - len(self.out_cols)
        if num_exo == 0:
            if prediction_steps is None:
                raise RuntimeError('prediction_steps must be given when there is no exogenuous data.')
        else:
            if exo_series is None:
                raise RuntimeError('Exogenuous data must be supplied for prediction to take place.')
            prediction_steps = len(exo_series)
        return prediction_steps

    def predict(self, exo_series=None, prediction_steps=None, start_model=0):
        """Perform a prediction with the already estimated AR parameters. Either
        of 'exo_series' or 'prediction_steps' must be given. If the
        model uses exogenuous data, these MUST be supplied, and
        prediction_steps will be inferred from the length of the exo
        series (prediction_steps parameter will be ignored). If no exo
        data are used in the model, prediction_steps must be defined.

        start_model is the index of the sub-model that is used
        first. Sub-models are then cycled as would be expected.
        """
        prediction_steps = self._check_pred_inputs(exo_series, prediction_steps)
        new_data = np.zeros((prediction_steps, self.series.shape[1]))
        if exo_series is not None:
            new_data[:, self.exo_cols] = exo_series
        self._pred_series = np.concatenate((self.series, new_data))
        for step in range(prediction_steps):
            i = len(self.series) + step
            im = start_model + step % self.num_models
            lag_offset = 0
            for col, lags in enumerate(self.vlags[im]):
                nlags = len(lags)
                if nlags > 0:
                    self._pred_series[i, self.out_cols] \
                        += np.dot(self._pred_series[i-lags + (self.num_models - im - 1), col],
                                  self.ar_params[im, lag_offset:lag_offset+nlags])
                    lag_offset += nlags
            if self.add_bias:
                self._pred_series[i, self.out_cols] += self.ar_params[im, -1]
        self._next_model = im + 1
        return self._pred_series[-prediction_steps:, self.out_cols]

    def predict_and_update(self, exo_series=None, prediction_steps=None):
        """Make a prediction and store the prediction along with the original
        time series and predictions from previous calls to
        predict_and_update."""
        prediction = self.predict(exo_series, prediction_steps, self._next_model)
        self.series = self._pred_series
        return prediction


class VanillaVectorPredictor(object):
    """Estimation and prediction for of load time series with the vector
    version of Tao Hong's vanilla benchmark. A regularization parameter
    controls the similarity in regressors between sub-models.

    """

    def __init__(self, data):
        self.data = data.copy()
        self.num_models = 24
        self.estimator_class = SmoothVectorLinregEstimator
        self.design, self.observed = make_vanilla_vector_dataset(self.data)
        self._next_model = 0

    def _get_initial_coeffs(self):
        return get_lstsq_initial_coeffs(self.design, self.observed, self.estim)
    
    def estimate(self, lambda_cont):
        """Estimate AR parameters for the given regularization parameter
        'lambda_cont'. Returns the estimated parameters."""
        self.estim = self.estimator_class(
            self.design, self.observed, lambda_cont=lambda_cont)
        coeffs0 = self._get_initial_coeffs()
        # print 'Skipping optimization, using coeffs0'
        # self.ar_params = coeffs0.reshape(self.estim.num_models, self.estim.p)
        self.ar_params = self.estim.estimate(coeffs0=coeffs0, verbose=True)
        return self.ar_params

    def predict(self, data):
        """Perform a prediction with the already estimated AR parameters. 'data'
        must be a Pandas Dataframe with time series index and a column
        named 'Temperature', which holds temperatures for the period
        that shall be predicted.

        """
        design, _ = make_vanilla_vector_dataset(data)
        prediction = np.zeros(len(data))
        for model in range(self.num_models):
            prediction[model::self.num_models] \
                = np.dot(design[model], self.ar_params[model])
        return pd.Series(data=prediction, index=data.index)

    
def make_vector_ar_test_series(noise_sd=0.4, num_days=4, num_models=2,
                               seed=None, relative_lags=True, exo=True):
    if seed is not None:
        np.random.seed(seed)
    model = np.array([[[0.25, 0.5], [0.2, 0.1]],
                      [[0.4, 0.2], [0.3, 0.3]]])
    if num_models > model.shape[0]:
        raise RuntimeError('At most {} models supported by this function.'.format(models.shape[0]))
    model = model[0:num_models, :]
    ar_order = 2
    N = num_models * num_days + ar_order
    series = np.empty((N, 2))
    series[:ar_order, 0] = 1 + 0.2 * np.arange(ar_order)
    series[ar_order:, 0] = np.nan
    series[:, 1] = np.arange(1, len(series)+1) % 3 # Exo series is [1, 2, 0, 1, 2, 0, ...]
    for day in range(num_days):
        for sub_idx, (sub_endo, sub_exo) in enumerate(model):
            i = ar_order + day * num_models + sub_idx
            series[i, 0] = np.dot(series[i-ar_order:i,0], sub_endo[::-1]) \
                           + np.random.normal(1) * noise_sd
            if exo:
                series[i, 0] += np.dot(series[i+1-ar_order:i+1,1], sub_exo[::-1])
    return series, model

def test_vector_linreg_estimator(noise_sd=0, num_days=10):
    data, model = make_vector_ar_test_series(noise_sd=noise_sd, num_models=2, num_days=num_days)
    design, observed, vlags = make_vector_ar_dataset(
        data, lags_2d=[[1, 2],[0, 1]], num_models=2, 
        out_cols=[0], relative_lags=True, add_bias=False)
    loglambdas = np.logspace(1, 11, 500) / 1e3
    log_opts = [SmoothVectorLinregEstimator(
        design, observed, lambda_cont=lc).estimate().flatten()
                for lc in loglambdas]
    log_opts = np.array(log_opts)
    x = np.vstack([loglambdas for _ in range(log_opts.shape[1])]).T
    import matplotlib.pyplot as plt
    fig = plt.figure()
    plt.suptitle('Regularized regression on 2-vector ARX(2,2), {} "days" with noise {}'.format(num_days, noise_sd))
    ax = fig.add_subplot(211)
    ax.plot(x, log_opts, '+-')
    ax.set_xscale('log')
    ax.set_ylabel('Regression coefficients')
    ax.set_xlabel('Lambda')
    ax.set_xlim(x.min(), x.max())
    ax.set_title('Regressors as a function of lambda')
    lsp = np.linalg.lstsq(np.vstack(design),
                          np.vstack(observed))[0]
    ax.plot(x[:,:len(lsp)], np.ones((x.shape[0], len(lsp))) * lsp.T, 'k-')
    ax = fig.add_subplot(212)
    ax.plot(data[:,0])
    ax.set_title('Series used for estimation')
    ax.set_xlim(0, len(data))
    ax.set_xlabel('"Hours" (= "days" * num_models)')
    print 'Showing plot.'
    sys.stdout.flush()
    plt.show()

def test_a_lot_of_vector_linreg_estimators():
    for d in [5, 10, 100, 1000]:
        for n in [0, 10, 100]:
            rr.test_vector_linreg_estimator(num_days=d, noise_sd=n)

if __name__ == '__main__':
    import unittest
    import numpy.testing as npt

    class DatasetTester(unittest.TestCase):
        def setUp(self):
            self.series = np.vstack([np.arange(1, 11), np.arange(11, 21),
                             np.arange(-1, -11, -1), np.arange(-11, -21, -1)]).T
            self.lags = [[1, 2, 3], [2, 3], [0, 1], [1, 3]]
            
        def test_make_vector_ar_dataset_relative(self):
            design, observed, vlags \
                = make_vector_ar_dataset(self.series, self.lags, num_models=3,
                                         out_cols=[0, 1], relative_lags=True)
            sub_1_design = np.array([[4, 3, 2, 13, 12, -5, -4, -14, -12, 1],
                                     [7, 6, 5, 16, 15, -8, -7, -17, -15, 1]])
            sub_1_observed = np.array([[5, 15], [8, 18]])
            step_design = np.array([[1, 1, 1, 1, 1, -1, -1, -1 ,-1, 0],
                                    [1, 1, 1, 1, 1, -1, -1, -1 ,-1, 0]])
            step_observed = np.array([[1, 1], [1, 1]])
            npt.assert_array_equal(design[0], sub_1_design)
            npt.assert_array_equal(observed[0], sub_1_observed)
            npt.assert_array_equal(design[1], sub_1_design + step_design)
            npt.assert_array_equal(observed[1], sub_1_observed + step_observed)
            npt.assert_array_equal(design[2], sub_1_design + 2 * step_design)
            npt.assert_array_equal(observed[2], sub_1_observed + 2 * step_observed)

        def test_make_vector_ar_dataset_absolute(self):
            design, observed, vlags \
                = make_vector_ar_dataset(self.series, self.lags, num_models=3,
                                         out_cols=[0, 1], relative_lags=False)
            sub_1_design = np.array([[4, 3, 2, 13, 12, -7, -6, -16, -14, 1],
                                     [7, 6, 5, 16, 15, -10, -9, -19, -17, 1]])
            design_target = np.array([sub_1_design, sub_1_design, sub_1_design])
            observed_target = np.array([[[5, 15], [8, 18]], [[6, 16], [9, 19]],
                                       [[7, 17], [10, 20]]])
            npt.assert_array_equal(design, design_target)
            npt.assert_array_equal(observed, observed_target)
            
    class LinearRegressionTester(unittest.TestCase):
        def setUp(self):
            self.data, self.model = make_vector_ar_test_series(noise_sd=0, num_models=2, num_days=20)
            self.num_models = self.model.shape[0]
            self.design, self.observed, self.vlags = make_vector_ar_dataset(
                self.data, lags_2d=[[1, 2],[0, 1]], num_models=2, 
                out_cols=[0], relative_lags=True, add_bias=False)

        def test_vector_linreg_no_regularization(self):
            tries = 10
            for i in range(tries):
                params = SmoothVectorLinregEstimator(
                    self.design, self.observed, lambda_cont=0).estimate()
                npt.assert_allclose(params.reshape(self.model.shape), self.model, atol=0.1)

        def test_vector_linreg_max_regularization(self):
            tries = 10
            for i in range(tries):
                params = SmoothVectorLinregEstimator(
                    self.design, self.observed, lambda_cont=1e8).estimate()
                self.assertLess(np.sum(np.abs(params[0,:] - params[1,:])), 1e-5)

        def test_regular_vector_equals_single_equation(self):
            params = SmoothVectorLinregEstimator(
                self.design, self.observed, lambda_cont=1000).estimate()
            target = np.linalg.lstsq(np.vstack(self.design),
                                     np.vstack(self.observed))[0]
            target = np.tile(target.flatten(), (self.num_models, 1))
            npt.assert_allclose(params.flatten(), target.flatten(), atol=0.05)


    class PredictorTester(unittest.TestCase):
        def setUp(self):
            self.num_models = 2
            self.test_days = 3
            self.data, self.model = make_vector_ar_test_series(
                noise_sd=0, num_models=self.num_models, num_days=20)
            self.train = self.data[:-self.test_days*self.num_models,:]
            self.test = self.data[-self.test_days*self.num_models:,:]

        def test_predicted_resembles_test(self):
            svp = SmoothVectorARPredictor(
                self.train, num_models=self.num_models,
                lags_2d=[[1, 2], [0, 1]], add_bias=False)
            svp.estimate(lambda_cont=0)
            svp.ar_params = self.model.reshape(svp.ar_params.shape)
            prediction = svp.predict(exo_series=np.atleast_2d(self.test[:,1]).T,
                                     prediction_steps=self.test_days*self.num_models)
            npt.assert_allclose(prediction, np.atleast_2d(self.test[:,0]).T)

    unittest.main()
