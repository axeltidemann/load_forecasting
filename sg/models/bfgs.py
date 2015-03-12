import copy
import sys

import Oger
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mdp
import sklearn

import sg.data.bchydro as bc
import sg.data.gefcom2012 as gefcom
import sg.models.splines as sp
import sg.utils

bcall = None
dist = None
gef = None
gef_1 = None
gef_3 = None
gef_sys = None

def load():
    global bcall, dist, gef, gef_1, gef_3, gef_sys
    bcall = bc.load_remove_zeros()
    bcall = bcall / 1000
    dist = ul.total_experiment_load()[0]['Load']
    gef = gefcom.load_solved_non_nan().load
    gef_1 = gef['zone_1']
    gef_3 = gef['zone_3']
    gef_sys = gef.sum(axis=1)

class FullOutBFGSTrainer(Oger.gradient.BFGSTrainer):
    def train(self, func, x0):
        self.kwargs['retall'] = True
        self.kwargs['full_output'] = True
        update, _, _, _, _, _, warnflag, allvecs = \
                Oger.gradient.BFGSTrainer.train(self, func, x0)
        return update, warnflag, allvecs
        
class FullOutCGTrainer(Oger.gradient.CGTrainer):
    def train(self, func, x0):
        self.kwargs['retall'] = True
        self.kwargs['full_output'] = True
        update, _, _, _, warnflag, allvecs = \
                Oger.gradient.CGTrainer.train(self, func, x0)
        return update, warnflag, allvecs


class ValidationStopBackpropNode(Oger.gradient.BackpropNode):
    def __init__(self, gflow, gtrainer, loss_func=None, derror=None, n_epochs=1, dtype='float64'):
        Oger.gradient.BackpropNode.__init__(self, gflow, gtrainer, loss_func, derror, n_epochs, dtype)

    @mdp.with_extension('gradient')
    def _train(self, x, *args, **kwargs):
        """Update the parameters according to the input 'x' and target output 't'."""
        if (len(args) > 0):
            t = args[0]
        else:
            t = kwargs.get('t')
        def func(params):
            return self._objective(x, t, params)

        # Full output is: xopt, fopt, gopt, Bopt, func_calls, grad_calls, warnflag, allvecs
        maxiter_exceeded = 1
        num_tries = 0
        max_tries = 5
        while True:
            update, warnflag, self._params_per_epoch = \
                self.gtrainer.train(func, self._params())
            if warnflag == maxiter_exceeded:
                break
            num_tries += 1
            if num_tries == max_tries:
                print >>sys.stderr, 'Too many tries, bailing out...'
                break
            print >>sys.stderr, 'Training aborted before maximum epochs reached. Trying again...'
            # Re-initialization plays badly with iterative p-norm loss optimization.
            # for node in self.gflow:
            #     node.initialize()
        self._final_params = update
        self._set_params(update)

    @mdp.with_extension('gradient')
    def validation_stop(self, x_valid, y_valid):
        num_epochs = len(self._params_per_epoch)
        losses = np.empty(num_epochs)
        for epoch in range(num_epochs):
            self.reset_params(epoch)
            yhat = self.gflow.execute(x_valid)
            losses[epoch] = self.loss_func(yhat, y_valid)
        best_epoch = np.where(losses == losses.min())[0]
        self.reset_params(best_epoch)
        return best_epoch

    @mdp.with_extension('gradient')
    def reset_params(self, epoch=None):
        if epoch is None:
            self._set_params(self._final_params)
        else:
            self._set_params(self._params_per_epoch[epoch])


def mse_derror(yhat, y):
    return 2 * (yhat - y)
    
def mae_derror(yhat, y):
    derr = np.ones(yhat.shape)
    derr[yhat < y] = -1
    return derr

def mae_loss(yhat, y):
    import ipdb; ipdb.set_trace()
    return sg.utils.mae(yhat, y)
    
def mape_loss(scaler, yhat, y):
    yhs = scaler.inverse_transform(yhat)
    ys = scaler.inverse_transform(y)
    mape = sg.utils.mape_skip_zeros(yhs, ys)[0]
    return mape

def noscale_mape_loss(yhat, y):
    return sg.utils.mape_skip_zeros(yhat, y)[0]

def mape_derror(scaler, yhat, y):
    #derr = scaler.transform(mae_derror(yhat, y) / scaler.inverse_transform(y))
    y_real = scaler.inverse_transform(y)
    derr = mae_derror(yhat, y) / (y_real / y_real.max())
    return derr

def mixmape_derror(scaler, yhat, y):
    #derr = scaler.transform(mae_derror(yhat, y) / scaler.inverse_transform(y))
    y_real = scaler.inverse_transform(y)
    derr = mse_derror(yhat, y) / (y_real / y_real.max())
    return derr

def noscale_mixmape_derror(yhat, y):
    return mse_derror(yhat, y) / y

def p_norm_loss(p, yhat, y):
    if p < 1 or p > 2:
        raise RuntimeError('Invalid value for p, should be in [1, 2]: {}'.format(p))
    diff = yhat - y
    return np.mean(np.power(np.abs(diff), p) / np.power(y, 2-p))

def p_norm_derror(p, yhat, y):
    if p < 1 or p > 2:
        raise RuntimeError('Invalid value for p, should be in [1, 2]: {}'.format(p))
    diff = yhat - y
    return p * np.power(np.abs(diff), p - 1) * np.sign(diff) / np.power(y, 2-p)
    

class BFGSPredictor(object):
    def __init__(self, nodes_per_layer):
        """'nodes_per_layer' is a list containing the number of nodes per hidden
        layer.

        """
        self._flow = None
        self._nodes_per_layer = nodes_per_layer
        self._x_norm = None
        self._y_norm = None
        self._err_measure = None
        self._loss_func = None
        self._derror_func = None
        self._bpnode = None
        
    def _create_flow(self, num_inputs, num_outputs):
        transfer_func = Oger.utils.TanhFunction
        layers = [Oger.nodes.PerceptronNode(num_inputs, self._nodes_per_layer[0],
                                            transfer_func=transfer_func)]
        for prev_layer, this_layer in \
            zip(self._nodes_per_layer[0:-1], self._nodes_per_layer[1:]):
            layers += [Oger.nodes.PerceptronNode(prev_layer, this_layer,
                                                 transfer_func=transfer_func)]
        layers += [Oger.nodes.PerceptronNode(self._nodes_per_layer[-1], num_outputs)]
        self._flow = mdp.Flow(layers)

    def _init_loss_funcs(self):
        if self._err_measure == 'mse':
            self._loss_func = Oger.utils.mse
            self._derror_func = mse_derror
        elif self._err_measure == 'mae':
            self._loss_func = sg.utils.mae
            self._derror_func = mae_derror
        elif self._err_measure == 'mape':
            self._loss_func = lambda yhat, y: mape_loss(self._y_norm, yhat, y)
            self._derror_func = lambda yhat, y: mape_derror(self._y_norm, yhat, y)
        elif self._err_measure == 'mix':
            self._loss_func = lambda yhat, y: mape_loss(self._y_norm, yhat, y)
            self._derror_func = lambda yhat, y: mixmape_derror(self._y_norm, yhat, y)
        elif self._err_measure == 'noscale_mix':
            self._loss_func = noscale_mape_loss
            self._derror_func = noscale_mixmape_derror
        else:
            raise RuntimeError('Unrecognized error measure: {}'.format(self._err_measure))

    def _create_trainer(self, epochs):
        trainer = FullOutCGTrainer(maxiter=epochs, gtol=0)
        self._bpnode = ValidationStopBackpropNode(\
            self._flow, trainer, loss_func=self._loss_func, derror=self._derror_func)

    # def _check_fix_rank(self, data):
    #     if np.rank(data) == 1:
    #         data.shape += (1,)

    def _as_2d_array(self, data):
        if isinstance(data, pd.DataFrame):
            return data.values
        elif isinstance(data, pd.Series):
            return np.atleast_2d(data.values).T
        else:
            return data

    def _init_training(self, data):
        x, y = [self._as_2d_array(d) for d in data]
        self._x_norm = sklearn.preprocessing.MinMaxScaler()
        x_in = self._x_norm.fit_transform(x)
        self._y_norm = sklearn.preprocessing.MinMaxScaler()
        self._init_output_scaler(y)
        y_in = self._y_norm.fit_transform(y)
        self._create_flow(x.shape[1], y.shape[1])
        return x_in, y_in

    def _init_output_scaler(self, y):
        if self._err_measure == 'noscale_mix':
            self._y_norm.set_params(feature_range=[y.min()/y.max(), 1])
        
    def train(self, err_measure, data, epochs, vdata=None):
        """Train the network on 'data' using error measure 'err_measure', one of
        'mse', 'mae' or 'mape'. 'data' should be a tuple (x,
        y). 'vdata', if provided, is the validation data used to stop model estimation.

        """
        self._err_measure = err_measure
        x_in, y_in = self._init_training(data)
        self._init_loss_funcs()
        self._create_trainer(epochs)
        self._bpnode.train(x_in, y_in)
        if vdata is not None:
            vx, vy = [self._as_2d_array(d) for d in vdata]
            return self._bpnode.validation_stop(self._x_norm.transform(vx),
                                                self._y_norm.transform(vy))

    def execute(self, x):
        x2d = self._as_2d_array(x)
        yhat = self._y_norm.inverse_transform(self._bpnode(self._x_norm.transform(x2d)))
        if isinstance(x, pd.DataFrame):
            yhat.shape = (yhat.shape[0],)
            return pd.Series(yhat, index=x.index)
        else:
            return yhat


class IterativeBFGSPredictor(BFGSPredictor):
    def _init_loss_funcs(self):
        if self._err_measure == 'mpe':
            # self._loss_func = Oger.utils.mse
            # self._derror_func = mse_derror
            self._loss_func = lambda yhat, y: p_norm_loss(self._p, yhat, y)
            self._derror_func = lambda yhat, y: p_norm_derror(self._p, yhat, y)
        else:
            return BFGSPredictor._init_loss_funcs(self)

    def _init_output_scaler(self, y):
        if self._err_measure == 'mpe':
            self._y_norm.set_params(feature_range=[y.min()/y.max(), 1])
        else:
            BFGSPredictor._init_output_scaler(self, y)

    def train(self, err_measure, data, epochs, vdata=None):
        """Train the network on 'data' using error measure 'err_measure', one of
        'mse', 'mae' or 'mape'. 'data' should be a tuple (x,
        y). 'vdata', if provided, is the validation data used to stop model estimation.

        """
        self._err_measure = err_measure
        if err_measure != 'mpe':
            return BFGSPredictor.train(self, err_measure, data, epochs, vdata)
        x_in, y_in = self._init_training(data)
        ps = np.linspace(2, 1, 10)
        #ps = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
        retval = []
        for p in ps:
            self._p = p
            print 'p-norm optimizing with p={}'.format(p)
            self._init_loss_funcs()
            self._create_trainer(epochs)
            self._bpnode.train(x_in, y_in)
            if vdata is not None:
                vx, vy = [self._as_2d_array(d) for d in vdata]
                retval.append(self._bpnode.validation_stop(self._x_norm.transform(vx),
                                                           self._y_norm.transform(vy)))
        if vdata is not None:
            return retval



class VectorModel(object):
    def __init__():
        self._models = None

    def _create_model(self, model_num):
        """Create the model that will forecast hour/period 'model_num' (for
        hourly data, this will be called 24 times).

        """
        pass

    def train(self, err_measure, datas, epochs, vdatas=None):
        """Train the network on 'data' using error measure 'err_measure', one of
        'mse', 'mae' or 'mape'. 'datas' should be a list of tuples (x,
        y). For hourly data, it should contain 24 tuples. 'vdatas', if
        provided, is the validation data used to stop model estimation.

        """
        num_models = len(datas)
        self._models = [self._create_model(i) for i in range(num_models)]
        if vdatas is None:
            vdatas = [None for _ in range(num_models)]
        # return [self._models[i].train(err_measure, d, epochs, v) \
        #         for i, d, v in zip(range(num_models), datas, vdatas)]
        rets = []
        for i, d, v in zip(range(num_models), datas, vdatas):
            print 'Training for hour {}'.format(i)
            print 'Random state:'
            print np.random.get_state()
            rets.append(self._models[i].train(err_measure, d, epochs, v))
        return rets

    def execute(self, xs):
        exec_ret = []
        for model, x in zip(self._models, xs):
            exec_ret.append(model.execute(x))
        return exec_ret


class BFGSVectorModel(VectorModel):
    def __init__(self, nodes_per_layer):
        self._nodes_per_layer = nodes_per_layer

    def _create_model(self, model_num):
        return BFGSPredictor(self._nodes_per_layer)




# def testme():
#     import error_functions as ef
#     gef_1, gef_3, gef_sys = ef.load_gef_dataset()
#     ds = ef.remove_nans(ef.create_ann_dataset(gef_sys))
#     train = ef.convert_to_vectormodel_xy_dataset(ds[:'2007-06'])
#     test = ef.convert_to_vectormodel_xy_dataset(ds['2007-07':])
#     vi, ti = Oger.evaluation.train_test_only(len(test[0][0]), 0.5, random=True)
#     vi = vi[0]
#     ti = ti[0]
#     valid, test = zip(*[((x[vi], y[vi]), (x[ti], y[ti])) for x, y in test])






#     for rule in ['mse', 'mae', 'mape', 'mix']:
#         for i in range(30):
#             print ''
#             print 'Rule {}, iteration {}:'.format(rule, i)
#             model = BFGSVectorModel([8, 3])
#             max_hour = 24
#             epochs = 2000
#             stop_epochs = model.train(rule, train[0:max_hour], epochs, valid[0:max_hour])
#             print 'Stop epochs: ', stop_epochs
#             test_x, test_y = zip(*test)
#             yhat = model.execute(test_x[0:max_hour])
#             print 'Final MAPE:', [sg.utils.mape_skip_zeros(yh, y)[0] for yh, y in zip(yhat, test_y[0:max_hour])]
#             sys.stdout.flush()

# def gridsearch_params
    
# if __name__ == "__main__":
#     testme()
