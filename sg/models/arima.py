
import numpy as np
import pandas as pd

try:
    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr
    from rpy2.robjects.numpy2ri import numpy2ri
except ImportError as ie:
    print ie
    # rpy2 not trivial to install on Vilje ("R was not built as a library")
    print "rpy2 not installed, R-based predictors won't work."

# It seems the forecast package does some magic/hackery with the (shape of?)
# xreg parameter. Stick to all-forecast or all-vanilla-R, but don't mix
# them. These work:
# predict(arima(x=_, order=_, xreg=xx), newxreg=yy)
# forecast(Arima(x=_, order=_, xreg=xx), xreg=yy)
# predict(Arima(x=_, order=_, xreg=xx), newxreg=yy)
# This doesn't:
# forecast(arima(x=_, order=_, xreg=xx), xreg=yy)
#
# Moreover, on-the-fly conversion from numpy to R (i.e. using R through rpy2
# "as if it were Python") work for the x argument, but not for the xreg
# argument.
class RPredictorBase(object):
    def __init__(self, data, genome, loci, prediction_steps):
        self._data = data
        self._genome = genome
        self._loci = loci
        self._prediction_steps = prediction_steps
        self._setup_R()
        self._setup_datasets(data, genome, loci, prediction_steps)

    def _setup_R(self):
        self._R = ro.r
        ro.conversion.py2ri = numpy2ri

    def _setup_datasets(self, data, genome, loci, prediction_steps):
        (temps, loads) = (data['Temperature'], data['Load'])
        ro.globalenv["loads"] = loads[-genome[loci.hindsight]-prediction_steps:-prediction_steps]
        ro.globalenv["temps_hc"] = temps[-genome[loci.hindsight]-prediction_steps:-prediction_steps]
        ro.globalenv["temps_fc"] = temps[-prediction_steps:]

    def _make_predictor(self):
        raise RuntimeError("Implement this in model-specific subclass.")
    
    def prediction(self):
        self._make_predictor()
        pred_r = self._R("pred$pred")
        prediction = pd.TimeSeries(data=pred_r[-self._prediction_steps:], 
                                   index=self._data.index[-self._prediction_steps:])
        return prediction
        
class RArimaPDQPredictorBase(RPredictorBase):
    def __init__(self, data, genome, loci, prediction_steps):
        RPredictorBase.__init__(self, data, genome, loci, prediction_steps)
        self._setup_ARIMA_params(genome, loci)

    def _setup_ARIMA_params(self, genome, loci):
        ro.globalenv["order"] = np.array([genome[loci.AR_order], 
                                          genome[loci.I_order],
                                          genome[loci.MA_order]])

def arima_with_weather(data, genome, loci, prediction_steps, spinup=0):
    class RArimaXRegPredictor(RArimaPDQPredictorBase):
        def _make_predictor(self):
            self._R('model <- arima(x=loads, order=order, xreg=temps_hc, method="ML")')
            self._R("pred <- predict(model, newxreg=temps_fc)")
    predictor = RArimaXRegPredictor(data, genome, loci, prediction_steps)
    return predictor.prediction()
    
def seasonal_arima_with_weather(data, genome, loci, prediction_steps, spinup=0):
    class RArimaXRegPredictor(RArimaPDQPredictorBase):
        def _make_predictor(self):
            cmds = ['model <- arima(x=loads, order=order, ' \
                    'seasonal=list(order=c({}, {}, {}), period={}), '\
                    'xreg=temps_hc, method="ML", optim.method="SANN",' \
                    'optim.control=list(maxit=10000))'.format(
                        genome[loci.ssn_AR_order], genome[loci.ssn_I_order],
                        genome[loci.ssn_MA_order], 24),
                    'pred <- predict(model, newxreg=temps_fc)']
            for cmd in cmds:
                #print "Running R command:", cmd
                self._R(cmd)
    predictor = RArimaXRegPredictor(data, genome, loci, prediction_steps)
    return predictor.prediction()

def arima_without_weather(data, genome, loci, prediction_steps, spinup=0):
    class RArimaPredictor(RArimaPDQPredictorBase):
        def _make_predictor(self):
            self._R('model <- arima(x=loads, order=order, method="ML")')
            self._R("pred <- predict(model, n.ahead=%d)" % prediction_steps)
    predictor = RArimaPredictor(data, genome, loci, prediction_steps)
    return predictor.prediction()

class DSHWPredictor(RPredictorBase):
    def _setup_R(self):
        RPredictorBase._setup_R(self)
        importr('forecast')

    def _make_predictor(self):
        self._R("pred <- dshw(y = loads, period1 = 24, period2 = 168, "\
                "h = 24, alpha = {}, beta = {}, gamma = {}, omega = {},"\
                "phi = {}, lambda = NULL, armethod = TRUE)".format(
                    self._genome[self._loci.alpha], 
                    self._genome[self._loci.beta], 
                    self._genome[self._loci.gamma],
                    self._genome[self._loci.omega],
                    self._genome[self._loci.phi]))

    def prediction(self):
        self._make_predictor()
        pred_r = self._R("pred$mean")
        prediction = pd.TimeSeries(data=pred_r[-self._prediction_steps:], 
                                   index=self._data.index[-self._prediction_steps:])
        return prediction

def dshw(data, genome, loci, prediction_steps, spinup=0):
    predictor = DSHWPredictor(data, genome, loci, prediction_steps)
    return predictor.prediction()

def auto_dshw(data, genome, loci, prediction_steps, spinup=0):
    class AutoDSHWPredictor(DSHWPredictor):
        def _make_predictor(self):
            self._R("pred <- dshw(y=loads, period1=24, period2=168, h=24)")

    predictor = AutoDSHWPredictor(data, genome, loci, prediction_steps)
    return predictor.prediction()

def auto_arima_with_weather(data, genome, loci, prediction_steps, spinup=0):
    class RAutoArimaXRegPredictor(RPredictorBase):
        def _make_predictor(self):
            self._R('model <- auto.arima(x=loads, order=order, xreg=temps_hc)')
            self._R("pred <- predict(model, newxreg=temps_fc)")
    predictor = RAutoArimaXRegPredictor(data, genome, loci, prediction_steps)
    return predictor.prediction()
    
def auto_arima_without_weather(data, genome, loci, prediction_steps, spinup=0):
    class RAutoArimaPredictor(RPredictorBase):
        def _make_predictor(self):
            self._R('model <- auto.arima(x=loads, order=order)')
            self._R("pred <- predict(model)")
    predictor = RAutoArimaPredictor(data, genome, loci, prediction_steps)
    return predictor.prediction()
    
def ar_without_weather(data, genome, loci, prediction_steps, spinup=0):
    class RArPredictor(RPredictorBase):
        def _nodiff_pred(self):
            self._R("model <- ar(loads, aic=FALSE, order.max=%d)" % genome[loci.AR_order])
            self._R("pred <- predict(model, n.ahead=%d)" % prediction_steps)
            return self._R("pred$pred")
        
        def _diff_pred(self, diff_order):
            self._R("loads_nodiff <- loads")
            self._R("loads <- diff(loads, differences=%d)" % diff_order)
            self._R("model <- ar(loads, aic=FALSE, order.max=%d)" % genome[loci.AR_order])
            self._R("pred <- predict(model, n.ahead=%d)" % prediction_steps)
            self._R("ll <- length(loads_nodiff)")
            return self._R("diffinv(pred$pred, differences=%d, xi=loads_nodiff[(ll-%d):ll])" \
              % (diff_order, diff_order-1))

        def prediction(self):
            diff_order = genome[loci.diff]
            if diff_order > 0:
                pred_r = self._diff_pred(diff_order)
            else:
                pred_r = self._nodiff_pred()
            prediction = pd.TimeSeries(data=pred_r[-self._prediction_steps:], 
                                       index=self._data.index[-self._prediction_steps:])
            return prediction
                
    predictor = RArPredictor(data, genome, loci, prediction_steps)
    return predictor.prediction()


def ar(data, lags_2d, prediction_steps, stride=1, out_cols=[0]):
    """Calculate autoregressive parameters for the given lags, based on the
    data except for the last prediction_steps elements. Design and response
    matrices are created by extracting every 'stride'th element from each
    column of data. Return prediction_steps/stride predicted values together
    with AR params. Note that there must be a consistency between lags,
    prediction_steps and stride: If the lags are not a multiple of the stride,
    prediction_steps/stride must be <= 1."""
    #ar_order = max([max(lags) for lags in lags_2d]) # Doesn't accept empty lists
    ar_order = int(np.concatenate(lags_2d).max())
    first_pred_idx = data.shape[0] - prediction_steps
    N = (first_pred_idx - ar_order) / stride
    if N < 1:
        raise ValueError(
            "Invalid input data: AR order (%d) higher than hindsight (%d)." % \
            (ar_order, first_pred_idx))
    p = sum([len(lags) for lags in lags_2d])
    design = np.empty((N, p+1))
    observed = np.empty((N, len(out_cols)))
    offset = 0
    start_at_stride_mult = ar_order + (first_pred_idx - ar_order) % stride
    for lags, col in zip(lags_2d, range(len(lags_2d))):
        nlags = len(lags)
        if nlags > 0:
            for (i, j) in zip(range(N), 
                              range(start_at_stride_mult, first_pred_idx, stride)):
                design[i, offset:offset+nlags] = data[j - lags, col]
                observed[i, :] = data[j, out_cols]
        offset += nlags

    design[:, p] = 1
    ar_params = np.linalg.lstsq(design, observed)[0]
    
    observed_next_day = data[-prediction_steps:].copy()
    design = np.empty(p + 1)
    for i in range(first_pred_idx, data.shape[0], stride):
        offset = 0
        for lags, col in zip(lags_2d, range(len(lags_2d))):
            nlags = len(lags)
            if nlags > 0:
                design[offset:offset+nlags] = data[i-lags, col]
                offset += nlags
        design[-1] = 1
        data[i, out_cols] = np.dot(design, ar_params)
    prediction = data[-prediction_steps:, out_cols].copy()
    data[-prediction_steps:] = observed_next_day
    return prediction, ar_params

def _ar_ga_with_lags(data, genome, loci, prediction_steps, lags_2d):
    pred, _ = ar(
        data[max(-len(data), -genome[loci.hindsight]-prediction_steps):].values, 
        lags_2d, 
        prediction_steps,
        out_cols=[data.columns.tolist().index('Load')])
    if pred.shape != (prediction_steps, 1):
        raise RuntimeError("Prediction made by AR model has wrong shape!")
    pred.shape = (pred.shape[0],) # TimeSeries expects 1D data.
    return pd.TimeSeries(data=pred, index=data[-prediction_steps:].index)

def _check_column_order(data, lags_2d):
    columns = data.columns.tolist()
    load_idx = columns.index('Load')
    temp_idx = columns.index('Temperature')
    if load_idx == 1 and temp_idx == 0:
        lags_2d = [lags_2d[1], lags_2d[0]]
    elif load_idx != 0 or temp_idx != 1:
        raise RuntimeError("Data columns not as expected by AR bitmap function.")
    return lags_2d

def lags_from_order_ga(data, genome, loci):
    lags_2d = [np.arange(1, genome[loci.AR_order] + 1),
               np.arange(0, genome[loci.EXO_order] + 1)]
    return _check_column_order(data, lags_2d)
    
def lags_from_bitmap_ga(data, genome, loci):
    """Create a list of n lists of lags, each list based on a bitmap in the list
    'bitmaps'."""
    lags_2d = [np.where(genome[loci.AR_lags] == 1)[0] + 1,
               np.where(genome[loci.EXO_lags] == 1)[0]]
    return _check_column_order(data, lags_2d)

def ar_ga(data, genome, loci, prediction_steps):
    lags_2d = lags_from_order_ga(data, genome, loci)
    return _ar_ga_with_lags(data, genome, loci, prediction_steps, lags_2d)

def bitmapped_ar_ga(data, genome, loci, prediction_steps):
    lags_2d = lags_from_bitmap_ga(data, genome, loci)
    return _ar_ga_with_lags(data, genome, loci, prediction_steps, lags_2d)

def vector_ar(data, lags_2d, prediction_steps, out_cols=[0]):
    '''Vector version of AR. All columns not listed in out_cols is assumed
    to be exo variables, hence observable also in the prediction period.

    '''
    prediction = []
    ar_params = []
    for step in range(prediction_steps, 0, -1):
        end = len(data) - step + 1
        pred, arp = ar(
            data[:end], 
            lags_2d, 
            1,
            stride=prediction_steps,
            out_cols=out_cols)
        prediction.append(pred[0,0])
        ar_params.append(arp)
        for out_col in out_cols:
            lags_2d[out_col] += 1 # Shift AR lags to avoid dependence on previous model estimates.
    return prediction, ar_params

def _hourbyhour_ar_ga_with_lags(data, genome, loci, prediction_steps, lags_2d):
    ar_col = data.columns.tolist().index('Load')
    start = max(0, len(data) - genome[loci.hindsight] - prediction_steps)
    prediction, _ = vector_ar(data[start:].values, lags_2d, prediction_steps, [ar_col])
    return pd.TimeSeries(data=prediction, index=data[-prediction_steps:].index)

def hourbyhour_ar_ga(data, genome, loci, prediction_steps):
    lags_2d = lags_from_order_ga(data, genome, loci)
    return _hourbyhour_ar_ga_with_lags(
        data, genome, loci, prediction_steps, lags_2d)

def bitmapped_hourbyhour_ar_ga(data, genome, loci, prediction_steps):
    lags_2d = lags_from_bitmap_ga(data, genome, loci)
    return _hourbyhour_ar_ga_with_lags(
        data, genome, loci, prediction_steps, lags_2d)
    
if __name__ == "__main__":
    from unittest import main
    main(module="test_" + __file__[:-3])

# def ar2(data, exo, data_lags, exo_lags):
#     '''Calculate autoregressive parameters for the given lags. Returns AR
#     params for subsequent prediction. Regression coefficients are
#     calculated for all columns of 'data', and none of the columns in
#     'exo'. Lags should contain one list of lags for each variable in the
#     dataset. Lag 0 is now, so in order to e.g. predict the load 6 hours
#     ahead, using the two last observed hours of load plus the
#     temperature forecast for the prediction hour, set lags to:
#        data_lags = [[6, 7, 8]]
#        exo_lags = [[0]].

#     '''
#     all_lags = data_lags + exo_lags
#     ar_order = int(np.concatenate(all_lags).max())
#     num_obs = data.shape[0]
#     N = num_obs - ar_order
#     if N < 1:
#         raise ValueError(
#             "Invalid input data: AR order (%d) higher than number of observations (%d)." % \
#             (ar_order, num_obs))
#     p = sum([len(lags) for lags in all_lags])
#     design = np.empty((N, p+1))
#     observed = np.empty((N, data.shape[1]))
#     offset = 0
#     for variable in range(len(data_lags)):
#         for lags, col in zip(data_lags, range(len(data_lags))):
#         nlags = len(lags)
#         if nlags > 0:
#             for (i, j) in zip(range(N), 
#                               range(start_at_stride_mult, first_pred_idx, stride)):
#                 design[i, offset:offset+nlags] = data[j - lags, col]
#                 observed[i, :] = data[j, out_cols]
#         offset += nlags

#     design[:, p] = 1
#     ar_params = np.linalg.lstsq(design, observed)[0]
    
#     observed_next_day = data[-prediction_steps:].copy()
#     design = np.empty(p + 1)
#     for i in range(first_pred_idx, data.shape[0], stride):
#         offset = 0
#         for lags, col in zip(lags_2d, range(len(lags_2d))):
#             nlags = len(lags)
#             if nlags > 0:
#                 design[offset:offset+nlags] = data[i-lags, col]
#                 offset += nlags
#         design[-1] = 1
#         data[i, out_cols] = np.dot(design, ar_params)
#     prediction = data[-prediction_steps:, out_cols].copy()
#     data[-prediction_steps:] = observed_next_day
#     return prediction, ar_params

    
