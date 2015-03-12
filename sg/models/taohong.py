import numpy as np
import pandas as pd

def vanilla(data, genome, loci, prediction_steps, spinup=0):
    """ Tao Hong's Vanilla Benchmark method, as described in
    "A Naive Multipe Linear Regression Benchmark for Short Term Load
    Forecasting" (Hong, 2011)

    Note: this model is specifically built for hourly based predictions,
    and will not work properly otherwise."""
    
    temps = data.Temperature

    num_params = 2 + 7*24 + 4*12 + 3*24
    print 'Created model with', num_params, 'parameters.'
    a = np.zeros((len(data), num_params))

    for i in range(a.shape[0]):
        day, hour, month = temps.index[i].dayofweek, temps.index[i].hour, temps.index[i].month
        month -= 1
        tmp = temps[i]
        trend = (temps.index[i].value - temps.index[0].value)/(3600*10**9) + 1
        a[i, 0:2] = [ 1, trend]
        offset = 2
        a[i, offset + day*hour] = 1
        offset += 7*24
        a[i, offset + month] = 1
        offset += 12
        a[i, offset + month] = tmp
        offset += 12
        a[i, offset + month] = tmp**2
        offset += 12
        a[i, offset + month] = tmp**3
        offset += 12
        a[i, offset + hour] = tmp
        offset += 24
        a[i, offset + hour] = tmp**2
        offset += 24
        a[i, offset + hour] = tmp**3
        assert(offset + 24 == num_params)
    x,_,_,_ = np.linalg.lstsq(a[:-prediction_steps], data.Load[:-prediction_steps])

    return pd.Series(data=np.dot(a[-prediction_steps:],x), 
                    index=data.index[-prediction_steps:])

