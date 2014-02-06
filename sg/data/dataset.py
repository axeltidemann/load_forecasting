import datetime
from datetime import timedelta as dt

import numpy as np
import pandas as pd
import copy

class Dataset(object):
    def __init__(self, series, period_length, step_length=None):
        """Initialize the dataset with the entire timeseries, and a
        datetime.timedelta indicating the length of each period to be
        extracted.

        If step_length is provided, this should be a datetime.timedelta that
        indicates the step length between each period that may be selected. For
        instance, a step_length of 1 day indicates that all the selected
        periods will start at the same hour of day, even if the dataset has
        higher frequency."""
        self._series = series
        self._period_length = \
          self._convert_timedelta_to_timeseries(period_length)
        if step_length is None:
            self._step_length = 1
        else:
            self._step_length = \
                self._convert_timedelta_to_timeseries(step_length)
        self._num_periods = (len(series) - self._period_length + 1) / \
          self._step_length

    def _get_start_and_end_times(self):
        """Return the start and end times of the time series. End time is the
        start time of the last entry in the series, not the end time, i.e. the
        duration of the last timestep is not included."""
        start_time = self._series.first_valid_index()
        end_time = self._series.last_valid_index()
        if isinstance(start_time, pd.Period):
            # start_time and end_time for a Period seem to be equivalent
            start_time = start_time.start_time
            end_time = end_time.start_time 
        return (start_time, end_time)
        
    def _convert_timedelta_to_timeseries(self, period_length):
        """Return the length of the period (a timedelta) represented as an
        integer, based on the frequency of the dataset."""
        # Calculating this cannot be done using the timeseries frequency, as
        # that falls apart when the frequency is undefined. This method should
        # work for all frequencies, as long as the time step is constant
        # between data points.
        start_time, end_time = self._get_start_and_end_times()
        dt_series = (end_time - start_time) / (len(self._series) - 1)
        if dt_series >= period_length:
            return 1
        else:
            # datetime.timedelta doesn't support division, so count the steps
            # incrementally.
            dt_acc = dt_series
            steps = 1
            while dt_acc < period_length:
                steps += 1
                dt_acc += dt_series
            if dt_acc > period_length:
                msg = "Could not create dataset, failed to convert time " \
                    "period length to a number steps in the time series array. " \
                    "The selected period length (%s) is not a multiple of " \
                    "the time step of the original data set (%s)." % \
                    (period_length, dt_series)
                raise RuntimeError(msg)
            return steps

    @property
    def num_periods(self):
        """The number of selectable periods. This is a read-only property."""
        return self._num_periods

    @property
    def series(self):
        """The entire time series from which dataset periods are selected. This
        is a read-only property."""
        return self._series

    def index_of(self, period_number):
        """Return index in entire time series of period number
        period_number."""
        return period_number * self._step_length
        
    def get_period(self, period_number):
        """Return period number period_number."""
        first = self.index_of(period_number)
        last = first + self._period_length
        return self._series[first:last]

    def get_random_period(self, return_period_number=False):
        """Select a random period of the predefined length. If
        return_period_number, return a tuple consisting of a random period and
        the period number the selected period. Otherwise return only the data."""
        number = np.random.randint(0, self.num_periods)
        data = self.get_period(number)
        if return_period_number:
            return (data, number)
        else:
            return data

    def split(self, ratio=0.5):
        """Splits the current dataset into two datasets defined by the ratio."""
        first = copy.copy(self)
        first._series = first._series[:int(len(first._series)*ratio)]
        last = copy.copy(self)
        last._series = last._series[int(len(last._series)*ratio):]
        return first, last
    
def remove_outlier_set_previous(dataset, outlier_val=0):
    """Set all 'outlier'-valued elements in the dataset to be the value at the
    position before. This routine does not copy the dataset before cleaning.

    If there are several consecutive outliers, they will all be set to the
    preceding non-outlier value."""
    outliers = np.where(dataset[1:] == outlier_val)
    for outlier in outliers[0]:
        dataset[outlier + 1] = dataset[outlier]
    return dataset

if __name__ == "__main__":
    from unittest import main
    main(module="test_" + __file__[:-3])
