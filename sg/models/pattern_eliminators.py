import copy

import pandas as pd
import numpy as np
import functools

class DailyPatternEstimator():
    """This class estimates a typical 'day' from the dataset, and stores in the
    property typical_day. Subclasses may choose to store typical week or
    similar instead. <typical_day> is a data frame indexed on the number of
    time steps in the typical period, i.e. 0-23 for days and 0-167 for
    weeks. The entire training set is normally used to estimate the typical
    day/week."""

    def __init__(self, dataset):
        """Initialize with the dataset for which the 'typical day' should be
        estimated."""
        self._dataset = dataset.copy()
        self._calc_typical_day()
        
    def _calc_typical_day(self):
        self._dataset['Hour of day'] = \
          [i.hour for i in self._dataset.index]
        self._typical_day = self._dataset.groupby(['Hour of day']).mean()

    @property
    def typical_day(self):
        return self._typical_day


class WeeklyPatternEstimator(DailyPatternEstimator):
    def _calc_typical_day(self):
        self._dataset['Hour of week'] = \
          [i.dayofweek * 24 + i.hour for i in self._dataset.index]
        self._typical_day = self._dataset.groupby(['Hour of week']).mean()


class DailyPatternEliminator():
    """This class subtracts the typical day/week from each day in the dataset
    given as input. This dataset is typically the data fed to the prediction
    model, i.e. 4-12 weeks + 24 hours of NaNs at the end."""

    def __init__(self, dataset, typical_day):
        """Initialize with the dataset for which the 'typical day' should be
        eliminated."""
        self._dataset = dataset.copy()
        self._columns = dataset.columns
        self._typical_cols = ['%s typical day' % col for col in self._columns]
        self._deviation_cols = ['%s minus typical day' % col for col in self._columns]
        self._subtract_typical_day(typical_day)

    def _subtract_byref(self, values, lookup, from_columns, mean_columns,
                       to_columns):
        """Subtract a value from 'from_columns' in two steps: first use 'lookup'
        to index into 'values', storing the correct value at each row in
        'mean_columns'. Then store the subtraction 'from_columns -
        mean_columns' into 'to_columns'."""
        for (from_col, mean_col) in zip(from_columns, mean_columns):
            self._dataset[mean_col] = \
              [values[from_col][int(i)] for i in self._dataset[lookup]]    
        for (from_col, mean_col, to_col) in \
          zip(from_columns, mean_columns, to_columns):
            self._dataset[to_col] = \
              self._dataset[from_col] - self._dataset[mean_col]

    def _subtract_typical_day(self, typical_day):
        """Calculate a "typical day" as the mean for each hour of day
        throughout the dataset. Then subtract this, leaving the "abnormality of
        the current day"."""
        self._dataset['Hour of day'] = \
          [i.hour for i in self._dataset.index]
        self._subtract_byref(typical_day, 'Hour of day', 
                             self._columns, self._typical_cols,
                             self._deviation_cols)

    def as_deviation_from_typical_day(self):
        """Return the data after removing the daily mean and the typical
        day."""
        deviate = self._dataset[self._deviation_cols]
        return deviate.rename(
            columns=dict(zip(self._deviation_cols, self._columns)))

    def add_typical_day(self, dataset):
        columns = self._columns.join(dataset.columns, how="inner")
        typical_map = dict(zip(self._columns, self._typical_cols))
        overlap = self._dataset.ix[dataset.index]
        return pd.DataFrame(
            dict([(col, dataset[col] + overlap[typical_map[col]]) \
                  for col in columns]))

    def add_typical_day_to_series(self, timeseries, col_name="Load"):
        """Given a time series, add the typical day from the column named
        'col_name' in the dataset used in the constructor."""
        dataset = pd.DataFrame({col_name: timeseries})
        reverted = self.add_typical_day(dataset)
        return reverted[col_name]

                   
class WeeklyPatternEliminator(DailyPatternEliminator):
    def _subtract_typical_day(self, typical_day):
        """Calculate a "typical day" as the mean for each hour of day
        throughout the dataset. Then subtract this, leaving the "abnormality of
        the current day"."""
        self._dataset['Hour of week'] = \
          [i.dayofweek * 24 + i.hour for i in self._dataset.index]
        self._subtract_byref(typical_day, 'Hour of week', 
                             self._columns, self._typical_cols,
                             self._deviation_cols)


class Pipeliner():
    """Wrapper that facilitates preprocessing and postprocessing where state
    is saved in the preprocessing step and subsequently used in the
    postprocessing.

    """
    def __init__(self, dataset):
        pass
    
    # Go via an extra function in superclass to make inheritance work,
    # since functools.partial explicitly calls this class.
    def _preprocess_super(self, data_in, genome, loci, prediction_steps):
        return self.preprocess(data_in, genome, loci, prediction_steps)

    def preprocess(self, data_in, genome, loci, prediction_steps):
        raise NotImplementedError()

    def _postprocess_super(self, data_out, genome, loci):
        return self.postprocess(data_out, genome, loci)

    def postprocess(self, data_out, genome, loci):
        raise NotImplementedError()
        
    def get_callbacks(self):
        return (functools.partial(Pipeliner._preprocess_super, self),
                functools.partial(Pipeliner._postprocess_super, self))


class PatternPipeliner(Pipeliner):
    """Wrapper that allows for typical pattern elimination in the preprocessing
    stage of a processing pipeline, and reversal in the postprocessing
    step. The challenge is that the mean, which is subtracted from the input
    signal in the preprocessing step, must be added to the output in the
    postprocessing step. We thus need to save some state in the preprocessing
    which can be accessed in the postprocessing."""
    def __init__(self, dataset):
        Pipeliner.__init__(self, dataset)
        self._dataset = dataset
        self._eliminator = None
        self._data_pre, _ = self._dataset.get_pre_post_processors()
        if not self._data_pre:
            self._estimator = self._make_estimator(self._dataset.train_data)

    def preprocess(self, data_in, genome, loci, prediction_steps):
        if self._data_pre:
            pattern_data = self._dataset.train_data.copy()
            for preproc in self._data_pre:
                pattern_data = preproc(pattern_data, genome, loci, prediction_steps)
            self._estimator = self._make_estimator(pattern_data)

        self._eliminator = self._make_eliminator(data_in)
        return self._eliminator.as_deviation_from_typical_day()
    
    def postprocess(self, data_out, genome, loci):
        return self._eliminator.add_typical_day_to_series(data_out)

    
class DailyPatternPipeliner(PatternPipeliner):
    def _make_estimator(self, pattern_data):
        return DailyPatternEstimator(pattern_data)
    
    def _make_eliminator(self, data_in):
        return DailyPatternEliminator(data_in, self._estimator.typical_day)

    
class WeeklyPatternPipeliner(PatternPipeliner):
    def _make_estimator(self, pattern_data):
        return WeeklyPatternEstimator(pattern_data)
    
    def _make_eliminator(self, data_in):
        return WeeklyPatternEliminator(data_in, self._estimator.typical_day)


class MixedPatternPipeliner(Pipeliner): 
    """Electricity consumption has both weekly and daily seasonality, but
    temperature has only daily seasonality. This pipeliner uses a daily
    pipeliner for temperature and a weekly pipeliner for load."""
    def __init__(self, dataset):
        Pipeliner.__init__(self, dataset)
        self._temp_pattern = DailyPatternPipeliner(dataset)
        self._load_pattern = WeeklyPatternPipeliner(dataset)

    def preprocess(self, data_in, genome, loci, prediction_steps):
        temp_elim = self._temp_pattern.eliminate(data_in, genome, loci, 
                                                 prediction_steps)
        load_elim = self._load_pattern.eliminate(data_in, genome, loci, 
                                                 prediction_steps)
        return pd.DataFrame({"Temperature": temp_elim["Temperature"],
                             "Load": load_elim["Load"]})
    
    def postprocess(self, data_out, genome, loci):
        return self._load_pattern._eliminator.add_typical_day_to_series(data_out)
        

class DiffPipeliner(Pipeliner):
    def preprocess(self, data_in, genome, loci, prediction_steps):
        self._init_val = data_in['Load'].ix[-1-prediction_steps]
        return data_in.diff()[1:]
    
    def postprocess(self, data_out, genome, loci):
        data_out.ix[0] += self._init_val
        for i in range(1, len(data_out)):
            data_out.ix[i] += data_out.ix[i-1]
        return data_out
        
class StandardizerPipeliner(Pipeliner):
    def preprocess(self, data_in, genome, loci, prediction_steps):
        self._std = data_in.std()
        self._mean = data_in.mean()
        return (data_in - self._mean)/self._std
    
    def postprocess(self, data_out, genome, loci):
        return data_out * self._std['Load'] + self._mean['Load']


def make_daily_pattern_eliminator(dataset):
    return DailyPatternPipeliner(dataset).get_callbacks()

def make_weekly_pattern_eliminator(dataset):
    return WeeklyPatternPipeliner(dataset).get_callbacks()

def make_mixed_pattern_eliminator(dataset):
    return MixedPatternPipeliner(dataset).get_callbacks()

def make_diff_pipeliner(dataset):
    return DiffPipeliner(dataset).get_callbacks()

def make_standardizer_pipeliner(dataset):
    return StandardizerPipeliner(dataset).get_callbacks()
