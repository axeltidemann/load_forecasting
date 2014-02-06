import os
import itertools
import datetime

import tables as h5
import pandas as pd
import numpy as np

from create_full_temp_data import data as _read_temperatures
from sg.globals import SG_DATA_PATH

DATA_DIR = os.path.join(SG_DATA_PATH, "sintef/")
DATA_WITH_DUPES_PATH = os.path.join(DATA_DIR, "tempfeeder_with_duplicates.blosc-9.h5")
DATA_WITHOUT_DUPES_PATH = os.path.join(DATA_DIR, "tempfeeder_without_duplicates.blosc-9.h5")


class UserLoads(object):
    """This class reads an HDF5 file with per-user load values, and presents
    them in a dict-style to the user. Late loading, so values are not read
    until they are requested the first time.

    The set of user IDs can be accessed via the read-only property user_ids.
    """

    def __init__(self, path):
        """Open the file given in path, and read user IDs."""
        self._path = path
        self._store = pd.HDFStore(path, "r")
        self._user_ids = self._load_user_ids()
        self._loads = dict()

    def __del__(self):
        self._store.close()

    def _load_user_ids(self):
        # Could also user self._store.keys(), but this would return strings of
        # the form "id_nnnnn".
        return set(self._store["user_ids"])

    def close(self): 
        """Close the underlying HDF5 file, so it can be accessed by other
        functions. Any subsequent attempts to read from the file will fail."""
        self._store.close()
        
    def __len__(self):
        return len(self.user_ids)

    def read(self, user_id):
        """Read load data for user_id from HDF5 storage. Any existing data will
        be overwritten by data read from file. Use the subscript operator to
        access the loads rather than calling this function. Only call this
        function when you want to reset a modified load value to the value on
        file."""
        if not user_id in self.user_ids:
            raise KeyError("Invalid user ID. Check the read-only property " \
                           "user_ids for a list of valid IDs.")
        user_load = self._store["id_" + str(user_id)]
        self._loads[user_id] = user_load
        return user_load

    def read_all(self):
        """Read all load values from file at once. Normally late reading
        through the subscript operator is preferrable, but this function can be
        used for timing purposes or when manipulating the data
        interactively."""
        for user_id in self.user_ids:
            self.read(user_id)

    def __getitem__(self, user_id):
        if not user_id in self._loads:
            return self.read(user_id)
        return self._loads[user_id]

    def __setitem__(self, user_id, user_load):
        """Set new load values for user_id. The ID must correspond to an ID
        existing in the HDF5 file. The new load will not be written back to the
        HDF5 file, and will be overwritten by a subsequent call to read or
        read_all."""
        if not user_id in self.user_ids:
            raise KeyError("Invalid user ID. Check the read-only property " \
                           "user_ids for a list of valid IDs.")
        self._loads[user_id] = user_load

    def pop(self, user_id):
        """Remove specified loads and return them. Similar to dict.pop, but
        will not remove user_id from self.user_ids."""
        if not user_id in self.user_ids:
            raise KeyError("Invalid user ID. Check the read-only property " \
                           "user_ids for a list of valid IDs.")
        if not user_id in self.loads:
            self.read(user_id)
        return self._loads.pop(user_id)

    @property
    def user_ids(self):
        """The set of user IDs for which load data are stored in the HDF5
        file. This is a read-only property."""
        return list(self._user_ids)
    
    @property
    def loads(self):
        """A dict containing all the loads that have been read so far. This is
        a read-only property, but changing its contents will affect the
        internal representation in this class as well."""
        return self._loads

    @property
    def path(self):
        """Path to the HDF5 file as given to the __init__ function."""
        return self._path

    def __contains__(self, user_id):
        return user_id in self.user_id

    def __str__(self):
        return self.user_ids.__str__() + self.loads.__str__()


def experiment_periods():
    """Return the two preselected periods for which experiments will be carried
    out. These correspond to the two longest periods for which we have
    consecutive load values for an acceptable number of meters on the feeder
    with temperature recordings. Note that the temperature reading start
    later."""
    # Temperature readings start March 22 2004, loads start Feb 01 2004.
    # period1_start = pd.Period("2004-02-01 00:00", "H")
    # period1_end = pd.Period("2005-07-01 00:00", "H")
    # period2_start = pd.Period("2005-10-01 00:00", "H")
    # period2_end = pd.Period("2006-10-01 00:00", "H")
    period1_start = datetime.datetime.strptime("2004-02-01 00:00", "%Y-%m-%d %H:%M")
    period1_end = datetime.datetime.strptime("2005-07-01 00:00", "%Y-%m-%d %H:%M")
    period2_start = datetime.datetime.strptime("2005-10-01 00:00", "%Y-%m-%d %H:%M")
    period2_end = datetime.datetime.strptime("2006-10-01 00:00", "%Y-%m-%d %H:%M")
    return ((period1_start, period1_end), (period2_start, period2_end))
    
def manually_screened_ids():
    """IDs of meters that passed the automated tests for resolution, missing
    data, etc, but whose data do still not meet our criteria. These include:
       * Meters with 1kWh/h resolution that somehow passed the resolution test,
         e.g. by installing new meters at some point during the experiment
         period
       * Meters with missing data that have been processed and set to 0 by the
         utility company (as indicated by the status tag)
    The list does NOT include meters that have "weird" timeseries.
    """
    low_resolution_meters = (35466301, 82218621, 15720078, 80690326, 65630886, 
                             13824685, 87785213, 12645122, 89454871)
    missing_data = (73122950, 39281849, 99260911, 92959456, 79042288, 97564405,
                    8751522)
    return set(itertools.chain(low_resolution_meters, missing_data))

class UserLoads_Experiment(UserLoads):
    def _load_user_ids(self):
        return set(self._store['user_ids_cln_pred_exp'])
    
# The actual data sets
_tempfeeder_dup = None
_tempfeeder_nodup = None
_tempfeeder_exp = None
def tempfeeder_dup():
    """Return userloads from the feeder that has temperature readings. Returns
    a time series with duplicates where there were duplicates in the original
    files."""
    global _tempfeeder_dup
    if _tempfeeder_dup is None:
        _tempfeeder_dup = UserLoads(DATA_WITH_DUPES_PATH)
    return _tempfeeder_dup

def tempfeeder_nodup():
    """Return userloads from the feeder that has temperature readings. Returns
    a time series where duplicates in the original files have been eliminated,
    keeping only the first value that was encountered during processing."""
    global _tempfeeder_nodup
    if _tempfeeder_nodup is None:
        _tempfeeder_nodup = UserLoads(DATA_WITHOUT_DUPES_PATH)
    return _tempfeeder_nodup

def tempfeeder_exp():
    """Return userloads from the feeder that has temperature readings. Returns
    a time series where duplicates in the original files have been eliminated,
    keeping only the first value that was encountered during processing.

    The list of user ids contains only those users/meters that have been
    selected for further processing as described in the EnergyCon paper.
    """
    global _tempfeeder_exp
    if _tempfeeder_exp is None:
        _tempfeeder_exp = UserLoads_Experiment(DATA_WITHOUT_DUPES_PATH)
    return _tempfeeder_exp

def tempfeeder_exp_nonzerotest_users():
    """Returns a list of user IDs that does not have a zero reading in the test period,
    which is from 2005-10-01 00:00 -> 2006-10-01 00:00. This selection is made so a
    MAPE (although somewhat useless nonetheless, since the values are so small)
    can be calculated."""

    return [ user for user in tempfeeder_exp().user_ids if all(tempfeeder_exp()[user]['Load']['2005-10-01 00:00':]) ]

def total_load(userloads, user_ids, period):
    """Calculate the total load for the given users period."""
    series = [userloads[user][period[0]:period[1]] for user in user_ids]
    total = series[0].copy()
    for single_series in series[1:]:
        total += single_series
    return total

def total_load_in_experiment_periods(userloads, user_ids):
    """Return a list of time series containing the total load for the given
    users in the experiment periods."""
    periods = experiment_periods()
    return [total_load(userloads, user_ids, period) for period in periods]

def mean_experiment_load_for_user_subset(num_users, seed=None):
    """Return a list of time series containing the total load over the
    experiment periods for the user_ids selected to be part of the
    clean+predict experiment."""
    loads = tempfeeder_exp()
    if seed is None:
        seed = np.random.randint(1, 2**16)
    user_ids = np.random.RandomState(seed).permutation(loads.user_ids)[:num_users]
    return [l / len(user_ids) for l in total_load_in_experiment_periods(loads, user_ids)]

def total_experiment_load():
    """Return a list of time series containing the total load over the
    experiment periods for the user_ids selected to be part of the
    clean+predict experiment."""
    loads = tempfeeder_exp()
    return total_load_in_experiment_periods(loads, loads.user_ids)

def add_temperatures(loads, period):
    """Given data frame with loads and status code, drop the status code and
    add temperatures in the given period. Temperatures come primarily from GS2
    files, using eklima readings to fill in missing values ."""
    # Temperature and load readings may start and end at different times.
    temps = _read_temperatures()
    (l, t) = loads.align(temps, join="inner", axis=0)
    frame = pd.concat((t, l['Load']), axis=1)
    frame = frame.rename(columns={0:"Temperature", 1:"Load"})
    return frame[period[0]:period[1]]


if __name__ == "__main__":
    from unittest import main
    main(module="test_" + __file__[:-3])
