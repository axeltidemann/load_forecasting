"""Convert SINTEF load data in HDF5 files from scikits.timeseries to
pandas.DataFrames."""

import datetime

import tables as h5
import pandas as pd
import scikits.timeseries.lib.tstables
import scikits.timeseries as ts

from preprocess_gs2 import PandasH5Storer

class Converter(PandasH5Storer):
    def __init__(self, path_ts_in, path_pd_out):
        PandasH5Storer.__init__(self, path_pd_out)
        self._h5file_ts = h5.openFile(path_ts_in, "r")

    def __del__(self):
        PandasH5Storer.__del__(self)
        self._h5file_ts.close()
        
    def _load_ts_user(self, user_id):
        return self._h5file_ts.getNode("/loads/id_" + str(user_id)).read()

    def _convert_dates(self, series_ts):
        return [date.datetime for date in series_ts.dates]
    
    def _make_pd_series_from_scikits_series(self, series_ts):
        data = {'Load' : series_ts[:,0],
                'Status Code' : series_ts[:,1]}
        dates = self._convert_dates(series_ts)
        return pd.DataFrame(data, index=dates)

    def _convert_user_id_lists(self):
        """The list of experiment users was stored in the original file. This
        must be carried over as a Series in the Pandas file."""
        user_ids = self._h5file_ts.root.loads.cln_pred_exp_ids.read()
        self.store_list('user_ids_cln_pred_exp', user_ids)
        user_ids = self._h5file_ts.root.loads.user_ids.read()
        self.store_list('user_ids', user_ids)
    
    def _convert_users(self):
        user_ids = self._h5file_ts.root.loads.user_ids.read()
        for user_id in user_ids:
            series_ts = self._load_ts_user(user_id)
            series_pd = self._make_pd_series_from_scikits_series(series_ts)
            self.store_pd_user(user_id, series_pd)
        
    def convert(self):
        self._convert_user_id_lists()
        self._convert_users()

def _get_targets_from_base_paths(paths):
    from os.path import split, join
    targets = []
    for path in paths:
        dir, base = split(path)
        targets.append(join(dir, "pandas_" + base))
    return targets

def _get_sintef_paths():
    import userloads as ul
    bases = (ul.DATA_WITH_DUPES_PATH, ul.DATA_WITHOUT_DUPES_PATH)
    targets = _get_targets_from_base_paths(bases)
    return zip(bases, targets)
    
def convert_sintef_files(interactive=False):
    paths = _get_sintef_paths()
    print "This script will convert scikits.timeseries to pandas in the " \
       "following files:"
    for (path_ts, path_pd) in paths:
       print "\n\t%s\nto\n\t%s" % (path_ts, path_pd)
    while True:
        response = raw_input("\nContinue (y/n)? ")
        if response == 'y':
            break
        elif response == 'n':
            return
    for (path_ts, path_pd) in paths:
        print "Converting %s to %s." % (path_ts, path_pd)
        Converter(path_ts, path_pd).convert()
    print "Done."

if __name__ == "__main__":
    convert_sintef_files(interactive=True)

    
