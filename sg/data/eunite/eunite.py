"""Import EUNITE dataset. Concatenates 1997, 1998 and January 1999 data. The
competition used Jan 1999 as test set."""

import os
import sys
import sqlite3
import datetime
import numpy as np

import pandas as pd

from sg.globals import SG_DATA_PATH
import sg.data

PATH_TO_EUNITE_DB = os.path.join(SG_DATA_PATH, "eunite", "eunite.db")

class Dataset(sg.data.Dataset):
    def __init__(self, period, step_length=None):
        """Loads the EUNITE time series and sets up for extraction of random
        slices of length 'period', 'step_length' apart. See class Dataset for
        more info."""
        sg.data.Dataset.__init__(self, load(), period, step_length)

def load(dbpath=PATH_TO_EUNITE_DB):
    """Read the load data from the given database. Return a pandas.DataFrame
    containing the data."""
    with sqlite3.connect(dbpath, detect_types=sqlite3.PARSE_DECLTYPES|
                             sqlite3.PARSE_COLNAMES) as conn:
        crs = conn.cursor()
        sel_stmt = "SELECT Timestamp as 'stamp [timestamp]', "\
            "Deg_C as 'temp [float]', "\
            "MWh as 'load [float]' "\
            "FROM "
        crs.execute(sel_stmt + "training" + \
                        " UNION " + \
                        sel_stmt + "testing" + \
                        " ORDER BY Timestamp ASC")
        stamps, temps, loads = zip(*crs.fetchall())
    return pd.DataFrame({'Temperature' : np.array(temps, dtype=float),
                         'Load' : np.array(loads, dtype=float)},
                         index=stamps)

if __name__ == '__main__':
    from unittest import main
    main(module='test_'+__file__[:-3])
