"""6.5% of the Porsgrunn temperature readings from the SINTEF files
are missing. They are concatenated with eklima.met.no data from
Gvarv-Nes, and interpolated. Furthermore, two periods have obvious
erroneous data readings, look up the periods 2004-11-11 14:00 ->
2004-11-22 23:00 and 2005-02-08 08:00 -> 2005-02-27 23:00. These two
periods are replaced with data from eklima. The final stage is
interpolation, so the dataset has hourly readings (eklima only reads
data 4 times a day). Note: the following command must be issued
beforehand, since it stores all the timeseries in a file that is loaded.

./gs2-grep.sh -l Grader | python plot_temp.py
"""

import os

import numpy.ma as ma
import numpy as np
import pandas as pd

import sg.data.eklima.parse_eklima_xml as xml
import sg.utils
from sg.globals import SG_DATA_PATH

_TEMP_DATA = os.path.join(SG_DATA_PATH, "eklima", "Telemark", 
                          "Gvarv-Nes2004-2006.xml")
_PATH_TO_HERE = os.path.dirname(os.path.abspath(__file__))

def data():
    temp = pd.load(os.path.join(_PATH_TO_HERE, 'temp_data.pickle'))
    temp = temp.sort_index().asfreq("H")
    # Extended periods with failed readings, replace with Gvarv
    temp['2004-11-11 14:00':'2004-11-21 23:00'] = np.nan
    temp['2005-02-08 08:00':'2005-02-27 23:00'] = np.nan
    # Shorter periods with failed readings, that we may leave to the cleansing
    # to take care of?
    # temp['2005-09-07 08:00':'2005-09-08 04:00'] = np.nan
    # temp['2006-02-28 05:00':'2006-02-28 04:00'] = np.nan
    # temp['2006-06-17 11:00':'2006-06-18 08:00'] = np.nan
    # temp['2006-12-19 06:00':'2006-12-21 03:00'] = np.nan
    gvarv = xml.parse(_TEMP_DATA)[temp.index[0]:].asfreq("H")
    gvarv_aligned = temp.align(gvarv, join="left")[1]
    filled = np.where(np.isnan(temp), gvarv_aligned, temp)
    filled = filled.interpolate()
    filled.name = "Temperature"
    # Interpolata away a couple of outliers and zero-recordings, or leave to
    # cleansing?
    # filled['2004-11-29 08:00'] = np.nan
    # filled['2005-11-30 00:00':'2005-11-30 02:00'] = np.nan
    # filled['2006-10-27 09:00'] = np.nan
    # filled = filled.interpolate()
    return filled
    
if __name__ == "__main__":
    data = data()
    sg.utils.plot_time_series([data], ['b.'], ['Porsgrunn + Gvarv temperature'])
