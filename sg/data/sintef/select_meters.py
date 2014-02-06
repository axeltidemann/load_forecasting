"""Code to inspect and verify the AMS data: look for gaps, duplicates etc."""

from collections import defaultdict
import cPickle as pickle
import os

import tables as h5
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import sys

import userloads as ul

def find_highres_meters(userloads):
    highres = []
    for user in userloads.user_ids:
        for load in userloads[user]:
            if load[0] != int(load[0]):
                highres.append(user)
                break
    return highres

def store_user_ids(user_ids, h5path, title, description=""):
    h5file = h5.openFile(h5path, "a")
    group = h5file.root.loads
    h5file.createArray(group, title, user_ids, description)
    h5file.close()

def count_unique_load_values(loads):
    values = set();
    for i in range(len(loads.data)):
        values.add(loads.data[i][0])
        if len(values) > 100:
            break
    return (len(values), i)

def unique_load_values_per_user(userloads, user_ids):
    """Iterate over user_ids, count the number of unique loads for each of
    these. Return a tuple (user_id, (num_unique, loads_seen))."""
    return [(user, count_unique_load_values(userloads[user])) \
            for user in user_ids]

def count_missing_dates(loads):
    ts = loads
    if not ts.is_full():
        ts = ts.fill_missing_dates()
    return len(np.where(ts.mask[:,0])[0])

def missing_dates_per_user(userloads, user_ids):
    """Iterate over user_ids, count the number of missing dates for each of
    these. Return a tuple (user_id, num_missing)."""
    return [(user, count_missing_dates(userloads[user])) for user in user_ids]

def fill_all_missing_dates_in_place(userloads, user_ids=None):
    """Fill (mask) missing dates of each userload."""
    if user_ids is None:
        user_ids = userloads.keys()
    for user in user_ids:
        if not userloads[user].is_full():
            userloads[user] = userloads[user].fill_missing_dates()

def first_start_date(userloads, user_ids):
    return min([userloads[user].start_date for user in user_ids])

def last_end_date(userloads, user_ids):
    return max([userloads[user].end_date for user in user_ids])

def zero_series(userloads, user_ids):
    """Return an zero-filled time-series covering all dates in userloads."""
    dates = ts.date_array(start_date=first_start_date(userloads, user_ids),
                          end_date=last_end_date(userloads, user_ids),
                          freq=userloads[user_ids[0]].freq)
    data = np.zeros(len(dates))
    return ts.time_series(data, dates=dates)

def _date_is_missing(loads, date):
    """Return True if the time series loads is missing data for the given
    date. Assumes that missing dates have been filled in advance."""
    if date < loads.start_date:
        return True
    if date > loads.end_date:
        return True
    return loads[date].mask[0]

def misses_per_date(userloads, user_ids):
    """Return a time series indicating the number of users with missing data
    for each date over the entire period for which we have data."""
    num_missing = zero_series(userloads, user_ids)
    fill_all_missing_dates_in_place(userloads, user_ids)
    for date in num_missing.dates:
        num_missing_now = 0
        for user in user_ids:
            if _date_is_missing(userloads[user], date):
                num_missing_now += 1
        num_missing[date] = num_missing_now
    return num_missing

def users_with_data_in_experiment_periods(userloads, user_ids):
    """Return a list of the users with no missing data in two specific,
    hard-coded periods."""
    return users_with_data_in_periods(userloads, user_ids,
        ul.experiment_periods())

def users_with_data_in_periods(userloads, user_ids, periods):
    """Return a list of the users with no missing data in the given periods"""
    no_misses = set(user_ids)
    for (start, end) in periods:
        nm_copy = list(no_misses)
        for user in nm_copy:
            series = userloads[user][start:end]
            if len(series) == 0 or series.start_date > start or \
                series.end_date < (end - 1) or \
                series.has_missing_dates() or np.any(series.mask[:,0]):
                no_misses.remove(user)
    return list(no_misses)

def compare_user_series(user_series_old, user_series_new):
    """Compare two user series dicts, checking only the contents of the first
    column."""
    for user_id in user_series_old:
        diff = user_series_old[user_id] - user_series_new[user_id]
        if not np.all(diff[:,0] == 0):
            print "Series %s differ" % user_id
            
def check_series_duplicates(series):
    """Check a single time series for duplicates with different values on the
    same date. Return a tuple where the first element is the number of
    duplicates, and the second is the number of differing duplicates."""
    dates = series.dates
    dupes = 0
    diffdupes = 0
    for idx in range(len(dates) - 1):
        if dates[idx] == dates[idx + 1]:
            dupes += 1
            if series.data[idx][0] != series.data[idx + 1][0]:
                diffdupes += 1
    return (dupes, diffdupes)

def check_remove_userload_dupes(userloads, user_ids):
    """Check the given user_ids in userloads for duplicates with differing
    values. Return a list of users from user_ids without differing
    duplicates."""
    ret_ids = set(user_ids)
    for user in user_ids:
        (dupes, diffs) = check_series_duplicates(userloads[user])
        if diffs > 0:
            print "User", user, "has", dupes, "duplicates and", diffs, \
                "differing values."
            ret_ids.remove(user)
    return list(ret_ids)

def remove_manually_screened_ids(user_ids):
    """Returns a copy of user_ids with the user IDs identified by manual
    screening removed."""
    user_set = set(user_ids)
    for user in ul.manually_screened_ids():
        if user in user_set:
            user_set.remove(user)
    return list(user_set)

def plot_to_pdf(userloads, user_ids, path, periods):
    pp = PdfPages(path)
    for ((start, end), index) in zip(periods, range(1, len(periods)+1)):
        for user in user_ids:
            plt.clf()
            userloads[user]['Load'][start:end].plot()
            plt.title("Period %d, user %d" % (index, user))
            #plt.ylim(0, 15)
            pp.savefig()
    pp.close()

if __name__ == "__main__":
    tempfeeder_dup = ul.tempfeeder_dup()
    tempfeeder_nodup = ul.tempfeeder_nodup()
    
    print "Reading with dupes..."
    sys.stdout.flush()
    tempfeeder_dup.read_all()
    print "Reading without dupes..."
    sys.stdout.flush()
    tempfeeder_nodup.read_all()
    print "Done reading."
    tempfeeder_dup.close()
    tempfeeder_nodup.close()
    print "Checking for highres meters..."
    sys.stdout.flush()
    highres = find_highres_meters(tempfeeder_nodup)
    print "Narrowing to users with data in experiment periods..."
    sys.stdout.flush()
    highres_few_missing = \
        users_with_data_in_experiment_periods(tempfeeder_nodup, highres)
    print "Checking for duplicates."
    sys.stdout.flush()
    keepers = check_remove_userload_dupes(tempfeeder_nodup, 
                                          highres_few_missing)
    print "Removing manually screened IDs"
    keepers = remove_manually_screened_ids(keepers)
    print len(keepers), "candidates for experiment:", keepers 
    path = "keepers.pickle"
    print "Pickling the keepers to %s." % path 
    with open(path, "w") as f:
        pickle.dump(keepers, f)
    path = os.path.join(ul.DATA_DIR, "Experiment timeseries.pdf")
    print "Plotting all selected timeseries to %s..." % path
    plot_to_pdf(tempfeeder_nodup, keepers, path, ul.experiment_periods())
    print "Storing experiment user IDs to HDF5 file."
    for path in (tempfeeder_dup.path, tempfeeder_nodup.path):
        store_user_ids(keepers, path, "cln_pred_exp_ids",
            "IDs of meters selected for clean+predict experiment early 2012.")
