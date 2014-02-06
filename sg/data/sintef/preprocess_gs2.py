#!/usr/bin/python

import os
import collections
import datetime
import sys
import sqlite3
import cPickle as pickle

import numpy as np
import pandas as pd
import tables as h5

from sg.utils.timer import SimpleTimer
import parse_gs2 as gs2


_PATH_TO_HERE = os.path.dirname(os.path.abspath(__file__))
_PATH_TO_GS2_TXT = os.path.join(_PATH_TO_HERE, "gs2.txt")

class PandasH5Storer(object):
    def __init__(self, h5path, complevel=9, 
                 complib='blosc', fletcher32=False):
        self._h5store = pd.HDFStore(h5path, mode='w', complevel=complevel, 
                                    complib=complib, fletcher32=fletcher32)

    def __del__(self):
        self._h5store.close()
    
    def store_pd_user(self, user_id, data):
        self._h5store["id_" + str(user_id)] = data

    def store_list(self, key, lst):
        pd_list = pd.Series(data=lst)
        self._h5store[key] = pd_list
        

def parse_all_print_info(pathfile=_PATH_TO_GS2_TXT):
    timer = SimpleTimer()
    for (path, contents) in parse_all_generator(pathfile):
        print "Parsed ", path, "found %d sections." % len(contents)

def load_all_except_values(pathfile=_PATH_TO_GS2_TXT):
    timer = SimpleTimer()
    all_files = []
    for (path, contents) in parse_all_generator(pathfile):
        print "Parsed %s." % path
        [section[1].pop("Value", None) for section in contents]
        all_files.append((path, contents))
    return all_files

def _section_union(section, union):
    for (key, value) in section.iteritems():
        if key == "Value":
            continue
        reference = union.get(key)
        if reference is None:
            union[key] = set(value)
        else:
            reference.add(value[0])

def _all_unique_from_single_file(file_sections, all_sections):
    for (name, section) in file_sections:
        _section_union(section, all_sections[name])

def all_unique_except_values(parsed_files):
    """Given a list of parsed files as (name, sections) tuples, return a dict
    of dicts. The outer dict uses section headings as keys and the inner dict
    as values. The inner dict is essentially a union of the sections in the GS2
    files: the keys are GS2 keys and the values are sets of unique values from
    the GS2 files. The "Value" element is dropped.

    The output from this function can be used to e.g. identify overall start
    and end times, count the number of users, etc.""" 
    timer = SimpleTimer()
    all_sections = collections.defaultdict(dict)
    for (path, sections) in parsed_files:
        _all_unique_from_single_file(sections, all_sections)
    return all_sections

def _print_basic_dataset_stats(all_sections, timeseries):
    print "Sections (should be 'Start-message', 'Time-series', 'End-message'):"
    print all_sections.keys()
    print "Time-series keys:", timeseries.keys()
    print ""
    print "Plants (hopefully not many):", timeseries['Plant']
    print "Step (should be one hour):", timeseries['Step']
    print "Units (should be kWh and perhaps Celsius):", timeseries['Unit']
    inst_ids = timeseries.get('Installation')
    print "Number of customers ID'ed as 'Installation':", \
        ("None" if inst_ids is None else len(inst_ids))
    ref_ids = timeseries.get('Reference')
    print "Number of customers ID'ed as 'Reference':", \
        ("None" if ref_ids is None else len(ref_ids))
    print "\nSummary:"
    for key in timeseries:
        print "Number of unique %s elements: %d" % (key, len(timeseries[key]))

def _fix_24hr_time_string(time_string):
    """Some of the timestamps use hour 24 rather than hour 0 the next day. This
    is not supported by Python's strptime. Fix it by replacing 24 with 23,
    converting to datetime, and then adding an hour."""
    if '.24:' in time_string:
        prev_hr = time_string.replace('.24:', '.23:')
        as_datetime = datetime.datetime.strptime(prev_hr, "%Y-%m-%d.%H:%M:%S")
        as_datetime += datetime.timedelta(hours=1)
        return as_datetime.strftime("%Y-%m-%d.%H:%M:%S")
    else:
        return time_string

def _convert_to_datetime(time_string):
    """Given a GS2 timestamp as string, return the corresponding datetime."""
    return datetime.datetime.strptime(_fix_24hr_time_string(time_string),
                                      "%Y-%m-%d.%H:%M:%S")

def _get_start_and_end_times(timeseries):
    start_times = [_convert_to_datetime(t) for t in timeseries["Start"]]
    end_times = [_convert_to_datetime(t) for t in timeseries["Stop"]]
    return (min(start_times), max(end_times))

def list_all_status_codes(parsed_files):
    """Given a list of parsed files as (name, sections) tuples, return two
    dicts containing all the different texts found after and in between the
    double slashes in the 'Value' field of the 'Time-series' section of the
    input files."""
    timer = SimpleTimer()
    betweens = collections.defaultdict(int)
    afters = collections.defaultdict(int)
    for (path, sections) in parsed_files:
        for section_idx in range(len(sections)):
            (name, section) = sections[section_idx]
            if name != "Time-series":
                continue
            unit = section.get("Unit")
            if unit is None or (unit != ["kWh"] and unit != ["Grader Celsius"]):
                print "Skipping time-series with unit", unit, "in file", path
                continue
            values = section.get("Value")
            if values is not None:
                for value in values:
                    if '/' in value:
                        try:
                            (val, between, after) = value.split('/')
                        except ValueError as ve:
                            raise ValueError(
                                "Error parsing values in section number %d of " \
                                "file %s. Did not find the exptected two "
                                "slashes? Error message was: '%s'" % \
                                (section_idx, path, ve))
                        betweens[between] += 1
                        afters[after] += 1
                    else:
                        float(value) # Should be a single measuremt if no slashes
                        betweens['No value'] += 1
                        afters['No value'] += 1
    return (betweens, afters)

class GS2ToDataFrame(object):
    def __init__(self):
        self._identifiers = ['Installation', 'Reference']

    def _reset(self):
        self._user_series = dict()
    
    def parse(self, path_to_gs2_list):
        self._reset()
        print "Generating per-user timeseries, using identifiers:", self._identifiers
        fileno = 1
        for path, sections in gs2.parse_all_generator(path_to_gs2_list):
            print "%d: converting %s to time series." % (fileno, path)
            fileno += 1
            for section in sections:
                self._add_section_to_user_series(section)
        return self._split_tuple_dict()

    def _add_section_to_user_series(self, section):
        """Add the values in section (if they exist) to the corresponding user in
        self._user_series."""
        name, contents = section
        new_series = self._convert_section_to_timeseries(name, contents)
        if new_series is None:
            return
        for identifier in self._identifiers:
            user = contents.get(identifier)
            if user is not None:
                break
        if user is None:
            print "Skipping section %s, user is None (wrong identifier?)" % name
            return
        user = user[0].strip()
        full_series = self._user_series.get(user)
        if full_series is None:
            self._user_series[user] = (new_series, new_series)
        else:
            self._user_series[user] = self._merge_timeseries(new_series, full_series)

    def _convert_section_to_timeseries(self, name, contents):
        """Look for values in the section. If they exist, return as timeseries."""
        file_tz = "UTC"
        # file_tz = "Europe/Oslo"
        unit = contents.get("Unit")
        if name != "Time-series":
            return
        if unit is None or unit != ["kWh"]:
            print "Skipping section %s with unit %s" % (name, unit)
            return
        start = pd.Timestamp(_convert_to_datetime(contents['Start'][0]), 
                             tz=file_tz).tz_convert("UTC")
        end = pd.Timestamp(_convert_to_datetime(contents['Stop'][0]), 
                             tz=file_tz).tz_convert("UTC")
        if contents['Step'] != ['0000-00-00.01:00:00']:
            raise ValueError("This function assumes that all data are stored as "\
                             "1-hour timesteps")
        freq = 'H'
        values = contents['Value']
        index = pd.date_range(start, end, freq=freq)[:-1]
        data = np.array([self._valuestring_to_numbers(vs) for vs in values])
        if len(index) != len(data):
            raise ValueError("The number of values does not match the period.")
        return pd.DataFrame(data=data, index=index, columns=('Load', 'Status Code'))

    def _valuestring_to_numbers(self, value_string):
        """Convert a value string to a number tuple. First element is the value,
        second element is the status code converted to float, or numpy.nan if no
        code is found."""
        if '/' in value_string:
            (value, timestamp, code) = value_string.split('/')
            if timestamp != '':
                raise ValueError("Invalid timestamp. Timestamp was '%s', should " \
                                 "have been empty." % timestamp)
            return (float(value), float(code))
        else:
            return (float(value_string), np.nan)

    def _merge_timeseries(self, source, target):
        """Merge timeseries 'source' into timeseries 'target' in such a way that
        dates are aligned. target is a tuple containing concatenated time series
        with and without duplicates."""
        target_duped, target_nodup = target
        return (pd.concat((source, target_duped)),
                source.combine_first(target_nodup))

    def _split_tuple_dict(self):
        """Input a dict where each value is a tuple, return a tuple of two
        dicts. Empties self._user_seris on the way."""
        dict_tuple = (dict(), dict())
        keys = self._user_series.keys()
        for key in keys:
            (first, second) = self._user_series.pop(key)
            dict_tuple[0][key] = first
            dict_tuple[1][key] = second
        return dict_tuple



def store_in_hdf5(user_series, h5_file_path, complib='bzip2', complevel=9):
    timer = SimpleTimer()
    storer = PandasH5Storer(h5_file_path, complevel=complevel, complib=complib)
    storer.store_list("user_ids", 
                      [int(userid) for userid in user_series.keys()])
    for (userid, timeseries) in user_series.iteritems():
        storer.store_pd_user(int(userid), timeseries)
    print "All series stored in database."

if __name__ == '__main__':
    print "Parsing GS2 files..."
    sys.stdout.flush()
    user_series = GS2ToDataFrame().parse("gs2_for_prediction.txt")
    print "Not storing values with duplicates to HDF5..."
    # store_in_hdf5(user_series[0], "tempfeeder_with_duplicates.blosc-9.h5", 
    #     'blosc', 9)
    print "Storing values without duplicates to HDF5..."
    store_in_hdf5(user_series[1], "tempfeeder_without_duplicates.blosc-9.h5", 
        'blosc', 9)
    print "All done."
