"""Takes as input a list of gs2-files that has temperature data, reads
them and plots them.
Example: grep -l Grader *.exp | python path/to/plot_temp.py """

import string
import sys
import datetime

import pandas as pd

import sg.utils
import sg.data.sintef.parse_gs2 as parse

def collect_and_plot(files):
    TS = []
    location = []
    for f in files:
        temperatures = [ section[1] for section in parse.parse_file(f)[1:-1] if section[1]['Plant'] == ['tmp'] ]
        for t in temperatures:
            if t['Step'][0] != '0000-00-00.01:00:00':
                print 'Not hourly readings of temperature. Abort.'
                break
            start_time = datetime.datetime.strptime(t['Start'][0], "%Y-%m-%d.%H:%M:%S")
            dates = pd.date_range(start=start_time, periods=len(t['Value']), 
                                    freq='H')
            data = [ float(value.rsplit('/')[0]) for value in t['Value'] ]
            TS.append(pd.Series(data=data, index=dates))
            if location and t['Installation'][0] != location:
                print 'Location changed during reading of gs2 files. Probably some bad grouping of gs2 files.'
            location = t['Installation'][0]
    if TS:
        all_series = pd.concat(TS).sort_index()
        all_series_no_duplicates = all_series.groupby(level=0).first()
        all_series_no_duplicates.dump('temp_data.pickle')
        sg.utils.plot_time_series([all_series_no_duplicates], ['b-'], [location])
    else:
        print 'No temperature data.'

if __name__ == "__main__":
    if not sys.stdin.isatty():
        collect_and_plot([ s.rstrip('\n') for s in sys.stdin.readlines() ])

