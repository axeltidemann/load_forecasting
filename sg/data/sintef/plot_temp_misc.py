"""Takes as input a list of gs2-files that has temperature data, reads
them and plots them.
Example: grep -l Grader *.exp | python path/to/plot_temp.py """

import string
import matplotlib.pyplot as plt
import sg.data.sintef.parse_gs2 as parse
import sys, os
import scikits.timeseries as ts
import scikits.timeseries.lib.plotlib as tpl
import sg.utils
import sg.data.eklima.parse_eklima_xml as xml

def _collect_and_plot(files):
    TS = []
    location = []
    for f in files:
        temperatures = [ section[1] for section in parse.parse_file(f)[1:-1] if section[1]['Plant'] == ['tmp'] ]
        for t in temperatures:
            if t['Step'][0] != '0000-00-00.01:00:00':
                print 'Not hourly readings of temperature. Abort.'
                break
            dates = ts.date_array(start_date=ts.Date('H', t['Start'][0]), length=len(t['Value']))
            data = [ float(value.rsplit('/')[0]) for value in t['Value'] ]
            TS.append(ts.TimeSeries(data=data, dates=dates))
            if location and t['Installation'][0] != location:
                print 'Location changed during reading of gs2 files. Probably some bad grouping of gs2 files.'
            location = t['Installation'][0]
    if TS:
        path = '/Users/tidemann/Documents/NTNU/devel/data/eklima/Telemark/'
        for file in os.listdir(path):
            try:
                series = xml.parse(path + file)
                sg.utils.plot_time_series([ts.concatenate((TS)), series], ['b-','r-'], [location, file])
            except:
                print file, 'had no data.'
    else:
        print 'No temperature data.'

if __name__ == "__main__":
    if not sys.stdin.isatty():
        _collect_and_plot([ s.rstrip('\n') for s in sys.stdin.readlines() ])

