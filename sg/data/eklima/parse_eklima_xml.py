import xml.etree.cElementTree as et
import sys, pdb
import re
import pandas as pd
from datetime import datetime
import calendar
import sg.utils

def parse(file):
    cal = dict((v,k) for k,v in enumerate(calendar.month_name))
    xml = et.parse(file)
    root = xml.getroot()
    station_name = root.findall('table/Stnr/Name')[0].text
    TS = []

    for table in root.findall('table'):
        if station_name in table.attrib['name']:
            month, year = table.attrib['name'].split(station_name)[-1].split()
            for date in table.findall('Date'):
                try:
                    day = int(date.attrib['id'])
                    data = [ float(ele.text) for ele in date.getchildren() if re.search('TA_\d*', ele.tag) ]
                    hours = [ int(ele.tag.split('_')[-1]) for ele in date.getchildren() if re.search('TA_\d*', ele.tag) ]
                    dates = [ datetime(year=int(year), month=cal[month], day=day, hour=hour) for hour in hours ]
                    TS.append(pd.Series(data=data, index=dates))
                except ValueError:
                    pass

    return pd.concat((TS)) 

if __name__ == "__main__":
    sg.utils.plot_time_series([parse(sys.argv[1])], ['-'], ['Dummy station'])
