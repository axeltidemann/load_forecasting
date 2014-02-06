import os
import datetime
import csv
import sys

_PATH_TO_HERE = os.path.dirname(os.path.abspath(__file__))
_BC_HOLIDAYS_FILE = os.path.join(_PATH_TO_HERE, "bcholidays.txt")

def calc_easter(year):
    """
    An implementation of Butcher's Algorithm for determining the date of Easter
    for the Western church. Works for any date in the Gregorian calendar (1583
    and onward). Returns Easter Sunday as a date object. 

    Based on
    http://code.activestate.com/recipes/576517-calculate-easter-western-given-a-year/
    """
    a = year % 19
    b = year // 100
    c = year % 100
    d = (19 * a + b - b // 4 - ((b - (b + 8) // 25 + 1) // 3) + 15) % 30
    e = (32 + 2 * (b % 4) + 2 * (c // 4) - d - (c % 4)) % 7
    f = d + e - 7 * ((a + 11 * d + 22 * e) // 451) + 114
    month = f // 31
    day = f % 31 + 1    
    return datetime.date(year, month, day)

def _weekdays():
    return {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
            'Friday': 4, 'Saturday': 5, 'Sunday': 6, 'Mon': 0, 'Tue': 1, 
            'Wed': 2, 'Thu': 3, 'Fri': 4, 'Sat': 5, 'Sun': 6}

def _directions():
    return {'before': -1, 'after': 1}

def _months():
    months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 
              'August', 'September', 'October', 'November', 'December']
    return dict(zip(months, range(1, len(months)+1)))
    
def nth_relative_to_date(reference, weekday, direction, count):
    """Return the date of the nth weekday before or after a given
    date. E.g. first Monday before May 25 2010."""
    weekdays = _weekdays()
    directions = _directions()
    try:
        weekday = weekdays[weekday]
    except:
        raise RuntimeError("Weekday must be one of Monday-Sunday or Mon-Sun.")
    try:
        direction = directions[direction]
    except:
        raise RuntimeError("Direction must be 'before' or 'after'.")
    delta = datetime.timedelta(days=direction)
    hits = 0
    now = reference
    while hits < count:
        while True:
            now = now + delta
            if now.weekday() == weekday:
                hits += 1
                break
    return now

def as_date(string, year):
    """Convert a string consisting of 'month daynumber', and an integer year
    to a datetime."""
    try:
        dt = datetime.datetime.strptime(str(year) + ' ' + string, '%Y %b %d')
    except:
        dt = datetime.datetime.strptime(str(year) + ' ' + string, '%Y %B %d')
    return dt.date()

def is_a_date(string, year=None):
    """Return True if string consists of 'month daynumber', e.g. December
    24. False otherwise. Accepts Feb 29 as a legal date if year is None or a
    leap year."""
    if year is None:
        year = 2008
    try:
        as_date(string, year)
        return True
    except:
        return False

def canada_day(year):
    """Return Canada day of the given year, i.e. July 1 or July 2 if July 1 is
    a Sunday."""
    cd = datetime.datetime.strptime(str(year) + " 7 1", '%Y %m %d')
    if cd.weekday() == 6:
        cd = datetime.datetime.strptime(str(year) + " 7 2", '%Y %m %d')
    return cd.date()

def nth_day_of_month(year, description):
    nth, day, ofin, month = description.split()
    ordinals = {'First': 1, 'Second': 2, 'Third': 3, 'Fourth': 4}
    if not nth in ordinals:
        raise RuntimeError("'" + nth + "' not recognized as an ordinal number.")
    if not day in _weekdays():
        raise RuntimeError("'" + day + "' not recognized as name of weekday.")
    if ofin != "of" and ofin != "in":
        raise RuntimeError(
            "Failed to parse '" + description + "' into 'nth day of/in month'.")
    if not month in _months():
        raise RuntimeError("'" + month + "' not recognized as name of month.")
    month = _months()[month]
    reference = datetime.date(year, month, 1) - datetime.timedelta(days=1)
    return nth_relative_to_date(reference, day, 'after', ordinals[nth])

def weekday_before_date(year, description):
    day, direction, date0, date1 = description.split()
    if not day in _weekdays():
        raise RuntimeError("'" + day + "' not recognized as name of weekday.")
    if not direction in _directions():
        raise RuntimeError("'" + direction + "' is not a valid direction.")
    if date0 == "Easter" and date1 == "Sunday":
        reference = calc_easter(year)
    elif is_a_date(date0 + " " + date1, year):
        reference = as_date(date0 + " " + date1, year)
    else:
        raise RuntimeError("Did not recognize '" + date0 + " " + date1 + \
                           "' as a date.")
    return nth_relative_to_date(reference, day, direction, 1)

        
def _date_from_year_name_and_description(year, name, description):
    if name == "Canada Day":
        return canada_day(year)
    if is_a_date(description, year):
        return as_date(description, year)
    try:
       return nth_day_of_month(year, description)
    except:
        return weekday_before_date(year, description)
        
def bc_holidays_for_year(year):
    """Parse the BC holidays file, return dates for holidays in the given
    year. Returns a list of tuples (name_of_holiday, date)."""
    holidays = []
    with open(_BC_HOLIDAYS_FILE, 'rb') as csvfile:
        dialect = csv.Sniffer().sniff(csvfile.read(1024))
        csvfile.seek(0)
        reader = csv.reader(csvfile, dialect)
        heading = reader.next()
        for row in reader:
            if len(row) < 2:
                raise RuntimeError(
                    "Format error in holiday description file, failed to "
                    "csv-parse line into [name, description]: " + str(row))
            if row[0] == "Family Day" and year < 2013:
                #print >>sys.stderr, "Skipping Family day for years before 2013."
                continue
            try:
                holidays.append(
                    (row[0], _date_from_year_name_and_description(
                        year, row[0], row[1])))
            except:
                raise RuntimeError(
                    "Failed to parse holiday '" + row[0] + \
                    "' with description '" + row[1] + "'.")
    return holidays

if __name__ == "__main__":
    from unittest import main
    main(module="test_" + __file__[:-3])
    
    
