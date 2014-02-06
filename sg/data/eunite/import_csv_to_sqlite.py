"""Import load, temperature and holiday data from csv files into sqlite.

May not work without modification, after separating data from code. The code in
this file assumes the csv files are in the working directory of the
interpreter.

"""
 
import csv
import datetime
import os
import sqlite3

from sg.data.eunite import PATH_TO_EUNITE_DB

def import_data(load_path, temp_path, cursor, table_name):
    load_reader = csv.reader(open(load_path), delimiter=';')
    temp_reader = csv.reader(open(temp_path), delimiter=';')
    loads = [l for l in load_reader]
    temperatures = [t for t in temp_reader]
    assert(len(temperatures) == len(loads))
    for (temp, load) in zip(temperatures, loads):
        load = [int(l) for l in load]
        ldate = datetime.datetime(year=load[0], month=load[1], day=load[2])
        tdate = datetime.datetime.strptime(temp[0], "%Y-%m-%d")
        assert(ldate == tdate)
        deg_c = float(temp[1])
        for half_hour in range(len(load)-3):
            stamp = ldate + datetime.timedelta(hours=float(half_hour) / 2)
            cursor.execute("INSERT INTO %s VALUES (?, ?, ?)" % table_name,
                           (stamp, deg_c, load[half_hour + 3]))

def import_holidays(cursor):
    with open("holidays.csv") as f:
        for l in f:
            date = datetime.datetime.strptime(l[:-1], "%Y-%m-%d")
            cursor.execute('INSERT INTO holidays VALUES (?)', (date,))

def _reformat_date_jan_1999():
    """Run this function only once, to transform the date format of
    temperature_1999.csv into ISO."""
    reader = csv.reader(open("temperatures_1999.csv"), delimiter=";")
    for (day, month, temp) in reader:
        date = datetime.datetime.strptime("-".join(["1999", month, day]), 
                                          "%Y-%m-%d")
        print "%s; %s" % (date.strftime("%Y-%m-%d"), temp)

def clear_db(cursor):
        try:
            cursor.execute("DROP TABLE training")
        except:
            pass
        try:
            cursor.execute("DROP TABLE testing")
        except:
            pass
        try:
            cursor.execute("DROP TABLE holidays")
        except:
            pass

def setup_db(cursor):
    cursor.execute('CREATE TABLE holidays ' \
                       '("Timestamp" datetime unique not null primary key)')
    for table in ("training", "testing"):
        cursor.execute('CREATE TABLE %s ' \
                        '("Timestamp" datetime unique not null primary key, ' \
                        '"Deg_C" float, "MWh" float)' % table)

if __name__ == "__main__":
    with sqlite3.connect(PATH_TO_EUNITE_DB,
                         detect_types=sqlite3.PARSE_DECLTYPES) as conn:
        cursor = conn.cursor()
        clear_db(cursor)
        setup_db(cursor)
        import_holidays(cursor)
        import_data("loads.csv", "temperatures.csv", cursor, "training")
        import_data("loads_1999.csv", "temperatures_1999.csv", cursor, 
                    "testing")
