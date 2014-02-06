import os
import datetime
import unittest
import StringIO
import tempfile

import numpy as np

from bchydro import * 
import sg.utils.testutils as testutils

_PATH_TO_HERE = os.path.dirname(os.path.abspath(__file__))
_PATH_TO_BCH_TESTDATA = os.path.join(_PATH_TO_HERE, "data_for_testing.csv")


class TestStoreInDatabase(testutils.ArrayTestCase):
    def setUp(self):
        handle, self.dbpath = tempfile.mkstemp(
            prefix="unittest_db_", suffix='.db', dir=".")
        os.close(handle)
        with open(_PATH_TO_BCH_TESTDATA, "r") as f:
            (self.timestamps, self.loads) = load_stream(f)
        # Add a NULL to check that it propagates correctly through the
        # database.
        self.loads[1] = np.nan
        with sqlite3.connect(self.dbpath) as conn:
            crs = conn.cursor()
            sql = 'CREATE TABLE loads ("Timestamp" datetime ' \
                'unique not null primary key, "MWh" float)'
            crs.execute(sql)

    def tearDown(self):
        if os.path.exists(self.dbpath):
            os.remove(self.dbpath)

    def test_storedup(self):
        """Check that storing a duplicate fails with an
        appropriate message."""
        stamps = self.timestamps
        stamps[0] = stamps[1]
        with self.assertRaises(RuntimeError) as cm:
            store_in_database(self.dbpath, stamps, self.loads)
                             
    def test_store(self):
        """Test that storing works by retrieving and comparing
        with original values."""
        nul=StringIO.StringIO()
        store_in_database(self.dbpath, self.timestamps, self.loads, stream=nul)
        with sqlite3.connect(self.dbpath, detect_types=sqlite3.PARSE_DECLTYPES|
                             sqlite3.PARSE_COLNAMES) as conn:
            crs = conn.cursor()
            crs.execute("SELECT Timestamp as 't [timestamp]', "
                         "MWh as 'l [float]' FROM loads")
            timestamps, loads = zip(*crs.fetchall())
        # Convert NULLs back to np.NaN, by specifying datatype float, otherwise
        # internal representation in numpy will be object (not
        # float). pandas.Series constructor converts list of float+None to
        # float+np.nan, but tuples are cast to ndarray.  This keeps the Nones
        # and stores the entire array as dtype=object.
        dbts = pd.Series(loads, index=timestamps, dtype=float)
        myts = pd.Series(self.loads, index=self.timestamps, dtype=float)
        self.assertNaNArraysEqual(myts, dbts)

    def test_loaddata(self):
        """Test loading by storing, loading and comparing with original
        values."""
        self.test_store()
        dbts = load(self.dbpath)
        myts = pd.Series(self.loads, index=self.timestamps)
        self.assertArraysEqual(myts.fillna(0.1), dbts.fillna(0.1))
        self.assertArraysEqual(myts.fillna(0.2), dbts.fillna(0.2))


class TestParseLine(unittest.TestCase):
    def test_wrongnumber(self):
        self.assertRaises(ValueError, parseline, "April 1, 2004; 1")
        self.assertRaises(ValueError, parseline, "April 1, 2004; 1; 12; 13")

    def test_parse_invalid_date(self):
        self.assertRaises(ValueError, parseline, "Aapril 12, 2004; 24; 123")
        self.assertRaises(ValueError, parseline, "June 12, 2004; 26; 1234")

    def test_parse_valid_date(self):
        (timestamp, value) = parseline("April 12, 2004; 24; 123")
        self.assertTrue(timestamp.year == 2004 and
                        timestamp.month == 4 and
                        timestamp.day == 12 and
                        timestamp.hour == 23 and
                        value == 123)
        (timestamp, value) = parseline("June 12, 1996; 16; 12345")
        self.assertTrue(timestamp.year == 1996 and
                        timestamp.month == 6 and
                        timestamp.day == 12 and
                        timestamp.hour == 15 and
                        value == 12345)


class TestCheckTimeDifference(unittest.TestCase):
    def setUp(self):
        self.d1 = datetime.datetime(year=2000, month=1, day=1, hour=12)
        self.d2 = datetime.datetime(year=2001, month=1, day=1, hour=12)
        self.d3 = datetime.datetime(year=2001, month=1, day=1, hour=13)
        self.d4 = datetime.datetime(year=2001, month=1, day=1, hour=14)
        
    def test_check_fails(self):
        self.assertRaises(RuntimeError, check_time_difference, self.d1, self.d2)
        self.assertRaises(RuntimeError, check_time_difference, self.d1, self.d3)
        self.assertRaises(RuntimeError, check_time_difference, self.d1, self.d4)
        self.assertRaises(RuntimeError, check_time_difference, self.d2, self.d4)
        self.assertRaises(RuntimeError, check_time_difference, self.d4, self.d3)

    def test_check_succeeds(self):
        self.assertIsNone(check_time_difference(self.d2, self.d3))
        self.assertIsNone(check_time_difference(self.d3, self.d4))

class TestLoadStream(unittest.TestCase):
    def test_loadstream(self):
        contents = "April 1, 2004;1;6055\n" + "April 1, 2004;2;5829\n" + \
            "April 1, 2004;3;5701\n" + "April 1, 2004;4;5736\n"
        stream = StringIO.StringIO(contents)
        (timestamps, loads) = load_stream(stream)
        self.assertEqual(loads, [6055, 5829, 5701, 5736])
        testtimes = [datetime.datetime(year=2004, month=4, day=1, hour=0),
                     datetime.datetime(year=2004, month=4, day=1, hour=1),
                     datetime.datetime(year=2004, month=4, day=1, hour=2),
                     datetime.datetime(year=2004, month=4, day=1, hour=3)]
        self.assertEqual(timestamps, testtimes)

if __name__ == '__main__':
    unittest.main()
