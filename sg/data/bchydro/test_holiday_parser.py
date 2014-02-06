import os
import sys
import unittest
from datetime import date

import numpy as np

import sg.utils.testutils as testutils

from holiday_parser import * 

_PATH_TO_HERE = os.path.dirname(os.path.abspath(__file__))

class Test(unittest.TestCase):
    def test_nth_relative(self):
        reference = date(2013, 6, 21)
        # Find first Monday before
        found = nth_relative_to_date(reference, 'Mon', 'before', 1)
        self.assertEqual(found, date(2013, 6, 17))
        # Find second Monday before
        found = nth_relative_to_date(reference, 'Mon', 'before', 2)
        self.assertEqual(found, date(2013, 6, 10))
        # Find second Thursday before
        found = nth_relative_to_date(reference, 'Thu', 'before', 2)
        self.assertEqual(found, date(2013, 6, 13))
        # Find first Friday before (today is a Friday)
        found = nth_relative_to_date(reference, 'Fri', 'before', 1)
        self.assertEqual(found, date(2013, 6, 14))
        # Find first Monday after
        found = nth_relative_to_date(reference, 'Mon', 'after', 1)
        self.assertEqual(found, date(2013, 6, 24))
        # Find tomorrow (a Saturday)
        found = nth_relative_to_date(reference, 'Sat', 'after', 1)
        self.assertEqual(found, date(2013, 6, 22))
        # Check a different year just for fun
        reference = date(2010, 4, 10) # Saturday
        found = nth_relative_to_date(reference, 'Sat', 'after', 1)
        self.assertEqual(found, date(2010, 4, 17))
        # Check bad input
        self.assertRaises(RuntimeError, nth_relative_to_date, reference, 'bunk', 'after', 1)
        self.assertRaises(RuntimeError, nth_relative_to_date, reference, 'Sat', 'bafter', 1)
        
    def test_is_a_date(self):
        self.assertTrue(is_a_date('Feb 29'))
        self.assertTrue(is_a_date('Feb 29', 2008))
        self.assertFalse(is_a_date('Feb 29', 2009))
        self.assertTrue(is_a_date('February 29'))
        self.assertTrue(is_a_date('March 29'))
        self.assertTrue(is_a_date('Mar 31'))
        self.assertTrue(is_a_date('January 1'))
        self.assertFalse(is_a_date('Jan 0'))
        self.assertFalse(is_a_date('April 31'))
        self.assertFalse(is_a_date('This is not a date'))
        self.assertFalse(is_a_date('Bogus 21'))
        self.assertFalse(is_a_date('Jan -9'))
        
    def test_weekday_before_date(self):
        date = weekday_before_date(2010, "Friday before Easter Sunday")
        # Easter was on April 4, Friday before was April 2
        self.assertEqual(date, datetime.date(2010, 4, 2))
        date = weekday_before_date(2011, "Monday before August 25")
        self.assertEqual(date, datetime.date(2011, 8, 22))
        date = weekday_before_date(2011, "Thursday after Nov 25")
        self.assertEqual(date, datetime.date(2011, 12, 1))
        self.assertRaises(RuntimeError, weekday_before_date, 2011, "Nosuchday after Nov 25")
        self.assertRaises(RuntimeError, weekday_before_date, 2011, "Thursday nosuchdir Nov 25")
        self.assertRaises(RuntimeError, weekday_before_date, 2011, "Thursday after invalid date")

    def test_nth_day_of_month(self):
        date = nth_day_of_month(2010, "First Monday of August")
        self.assertEqual(date, datetime.date(2010, 8, 2))
        date = nth_day_of_month(2010, "Second Monday in August")
        self.assertEqual(date, datetime.date(2010, 8, 9))
        date = nth_day_of_month(2012, "First Sunday of January")
        self.assertEqual(date, datetime.date(2012, 1, 1))
        date = nth_day_of_month(2012, "Third Monday of January")
        self.assertEqual(date, datetime.date(2012, 1, 16))
        self.assertRaises(RuntimeError, nth_day_of_month, 2011, "Nosuchordinal Monday of January")
        self.assertRaises(RuntimeError, nth_day_of_month, 2011, "Third Nosuchday of January")
        self.assertRaises(RuntimeError, nth_day_of_month, 2011, "Third Monday notofin January")
        self.assertRaises(RuntimeError, nth_day_of_month, 2011, "Third Monday of Notamonth")
        
    def test_bc_holidays(self):
        # List from http://www.statutoryholidays.com/bc.php. Canada day not
        # correctly moved to the following Monday when it falls on a Sunday
        # (2012). Civic Day is called British Columbia Day in BC. It appears to
        # be on the first Monday in August, rather than on first Monday after
        # first Sunday of August, as it says below.
        # 
        holidays = [['Holiday', 'Description', '2010', '2011', '2012', '2013', '2014'],
                    ["New Year's Day", 'January 1', 'Fri, January 1', 'Sat, January 1', 'Sun, January 1', 'Tue, January 1', 'Wed, January 1'],
                    ['Family Day', '2nd Monday in February', None, None, None, 'Mon, February 11', 'Mon, February 10'],
                    ['Good Friday', 'Friday before Easter Sunday', 'Fri, April 2', 'Fri, April 22', 'Fri, April 6', 'Fri, March 29', 'Fri, April 18'],
                    ['Easter Monday', 'Monday after Easter Sunday', 'Mon, April 5', 'Mon, April 25', 'Mon, April 9', 'Mon, April 1', 'Mon, April 21'],
                    ['Victoria Day', 'Monday before May 25', 'Mon, May 24', 'Mon, May 23', 'Mon, May 21', 'Mon, May 20', 'Mon, May 19'],
                    ['Canada Day', 'July 1', 'Thu, July 1', 'Fri, July 1', 'Mon, July 2', 'Mon, July 1', 'Tue, July 1'],
                    ['Civic Holiday', 'Monday after the 1st Sunday of August', 'Mon, August 2', 'Mon, August 1', 'Mon, August 6', 'Mon, August 5', 'Mon, August 4'],
                    ['Labour Day', 'First Monday in September', 'Mon, September 6', 'Mon, September 5', 'Mon, September 3', 'Mon, September 2', 'Mon, September 1'],
                    ['Thanksgiving', 'Second Monday in October', 'Mon, October 11', 'Mon, October 10', 'Mon, October 8', 'Mon, October 14', 'Mon, October 13'],
                    ['Remembrance Day', 'November 11', 'Thu, November 11', 'Fri, November 11', 'Sun, November 11', 'Mon, November 11', 'Tue, November 11'],
                    ['Christmas Day', 'December 25', 'Sat, December 25', 'Sun, December 25', 'Tue, December 25', 'Wed, December 25', 'Thu, December 25']]
        for yearcol in range(5, len(holidays[0])):
            year = int(holidays[0][yearcol])
            names = [line[0] for line in holidays[1:]]
            if year < 2013:
                names.remove('Family Day')
            yeardays = [line[yearcol] for line in holidays[1:]]
            dates = [datetime.datetime.strptime(
                day + ' ' + str(year), "%a, %B %d %Y").date()
                for day in yeardays if day is not None]
            holiday_list = dict(zip(names, dates))
            generated = dict(bc_holidays_for_year(year))
            for (desc, date) in generated.iteritems():
                self.assertIn(desc, holiday_list)
                self.assertEqual(date, holiday_list.pop(desc))
            self.assertTrue(len(holiday_list) == 0)
            
            
if __name__ == '__main__':
    unittest.main()
