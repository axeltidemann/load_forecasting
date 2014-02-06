import unittest
import os
import StringIO

_PATH_TO_HERE = os.path.dirname(os.path.abspath(__file__))
_PATH_TO_GS2_TESTDATA = os.path.join(_PATH_TO_HERE, "testfile.gs2")
_PATH_TO_GS2_TESTDATA_SHORT = os.path.join(_PATH_TO_HERE, "testfile_short.gs2")

import parse_gs2 as gs2

class TestGS2Parser(unittest.TestCase):
    def test__is_entry(self):
        self.assertTrue(gs2._is_entry("#Sum=0\n"))
        self.assertFalse(gs2._is_entry(""))
        self.assertFalse(gs2._is_entry(" something is wrong"))

    def test__is_heading(self):
        self.assertTrue(gs2._is_heading("##Heading"))
        self.assertFalse(gs2._is_heading("#Entry"))
        self.assertFalse(gs2._is_heading("Neither"))

    def _set_up_section(self, heading):
        line = "##%s\n" % heading
        return gs2._make_section(line)
        
    def test_make_section(self):
        heading = "Time-series"
        section = self._set_up_section(heading)
        self.assertEqual(section[0], heading)
        self.assertIs(type(section[1]), dict)

    def test_split_single_line(self):
        key, value = gs2._split_line("#No-of-values=168\n", None)
        self.assertEqual(key, "No-of-values")
        self.assertEqual(value, ["168"])

    def test_split_multi_line(self):
        line = "#Value=<\n"
        stream = StringIO.StringIO(" 2.2//0\n 2.5//0\n 2.8//0\n>\n")
        key, value = gs2._split_line(line, stream)
        self.assertEqual(key, "Value")
        self.assertEqual(value, ["2.2//0", "2.5//0", "2.8//0"])
        
    def test_split_raises(self):
        self.assertRaises(ValueError, gs2._split_line,
                          "#No-of-values:168\n", None)

    def _set_up_section_generator(self, path):
        stream = open(path, "r")
        return [section for section in gs2.section_generator(stream)]
        
    def test_section_generator(self):
        parsed = self._set_up_section_generator(_PATH_TO_GS2_TESTDATA_SHORT)
        self.assertEqual(len(parsed), 4)
        self.assertEqual(parsed[0][0], "Start-message")
        self.assertEqual(parsed[1][0], "Time-series")
        self.assertEqual(parsed[2][0], "Time-series")
        self.assertEqual(parsed[3][0], "End-message")
        self.assertEqual(parsed[1][1]["Value"],
                         ["0//0", "1.285//0", "0//0", "1.285//0", "0//0",
                          "1.285//0", "0//0"])

    def test_parse_gs2(self):
        parsed_manual = self._set_up_section_generator(_PATH_TO_GS2_TESTDATA)
        parsed_auto = gs2.parse_file(_PATH_TO_GS2_TESTDATA)
        self.assertEqual(parsed_manual, parsed_auto)
        self.assertEqual(len(parsed_auto), 29)
        self.assertEqual(parsed_auto[0][0], "Start-message")
        for i in range(1, 28):
            self.assertEqual(parsed_auto[i][0], "Time-series")
        self.assertEqual(parsed_auto[-1][0], "End-message")

    def test_find_first_heading(self):
        stream = StringIO.StringIO("##Start-message\n#Id=PD-gs2exp\n")
        line = gs2._find_first_heading(stream)
        self.assertEqual(line, "##Start-message\n")

    def test_find_first_heading_no_heading(self):
        stream = StringIO.StringIO("#Start-message\n#Id=PD-gs2exp\n")
        line = gs2._find_first_heading(stream)
        self.assertIs(line, None)

    def test_find_first_heading_empty_file(self):
        stream = StringIO.StringIO("")
        line = gs2._find_first_heading(stream)
        self.assertIs(line, None)

if __name__ == '__main__':
    unittest.main()
