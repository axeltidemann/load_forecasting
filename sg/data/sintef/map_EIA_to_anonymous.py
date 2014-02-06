"""This script was written to create a mapping from actual installation IDs to
the anonymized ones, after the anonymization process has been performed. The
motivation for this is that the anonymous IDs have already been used
extensively, but for Buskerud we need to select readings based on actual IDs,
in order to build a load profile from all meters below a certain substation or
other grid connection point.

The script expects the output of a 'diff' between original and anonymized files
(in that order!) as input. If the files differ in any other way than in IDs,
the script will fail by design."""

# A diff output in "normal format" consists of hunks of differences. Each hunk
# has four parts: the change command, the text from the left input file, a
# separator, and the text from the right input file (ref
# http://www.chemie.fu-berlin.de/chemnet/use/info/diff/diff_3.html).

# Usage:
# cat gs2_buskerud.txt| while read line; do diff "$line" "`echo $line | sed -e's*/unanom**'`"; done |python map_EIA_to_anonymous.py
#
# Note that "for line in $(<gs2_buskerud_short.txt); do ..." will not work, as
# there are directory names with spaces in the file.

import sys

from sg.utils import ask_user

class DiffParser(object):
    def __init__(self):
        self._reset(stream=None)
        self._ids = set(("#Installation", "#Reference", "#Plant"))
        self._id_sep = "="
        
    def _reset(self, stream):
        self._stream = stream
        self._lineno = 1
        self._map = dict()

    def _parse_change_command(self):
        """Parse a change command. This should consist of <line>c<line> or
        <line1,line2>c<line1,line2>. Returns the number of lines changed, or 0
        on end of file."""
        line = self._stream.readline()
        if len(line) == 0:
            return 0
        self._lineno += 1
        left, right = line[:-1].split('c')
        if left != right:
            raise RuntimeError("Error in change command, line mismatch between files: %d vs %d." % (left, right))
        try:
            (d1_1, d1_2) = [int(d) for d in left.split(',')]
            (d2_1, d2_2) = [int(d) for d in right.split(',')]
            return d1_2 - d1_1 + 1
        except:
            pass
        try:
            d1, d2 = (int(left), int(right))
            return 1
        except:
            raise RuntimeError("Failed to parse change command.")
        
    def _next_line(self, desc):
        line = self._stream.readline()
        if len(line) == 0:
            raise RuntimeError("Error while parsing %s: Unexpected end of file." % desc)
        self._lineno += 1
        return line[:-1]
        
    def _parse_contents(self, prefix):
        line = self._next_line("diff contents")
        if len(prefix) > len(line) or line[:len(prefix)] != prefix:
            raise RuntimeError("Error while parsing diff contents: Expected '%s', got '%s'." \
                               % (prefix, line[:len(prefix)]))
        (tag, value) = line[len(prefix):].split(self._id_sep)
        if not tag in self._ids:
            raise RuntimeError("Error while parsing diff contents: Not a recognized identifier: '%s'." % tag)
        return value

    def _parse_separator(self):
        line = self._next_line("separator")
        sep = "---"
        if sep != line:
            raise RuntimeError("Error while parsing separator: Expected '%s', got '%s'." % (sep, line))
            
    def _parse_hunk(self):
        num_changes = self._parse_change_command()
        if num_changes == 0:
            return False
        keys, values = [], []
        for _ in range(num_changes):
            keys.append(self._parse_contents("< "))
        self._parse_separator()
        for key in keys:
            self._map[key] = self._parse_contents("> ")
        return True

    def parse(self, input_stream):
        self._reset(input_stream)
        try:
            while self._parse_hunk():
                pass
        except Exception as e:
            print >>sys.stderr, "Error while parsing, probably on line %d." % self._lineno
            print >>sys.stderr, "Exception message: "
            print >>sys.stderr, e
            return
        return self._map

def _interactive(pickle_path):
    if ask_user("Input parsed. Save map to %s" % pickle_path, None):
        import cPickle as pickle
        with open(pickle_path, "wb") as f:
            pickle.save(f, id_map)
        print "Done."
    else:
        for (key, value) in id_map.iteritems():
            print key, ":", value

def _main():
    id_map = DiffParser().parse(sys.stdin)
    if id_map is None:
        print >>sys.stderr, "Parsing failed."
    else:
        pickle_path = "id_map.pickle"
        import cPickle as pickle
        with open(pickle_path, "wb") as f:
            pickle.dump(id_map, f)
        print "Done, mapping saved to", pickle_path

if __name__ == "__main__":
    _main()
