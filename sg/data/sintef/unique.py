import sys
import collections
import pprint

tags = collections.defaultdict(int)
for line in sys.stdin:
    tags[line[:-1]] += 1

for key in tags:
    print key, ":", tags[key]

