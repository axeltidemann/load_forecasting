#!/bin/bash

echo -n "Creating a list of all the GS2 files and storing it in ./gs2.txt.. "
find "`pwd`/../../../../data/sintef/raw" -iname '*.exp' -or -iname '*.gs2' >gs2.txt
echo "Done."

echo -n "Creating a list of a few small GS2 files and storing it in ./gs2_short.txt.. "
find "`pwd`/../../../../data/sintef/raw" -iname '*.exp' -or -iname '*.gs2' -size -3MB |head -n 5 >gs2_short.txt
echo "Done."
