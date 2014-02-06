#!/bin/bash

# Remove lines 2-35 (all the PBS and virtualenv stuff) and the --parallel
# option from all job files.
for f in jobscript_evo_*; do
    terminator="^sg_dir="
    gsed -i -e"2,/$terminator/ { /$terminator/b; d }; /--parallel/d" $f
done

