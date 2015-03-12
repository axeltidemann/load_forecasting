#!/bin/bash

# Split the output from running a prediction model on the test set (i.e. the
# output file from running a GA) into two halves: validation and test.
#
# In other words: filter the input, keeping only lines containing
# $filter_pattern. Split the resulting set of lines in the middle, and average
# the values found in the last field on each line in each half. 

filter_pattern="^Error for test at"
filter () {
    extension="${1##*.}"
    if [ "$extension" == "bz2" ]; then
        CAT=bzcat
    else
        CAT=cat
    fi
    $CAT $1 | sed -n -e"/$filter_pattern/p"
}

split_and_calc() {
    flines=`filter $1 | wc | awk '{print $1}'`
    filter $1 \
        | awk "{
                  if (NR <= $flines/2) {
                     valid += (\$NF)^2;
                     vlines++;
                  } else {
                     test += (\$NF)^2;
                     tlines++;
                  }
               } 
               END {
                  print \"This script assumes RMSE for each day is based on the same number of predictions (24).\";
                  print \"RMSE on validation phase (\", vlines, \" lines): \", sqrt(valid/vlines);
                  print \"RMSE on test phase (\", tlines, \" lines): \", sqrt(test/tlines);
               }"
    echo ""
}

split_and_calc $1
