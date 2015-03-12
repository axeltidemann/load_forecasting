#!/bin/bash
#
# This script takes a list of output_....txt evolution log files as
# input, and parses them into one big csv file where the fitnesses
# (scaled and raw) and genes of each individual in each generation is
# stored. Each line is prepended with the run number and the generation
# number.
#
# Input: List of files to parse, already bunzipped
# Output: Huge CSV file

inputs=$@
output=all.csv

dirs="generated filed genes stripped"
for d in $dirs; do
    if test -e $d; then
        echo "Temporary directory '$d' already exists. Exiting."
        exit
    fi
done
if test -e $output; then
    echo "Output file '$output' already exists. Exiting."
    exit
fi

mkdir -v $dirs

echo "Appending generation number to each line of logs..."
for f in $inputs; do
    awk 'BEGIN{gen=0} /Best genome at generation [0-9]* had/{gen++} {print gen, ",", $0}' $f >generated/$f;
done

echo "Appending run number to each line in each file..."
pushd generated/
for f in $inputs; do
    run=`echo $f | awk -F_ '{print $4}'`;
    awk "{print $run, \",\", \$0}" $f >../filed/$f;
done
popd

echo "Extracting lines with genes and fitnesses (population dump) from logs"
pushd filed/
for f in $inputs; do
    gsed -n -e '/^[][0-9,[:space:]e.+-]\+$/p' $f |gsed -n -e'/[][]\+/p' >../genes/$f;
done
popd
echo "Removing braces from genes, comma-separating fields."
pushd genes/
for f in $inputs; do
    sed -e's/[][]//g' $f |sed -e's/\([0-9]\) \([0-9]\)/\1, \2/g';
done >../$output
popd

echo "CSV saved to $output."

echo "Removing temporary directories."
rm -rf $dirs
