#!/bin/bash

NAME=$0
if test -n "`type rev 2>/dev/null`"; then
    NAME="`echo $0 | rev | cut -d '/' -f 1 | rev`";
fi

NAME=$0
if test -n "`type basename 2>/dev/null`"; then
    NAME="`basename $0`";
fi

USAGE="Usage: 
   $NAME [options] [ file [ file2 ...] ]

Print the best genomes found (as alleles) by evolution for each of the files
given on the command line, or from standard input if no files are given. The
script works by finding the last line containing the word 'alleles', and then
printing everything inside brackets on that line.

Options:
  -a
    Print all generations, not just the last (or first).
  -r
    Print raw genes (instead of genes mapped to allele ranges) from old log files.
  -m
    Print mapped genes from old log files.
  -h
    Prints this help.
  -f
    Print the first generation rather than the last
"

KEYWORD="Best genome at generation [0-9]\+: \["
TAIL="tail -n 1"
while getopts afmrh'?' opt
do
    case $opt in
        a)
            TAIL="cat";;
        f)
            TAIL="head -n 1";;
        r)
            KEYWORD="raw genes";;
        m)
            KEYWORD="alleles";;
	h|'?'|?|*)
	    echo "$USAGE"
	    exit 2;;
    esac
done
shift `expr $OPTIND - 1`

if [ $# -eq 0 ]; then
    INPUT='-'
else
    INPUT="$@"
fi

for f in $INPUT; do 
    extension="${f##*.}"
    # Likewise, filename="${f%.*}"
    if [ "$extension" == "bz2" ]; then
        GREP=bzgrep
    else
        GREP=grep
    fi
    $GREP "$KEYWORD" $f |$TAIL;
done |sed -e's/.*\[//; s/\]$//'
