#!/bin/bash

NAME=$0
if test -n "`type rev 2>/dev/null`"; then
    NAME="`echo $0 | rev | cut -d '/' -f 1 | rev`";
fi

NAME=$0
if test -n "`type basename 2>/dev/null`"; then
    NAME="`basename $0`";
fi

submitter=$USER

USAGE="Usage: 
   $NAME jobscript [ more jobscripts...]

Submit the job(s) given on the command line, as long as it/they aren't already
present in the job queue.

Options:
  -h
    Prints this help.
  -u user
    Check jobs submitted by user rather than jobs submitted by '$submitter'.
"

while getopts u:h'?' opt
do
    case $opt in
        u)
            submitter=$OPTARG;;
	h|'?'|?|*)
	    echo "$USAGE"
	    exit 2;;
    esac
done
shift `expr $OPTIND - 1`

if [ $# -eq 0 ]; then
    echo "$USAGE"
    exit 2
fi

for job in $@; do
    if [ -z "`qstat -f -u $submitter | grep "$job"`" ]; then
        echo "(Re)submitting job $job:"
        qsub "$job"
    else
        echo "Job '$job' already in queue, skipping."
    fi
done
