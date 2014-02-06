#!/bin/bash

. $HOME/local/bin/query.sh

NAME=$0
if test -n "`type basename 2>/dev/null`"; then
    NAME="`basename $0`";
fi

USAGE="Usage: 
   $NAME [options]

Summarize simulation results, with or without plotting fitness graphs

Options:
-g
   Show graphs
"

show_graphs="FALSE"

while getopts 'gh?' opt
do
    case $opt in
        g)
            show_graphs="TRUE";;
	h|'?'|?|*)
	    echo "$USAGE"
	    exit 2;;
    esac
done
shift `expr $OPTIND - 1`


children() {
    ps -o pid,ppid,command | grep "[0-9][0-9]*[[:space:]]\+$$" | awk '{print $1}'
}

for dataset in "" _total-load _bc-data; do
    if [ -z "$dataset" ]; then
        ymax=0.7
    elif [ "$dataset" == "_total-load" ]; then
        ymax=0.07
    else
        ymax=0.009
    fi
    for model in wavelet arima arima_24 bmparima esn CBR; do
	clean=_no-cleaning # Axl: Everything is without cleaning. To avoid errors in the following loop (cleaning
                           # and subtract can not be in the same parameter, since they both exist in the filename), 
	                   # this is where the clean parameter loop should be.
        for subtract in "" _subtract-weekly-pattern _subtract-daily-pattern; do
            # Find the relevant log files. Using 'find' to ensure regexp search
            # rather than full wildcard matching on the '*' in the filename.
            GREP=bzgrep
            ext=txt.bz2
            pattern="./output_${model}_run_[0-9]*${dataset}${subtract}${clean}_100_0" # Axl note: _100_0 was added. 
            logs=`find . -regex "$pattern.${ext}"`
            if [ -z "$logs" ]; then
                echo -e "\nNo matches for $pattern.${ext}, trying non-compressed files."
                ext=txt
                logs=`find . -regex "$pattern.${ext}"`
                if [ -z "$logs" ]; then
                    echo "No matches for $pattern.${ext} either."
                    continue
                fi
                GREP=grep
            fi
            #echo $logs
            # Use log file paths to find database paths
            dbs=`echo $logs | sed -e"s/output_/pyevolve_/g; s/\.${ext}/.db/g"`

            # Make a "nice" title
            if [ -z "$dataset" ]; then
                datasettxt="single-user"
            else
                datasettxt=$dataset
            fi
            if [ -z "$clean" ]; then
                cleantxt="with cleaning"
            else
                cleantxt=$clean
            fi
            nlogs=`ls -1 $logs 2>/dev/null | wc | awk '{print $1}'`
            ndbs=`ls -1 $dbs 2>/dev/null | wc | awk '{print $1}'`
            title="$model ${datasettxt} ${subtract} ${cleantxt} ($nlogs output logs, $ndbs databases)"

            # Calculate average prediction errors
            mm="python $HOME/local/bin/minmax.py"
            echo -e "\n               $title"
            echo "Test set prediction error as stored in file (old and new runs use different measures):"
            $GREP "Error on test phase" $logs | awk '{print $12}' | $mm --header
            echo "Test set prediction error as mean of daily errors:"
            # for log in $logs; do
            #     echo "$GREP 'Error for test at' $log"
            #     $GREP 'Error for test at' $log | awk '{rows++; total += $NF}END{print total/rows}'
            # done
            for log in $logs; do
                $GREP 'Error for test at' $log | awk '{rows++; total += $NF}END{print total/rows}'
            done | $mm --header
            #echo "Fitnesses (not error) of last generation (rows=Min,ave,max,dev):"
            echo -n "Average maximum fitness (not error) of last generation: "
            # The sed selects the max fitness row, the awk selects the average column.
            for db in $dbs; do
                cat <<EOF | sqlite3 $db
.separator " "
select rawMin, rawAve, rawMax, rawDev from statistics where generation=99;
EOF
            done | $mm | sed -n -e'3p' | awk '{print $2'}
            if [[ "$clean" != "_no_clean" && "$clean" != "_no-cleaning" ]]; then
                echo "Load smoothing:"
                $GREP 'Best genome found shown as alleles' $logs | sed -e's/\]//g; s/\[//g' | awk '{print $(NF-2)}' | $mm
                echo "Load Z-score:"
                $GREP 'Best genome found shown as alleles' $logs | sed -e's/\]//g; s/\[//g' | awk '{print $NF}' | $mm
            fi
            # Plotting
            if [ "${show_graphs}" == "TRUE" ]; then
                python $HOME/Documents/SmartGrid/src/sg/utils/plot_fitnesses.py --ymax=$ymax --title="$title" $dbs >/dev/null &
            fi

        done
        echo ""
    done
done

if [ "${show_graphs}" == "TRUE" ]; then
    if query "Close plot windows?" "y"; then
        # Children will include the ps and grep processes, so redirect stderr.
        kill $(children) 2>/dev/null
        exit
    fi
fi

