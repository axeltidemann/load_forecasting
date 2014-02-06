#!/bin/bash

models=arima
datasets="total-load bc-data"
match="Error on test phase for best genome found"
for model in $models; do
    for data in $datasets; do
        for (( r=0; $r<30; r++ )); do
            base=output_${model}_run_${r}_${data}-noholidays_;
            for prep in "" subtract-daily-pattern_ subtract-weekly-pattern_; do
                f=${base}${prep}100_0.txt;
                have_txt=false
                fin_txt=false
                have_bz2=false
                fin_bz2=false
                if [[ -e "$f" ]]; then
                    have_txt=true
                    if [[ -n "`grep \"$match\" $f`" ]]; then
                        fin_txt=true
                    fi
                fi
                if [[ -e "$f.bz2" ]]; then
                    have_bz2=true
                    if [[ -n "`bzgrep \"$match\" $f.bz2`" ]]; then
                        fin_bz2=true
                    fi
                fi
                if [[ ${have_txt} == true && ${have_bz2} == true ]]; then
                    if [[ ${fin_txt} == true && ${fin_bz2} == true ]]; then
                        echo "Duplicate, both finished: $f/.bz2"
                    elif [[ ${fin_txt} == true ]]; then
                        echo "Duplicate, only .txt finished: $f/.bz2"
                    elif [[ ${fin_bz2} == true ]]; then
                        echo "Duplicate, only .bz2 finished: $f/.bz2"
                    else
                        echo "Duplicate, both incomplete: $f/.bz2"
                    fi
                elif [[ ${have_txt} == true ]]; then
                    if [[ ${fin_txt} == true ]]; then
                        echo "Finished: $f"
                    else
                        echo "Incomplete: $f (no .bz2)"
                    fi
                elif [[ ${have_bz2} == true ]]; then
                    if [[ ${fin_bz2} == true ]]; then
                        echo "Finished: $f.bz2"
                    else
                        echo "Incomplete: $f.bz2 (no .txt)"
                    fi
                else
                    echo "Both missing: $f/.bz2"
                fi
            done;
        done
    done
done
