#!/bin/bash

num_runs=30
env_replace=0
num_trials=100
errfunc=""
for ((i=0; $i<$num_runs; i++)); do
    for model in arima ar24 esn wavelet dshw; do
        for data in "bc-data" "total-load"; do
            for preproc in  "" "subtract-weekly-pattern" "subtract-daily-pattern"; do
                name=evo_${model}_${data}_${cleaning}_${preproc}_${num_trials}-${env_replace}_${errfunc}_run_$i
                target=jobscript_$name.sh;
                sed -e"s/run_num=xxx/run_num=$i/;
                       s/model=xxx/model=$model/;
                       s/data_seed=xxx/data_seed=\"--data-seed=$i\"/;
                       s/load_prediction_xxx.py/load_prediction_$model.py/; 
                       s/dataset=xxx/dataset=$data/;
                       s/preproc=xxx/preproc=$preproc/;
                       s/no_cleaning=xxx/no_cleaning=$cleaning/;
                       s/errfunc=xxx/errfunc=$errfunc/;
                       s/env_replace=xxx/env_replace=${env_replace}/;
                       s/num_trials=xxx/num_trials=${num_trials}/;
                       s/#PBS -N xxx/#PBS -N ev${model}${i}/;
                       " template.sh >$target
                chmod u+x $target
                echo "Created $target"
            done
        done
    done
done


