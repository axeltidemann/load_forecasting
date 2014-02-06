#!/bin/bash

# Make collective scripts that run several job scripts in series. This script
# differs from the other one in that it creates 30 scripts where each script
# runs all the models, all the data sets and all the cleansing options for the
# given run number.


num_runs=30
sim_dir=$HOME/Smartgrid/simulations/isgt-env-replace-3-of-7

for ((run=0; $run<$num_runs; run++)); do
    target=multirun_$run.sh
    dir=`pwd`

    models='esn wavelet arima'
    #arima_indexed average_hourly average_daily'
    datasets='bc-data total-load ""'
    cleaners='no-cleaning'
    #subtract-weekly-pattern subtract-daily-pattern ""'
    errfuncs='sg.utils.mape_plus_one sg.utils.relative_rmse'
    cat <<EOF >$target
#!/bin/bash
#
# These commands set up the environment for your job:
#
# Name of the job
#PBS -N $target
#
# Using IDI account when possible
#PBS -A acc-idi
#
# 
#PBS -l walltime=20:00:00
#
# Specify resources number of nodes:cores per node
#PBS -l nodes=1:ppn=12
 
# Specify queue to submit to: default, bigmem, express or default
#PBS -q default

# Send me an email if it is killed.
#PBS -m ae
# PBS -M hoversta@idi.ntnu.no

cd $dir
for model in $models; do
  for data in $datasets; do
    for cleaning in $cleaners; do
      for errfunc in $errfuncs; do
        name=evo_\${model}_\${data}_\${cleaning}_\${errfunc}_run_$run
        job=jobscript_\$name.sh;
        if [ -n "\$data" ]; then
          tdata=_\$data
        else
          tdata=""
        fi
        if [ -n "\$cleaning" ]; then
          tcleaning=_\$cleaning
        else
          tcleaning=""
        fi
        tcleaning=\`echo \$tcleaning | sed -e's/no-cleaning/no_clean/'\`
        if [ -n "\$errfunc" ]; then
          terrfunc=_\$errfunc
        else
          terrfunc=""
        fi
        target=$sim_dir/output_\${model}_run_$run\${tdata}\${tcleaning}\${terrfunc}.txt
        if [ -f \$target ]; then
          echo "   Skipping job, output already exists: \$target."
        else
          echo "Running \$target."
          ./\$job
        fi
      done
    done
  done
done
EOF
    chmod u+x $target
done

