bindir="$HOME/SmartGrid/src/sg/models"
rundir="$bindir/subset_runs"
outdir="$rundir/evo_output_files"

generations=10
popsize=100

evocmd="python $bindir/load_prediction_arima.py --out-dir=$outdir --out-postfix=|postfix| --generations=$generations --pop-size=$popsize --mutation=0.2 --crossover=0.5 --mutation-sigma=10 --no-plot --elite=0 --num-trials=7 --env-replace=3 --data-seed=|dataseed| --no-cleaning --parallel --user-subset=|numberofusers|"

max_subset_size=150
runs_per_subset=10

runfile_path_base="$rundir/run-subset-"
rm ${runfile_path_base}*

num_subset_files_created=0
for (( subset=1; $subset<${max_subset_size}; subset=$subset+1 )); do
    runfile=${runfile_path_base}$subset.sh
    let num_subset_files_created=${num_subset_files_created}+1
    cat >$runfile <<EOF
#!/bin/bash
#
# These commands set up the environment for your job:
#
# Name of the job
#PBS -N subset_$subset
#
# Using IDI account when possible
#PBS -A acc-idi
#
# 
#PBS -l walltime=24:00:00
#
# Specify resources number of nodes:cores per node
#PBS -l nodes=1:ppn=12
 
# Specify queue to submit to: default, bigmem, express or default
#PBS -q default

# Send me an email if it is killed.
#PBS -m ae
## PBS -M hoversta@idi.ntnu.no
## PBS -M tidemann@idi.ntnu.no

# Set up Python virtual environment. Virtualenv doesn't play well in parallel,
# so reduce the risk of errors due to synchronized calls to 'workon' by
# sleeping for a brief random period before calling workon.
seconds=\`awk "END {srand(\$RANDOM); print int(rand()*5)}" /dev/null\`
echo "Sleeping for \$seconds second(s) before activating the virtual environment"
sleep \$seconds
echo 'Go!'
source $HOME/.bash_profile
workon smartgrid

outputfile=$rundir/output_subset_$subset.txt
evocmd="$evocmd"
echo "Command; Subset size; Data seed; RMSE" >\$outputfile
for (( run=0; \$run<${runs_per_subset}; run=\$run+1 )); do
    cmd="\`echo \$evocmd | sed -e\"s/|postfix|/subset_${subset}_/; s/|dataseed|/\$run/\; s/|numberofusers|/${subset}/"\`"
    echo "Launching \$cmd..."
    rmse=\`\$cmd 2>/dev/null | tail -n 3 | head -n 1 |awk '{print \$NF}'\`
    echo "\$cmd; $subset; \$run; \$rmse" >>\$outputfile
    echo "Done with run \$run for subset size $subset."
done
EOF
    chmod u+x $runfile
done

echo "Made ${num_subset_files_created} from 1 to ${max_subset_size} files with ${runs_per_subset} runs per subset, evolving a population of $popsize individuals over $generations generations."