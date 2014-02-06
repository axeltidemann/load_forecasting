#!/bin/bash
#
# Vilje settings:
##PBS -A nn9270k
##PBS -l select=7:ncpus=32:mpiprocs=16:ompthreads=1

# Kongull settings:
# Specify resources as 'number_of_nodes:cores_per_node:condition', where
# "condition" is intel, amd, default or bigmem. default and bigmem both run on
# AMD cores, but the bigmem nodes have slower memory. There are 12 Intel nodes,
# these are only available to the express and optimist queues.
#PBS -l nodes=3:ppn=12:default
#PBS -q default
#PBS -A acc-idi

# Common settings:
#PBS -N xxx
#PBS -l walltime=10:00:00

# Send me an email if it is killed.
#PBS -m ae
## PBS -M hoversta@idi.ntnu.no
## PBS -M tidemann@idi.ntnu.no

# Set up Python virtual environment. Virtualenv doesn't play well in parallel,
# so reduce the risk of errors due to synchronized calls to 'workon' by
# sleeping for a brief random period before calling workon.
#   This doesn't work when using MPI, so there is really no point to it any more.
# seconds=`awk "END {srand($RANDOM); print int(rand()*5)}" /dev/null`
# echo "Sleeping for $seconds second(s) before activating the virtual environment"
# sleep $seconds
# echo 'Go!'

sg_dir=$HOME/smartgrid
out_dir=$sg_dir/simulations/convergence.with.bc-weather
run_dir=$sg_dir/src/sg/models

cd $run_dir

dataset=xxx
model=xxx
model_path=$run_dir/load_prediction_$model.py
generations=100
pop_size=100
mutation=0.2
crossover=0.3
data_seed=xxx
elite=1
preproc=xxx
no_cleaning=xxx
run_num=xxx
errfunc=xxx
num_trials=xxx
env_replace=xxx

echo "out dir is $out_dir"
test -d $out_dir || mkdir -p $out_dir

postfix=${model}_run_${run_num}
if [ -n "$dataset" ]; then
    postfix=${postfix}_${dataset}-noholidays
    dataset="--$dataset"
fi
if [ -n "$preproc" ]; then
    postfix=${postfix}_${preproc}
    preproc="--$preproc"
fi
if [ -n "$no_cleaning" ]; then
    postfix=${postfix}_${no_cleaning}
    no_cleaning="--$no_cleaning"
fi
postfix=${postfix}_${num_trials}_${env_replace}
if [ -n "$errfunc" ]; then
    postfix=${postfix}_${errfunc}
    errfunc="--error-func=$errfunc"
fi

stdout_path=$out_dir/output_$postfix.txt

echo "Launching $model_path, output to $stdout_path, at `date`"

export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=${MKL_NUM_THREADS}

#mpiexec_mpt omplace -nt ${MKL_NUM_THREADS} sg-python $model_path \
mpirun -x MKL_NUM_THREADS -hostfile $PBS_NODEFILE sg-python $model_path \
    --out-dir=$out_dir \
    --out-postfix=$postfix \
    --generations=$generations \
    --pop-size=$pop_size \
    --mutation=$mutation \
    --crossover=$crossover \
    --no-plot \
    --elite=$elite \
    --num-trials=${num_trials} \
    --env-replace=${env_replace} \
    --print-pop \
    --remove-holidays \
    $data_seed \
    $dataset \
    $preproc \
    $no_cleaning \
    $errfunc \
    --MPI \
    >$stdout_path 2>&1

bzip2 $stdout_path

echo "Evolution completed for $model_path, output to $stdout_path, at `date`"
