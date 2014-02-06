#!/bin/bash

# Make collective scripts that run several job scripts in series

bases=`ls -1 jobscript*CBR* | sed -e's/run_.*//' |sort |uniq`
for base in $bases; do
    target=multi_$base.sh
    dir=`pwd`
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
#PBS -l walltime=168:00:00
#
# Specify resources number of nodes:cores per node
#PBS -l nodes=1:ppn=12
 
# Specify queue to submit to: default, bigmem, express or default
#PBS -q default

# Send me an email if it is killed.
#PBS -m ae
# PBS -M hoversta@idi.ntnu.no

cd $dir
for job in $base*; do
  ./\$job
  rm $HOME/Smartgrid/src/sg/models/\(*\).dat
  rm $HOME/Smartgrid/src/sg/models/\(*\).idx
done
EOF
    chmod u+x $target
done

