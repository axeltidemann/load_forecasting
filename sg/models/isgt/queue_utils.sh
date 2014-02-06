#!/bin/bash

. $HOME/local/bin/query.sh

jobs_user() {
    user=$1
    qstat |grep $user
}

queued_job_numbers_user() {
    jobs_user $1 | grep Q | awk -F. '{print $1}'
}

running_jobs_user() {
    jobs_user $1 | grep R | awk -F. '{print $1}'
}

running_job_numbers_user() {
    running_jobs_user $1 | awk -F. '{print $1}'
}

_show_job_info() {
    jobs=$1
    for job in $jobs; do
        echo ""
        qstat $job | tail -n 1
        qstat -f $job | grep Job_Name
        qstat -f $job | grep submit_args
    done
}

show_queued_jobs_user() {
    user=$1
    _show_job_info "`queued_job_numbers_user $user`"
}

show_running_jobs_user() {
    user=$1
    _show_job_info "`running_job_numbers_user $user`"
}

show_queued_jobs() {
    jobs="`queued_job_numbers_user $USER`"
    _show_job_info "$jobs"
    echo "`echo $jobs | awk '{print NF}'` jobs in total."
}

show_running_jobs() {
    jobs="`running_job_numbers_user $USER`"
    _show_job_info "$jobs"
    echo "`echo $jobs | awk '{print NF}'` jobs in total."
}

_delete_jobs() {
    pattern=$1
    jobs=$2
    delete_list=""
    for job in $jobs; do
        name=`qstat -f $job |grep Job_Name`
        if [ -n "`echo $name | grep \"$pattern\"`" ]; then 
            echo "Will delete $job $name."
            delete_list="$job $delete_list"
        else
            echo "Will not delete $job $name."
        fi;
    done
    if query "Delete jobs?" "n"; then
        echo qdel $delete_list
        qdel $delete_list
    fi
}

# Delete a specific selection of jobs
delete_queued_jobs_with_name_match() {
    pattern="$1"
    _delete_jobs $pattern "`queued_job_numbers_user $USER`"
}

delete_running_jobs_with_name_match() {
    pattern="$1"
    _delete_jobs $pattern "`running_job_numbers_user $USER`"
}

switch_to_default_queue() {
    sed -i -e's/-q optimist/-q default/; s/-A freecycle/-A acc-idi/' $@
}

switch_to_optimist_queue() {
    sed -i -e's/-q default/-q optimist/; s/-A acc-idi/-A freecycle/' $@
}

