#!/bin/bash

# Perform a command on each file in gs2.txt
# Example usage:
# ./gs2-do.sh sed -n -e'/Istad Nett/p' 
cat gs2.txt | while read path; do "$@" "$path"; done
