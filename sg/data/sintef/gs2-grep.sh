#!/bin/bash

# Perform a grep on each file in gs2.txt

cat gs2.txt | while read path; do grep $@ "$path"; done
