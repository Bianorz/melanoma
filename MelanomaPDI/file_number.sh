#!/bin/bash

cd $1
count=$( ls -l | grep -v ^d | grep -v ^t | wc -l)
exit $count
