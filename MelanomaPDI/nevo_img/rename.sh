#!/bin/bash
counter=0
for file in *jpg; do 
    [[ -f $file ]] && mv -i "$file" $((counter+1)).jpg && ((counter++))
done
