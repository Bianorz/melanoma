#!/bin/bash
counter=0
mkdir temp_folder
cd $1
for file in *jpg; do 
    [[ -f $file ]] && mv -f "$file" ../temp_folder/$((counter+1)).jpg && ((counter++))
done
cd ..
rm -rf $1/*
cp -a temp_folder/. $1/
rm -r temp_folder

  
