#!/bin/bash

if [ ! -f quamsim ]; then
    echo "Executable quamsim does not exist"
    exit 1
fi

for file in input/input*.txt
do
    echo "Testing $file"
    ./quamsim $file >> $file.out
    output_file=${file//input/output}
    diff $file.out $output_file
    rm $file.out
done
