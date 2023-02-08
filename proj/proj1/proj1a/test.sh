#!/bin/bash

if [ ! -f quamsimV1 ]; then
    echo "Executable quamsimV1 does not exist"
    exit 1
fi

for file in input/input*.txt
do
    echo "Testing $file"
    ./quamsimV1 $file >> $file.out
    output_file=${file//input/output}
    diff $file.out $output_file
    rm $file.out
done
