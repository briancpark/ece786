#!/bin/bash

make clean
make BENCHMARK=1 -j

for file in input/input*.txt
do
    echo "Benchmarking $file"
    ./quamsimV1 $file >> $file.out
    output_file=${file//input/output}
    diff $file.out $output_file
    rm $file.out
done

for file in input/input*.txt
do
    echo "Benchmarking $file"
    ./quamsimV2 $file >> $file.out
    output_file=${file//input/output}
    diff $file.out $output_file
    rm $file.out
done
