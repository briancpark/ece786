#!/bin/bash

make clean
make -j

for file in input/input*.txt
do
    echo "Benchmarking $file"
    ./quamsimV1_benchmark $file >> $file.out
    output_file=${file//input/output}
    diff $file.out $output_file
    rm $file.out
done

for file in input/input*.txt
do
    echo "Benchmarking $file"
    ./quamsimV2_benchmark $file >> $file.out
    output_file=${file//input/output}
    diff $file.out $output_file
    rm $file.out
done
