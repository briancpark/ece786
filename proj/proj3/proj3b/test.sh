#!/bin/bash

make clean
make -j

# for file in input/input*.txt
# do
#     echo "Testing $file"
#     ./quamsimV1 $file >> $file.out
#     output_file=${file//input/output}
#     diff $file.out $output_file
#     rm $file.out
# done


for file in input/input*.txt
do
    echo "Testing $file"
    ./quamsimV2 $file >> $file.out
    output_file=${file//input/output}
    python3 output_checker.py $file.out $output_file
    rm $file.out
done