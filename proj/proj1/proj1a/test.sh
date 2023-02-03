#!/bin/bash

if [ ! -f quamsim ]; then
    echo "Executable quamsim does not exist"
    exit 1
fi


# Iterate through all of the input files and execute the program
# Then compare the outputs to the ones in output directory
# Here's an example use case
# quamsim input/input_qc4_q0.txt >> input_qc4_q0.out
# diff input_qc4_q0.out output/output_qc4_q0.out
# Note that the formats of inputs are in input/input*.txt and outputs are in output/output*.txt

# Iterate through all of the input files and execute the program
# Then compare the outputs to the ones in output directory

for file in input/input*.txt
do
    ./quamsim $file >> $file.out
    # replace name input with output
    output_file=$(echo $file | sed 's/input/output/')
    diff $file.out $output_file
    rm $file.out
done
