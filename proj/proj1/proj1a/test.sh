#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

make clean
make -j

# Function to test and check for differences
test_and_diff() {
    local executable=$1
    for file in input/input*.txt; do
        echo "Testing $file with $executable"
        ./$executable $file >> "$file.out"
        output_file=${file//input/output}
        if ! diff "$file.out" "$output_file"; then
            echo "Error: Output for $file with $executable differs from expected output."
            exit 1
        fi
        rm "$file.out"
    done
}

# Test both executables
test_and_diff "quamsimV1"
test_and_diff "quamsimV2"

