/*
Develop two versions of the host code, one uses cudaMalloc and cudaMemcpy to move data explicitly
and the other uses cudaMallocManaged to leverage unified virtual memory to move the data. Report the
timing results for both versions.

Input format:

- Your implementation should not be based on specific sizes of the input vector although you can
assume that the vector length is smaller than 230.

- The submitted code will be tested using randomly generated data inputs.

- Example input file:

input file format

- The corresponding output file:

output file format

- The element type should be assumed as the single-precision float.

- In the input file, the first 2x2 matrix represents a single-qubit quantum gate, and the second
128x1 matrix (i.e., vector) represents a 7-qubit (from qubit 0 to qubit 6) quantum state. They are
separated by a blank line. Each matrix row is in a separate line (i.e. ends with a linefeed "\n").
Matrix elements in the same row are separated by a single space. And the number in the last line of
the input  file represents which qubit the single-qubit gate is applied on.

- In the corresponding output format file, the vector represents the 7-qubit quantum output state
after applying the single-qubit gate on qubit 2.



The Expected Output:

- You should output a vector of length N = 2n and print it to the console screen (using stdio).

- Every row should be in a separate line (i.e. it ends with a newline "\n"). The values should be
printed with 3 decimal points precision (not more nor less).
- Other than the output vector, NOTHING else should be printed. Do not print "done". Do not print
the execution time. Do not print anything other than the output vector.

- Note that we will use diff to check the output. That means you need to exactly match the output.
- For grading, your program will be compiled and run with command like "./quamsim ./input.txt"


*/

#include <iostream>
#include <stdio.h>
// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

using namespace std;

void read_matrix(FILE* fp, float* matrix) {
    for (int i = 0; i < 4; i++) {
        fscanf(fp, "%f", &matrix[i]);
    }
}

void read_vector(FILE* fp, float* vector) {
    for (int i = 0; i < 128; i++) {
        fscanf(fp, "%f", &vector[i]);
    }
}

void read_qubit(FILE* fp, size_t* qubit) { fscanf(fp, "%zu", qubit); }

int main(int argc, char** argv) {
    // Parse the command line arguments
    if (argc != 2) {
        fprintf(stderr, "Usage: %s input.txt\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    // Read the input file
    FILE* fp = fopen(argv[1], "r");
    if (fp == NULL) {
        fprintf(stderr, "Failed to open file %s\n", argv[1]);
        exit(EXIT_FAILURE);
    }

    // Read the first matrix
    // TODO: change to GPU later
    float* matrix = (float*)malloc(4 * sizeof(float));
    float* vector = (float*)malloc(128 * sizeof(float));
    float* output = (float*)malloc(128 * sizeof(float));
    size_t qubit;

    // for (int i = 0; i < 4; i++) {
    //     cout << matrix1[i] << " ";
    // }
    read_matrix(fp, matrix);
    read_vector(fp, vector);
    read_qubit(fp, &qubit);

    // cout << qubit << endl;

    // Perform quantum simulation on qubit
    for (size_t i = 0; i < 128; i++) {
        if ((i & (1 << qubit)) == 0) {
            output[i] = matrix[0] * vector[i] + matrix[1] * vector[i + (1 << qubit)];
        } else {
            output[i] = matrix[2] * vector[i - (1 << qubit)] + matrix[3] * vector[i];
        }
    }

    // Print the output vector
    for (int i = 0; i < 128; i++) {
        printf("%.3f\n", output[i]);
    }
    return 0;
}