#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <string>
#include <vector>

using namespace std;

__global__ void quantum_simulation_gpu(float* U, float* a, float* output, int qubit, int N) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    size_t qid = 1 << qubit;

    if (tid > N)
        return;

    if (tid & qid)
        output[tid] = U[2] * a[tid - qid] + U[3] * a[tid];
    else
        output[tid] = U[0] * a[tid] + U[1] * a[tid + qid];
    __syncthreads();
}

int main(int argc, char** argv) {
    // Parse the command line arguments
    if (argc != 2) {
        fprintf(stderr, "Usage: %s input.txt\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    // Read the input file
    ifstream input_file;
    input_file.open(argv[1]);

    // Setup variables to store matrix and vector
    float *U, *input, *output;
    vector<float> a;
    size_t qubit;
    string line;

    cudaMallocManaged(&U, 4 * sizeof(float));

    for (int i = 0; i < 4; i++) {
        input_file >> U[i];
    }

    // Read in the vector until we hit an empty line
    std::getline(input_file, line);
    std::getline(input_file, line);

    while (std::getline(input_file, line) && !line.empty()) {
        a.push_back(stof(line));
    }

    cudaMallocManaged(&input, a.size() * sizeof(float));
    for (int i = 0; i < a.size(); i++) {
        input[i] = a[i];
    }

    // Read in the qubit
    input_file >> qubit;

    cudaMallocManaged(&output, a.size() * sizeof(float));

    cudaDeviceSynchronize();
    int threadsPerBlock = 256;
    int blocksPerGrid = (a.size() + threadsPerBlock - 1) / threadsPerBlock;
#ifdef BENCHMARK
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
#endif
    quantum_simulation_gpu<<<blocksPerGrid, threadsPerBlock>>>(U, input, output, qubit, a.size());
#ifdef BENCHMARK
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "Time taken: " << milliseconds << " ms" << endl;
#endif

    cudaDeviceSynchronize();
    // Print the output vector
    for (int i = 0; i < a.size(); i++) {
        printf("%.3f\n", output[i]);
    }

    cudaFree(U);
    cudaFree(input);
    cudaFree(output);

    return 0;
}