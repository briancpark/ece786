#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <string>
#include <vector>

using namespace std;

void quantum_simulation_cpu(float* U, float* a, float* output, size_t qubit, size_t N) {
    // Perform quantum simulation on qubit
    for (size_t i = 0; i < N; i++) {
        if ((i & (1 << qubit)) == 0) {
            output[i] = U[0] * a[i] + U[1] * a[i + (1 << qubit)];
        } else {
            output[i] = U[2] * a[i - (1 << qubit)] + U[3] * a[i];
        }
    }
}
__global__ void quantum_simulation_gpu(const float* U, const float* a, float* output, int qubit,
                                       int N) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    register size_t qid = 1 << qubit;

    if (tid > N)
        return;

    if (tid & qid)
        output[tid] = U[2] * a[tid - qid] + U[3] * a[tid];
    else
        output[tid] = U[0] * a[tid] + U[1] * a[tid + qid];
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
    float* U = (float*)malloc(4 * sizeof(float));
    vector<float> a;
    size_t qubit;

    for (int i = 0; i < 4; i++) {
        input_file >> U[i];
    }

    // Read in the vector until we hit an empty line
    string line;
    getline(input_file, line);
    getline(input_file, line);

    while (getline(input_file, line) && !line.empty()) {
        a.push_back(stof(line));
    }

    // Read in the qubit
    input_file >> qubit;

    float* output = (float*)malloc(a.size() * sizeof(float));

// Perform quantum simulation on qubit
// quantum_simulation_cpu(U, a.data(), output, qubit, a.size());
#ifdef BENCHMARK
    size_t FLOPs = 3 * a.size();
    cout << "FLOPs: " << FLOPs << endl;
#endif
    // Copy memory to GPU
    float* U_gpu;
    float* a_gpu;
    float* output_gpu;
    cudaMalloc(&U_gpu, 4 * sizeof(float));
    cudaMalloc(&a_gpu, a.size() * sizeof(float));
    cudaMalloc(&output_gpu, a.size() * sizeof(float));
    cudaMemcpy(U_gpu, U, 4 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(a_gpu, a.data(), a.size() * sizeof(float), cudaMemcpyHostToDevice);

    // quantum_simulation_gpu(U, a.data(), output, qubit, a.size());
    int threadsPerBlock = 256;
    int blocksPerGrid = (a.size() + threadsPerBlock - 1) / threadsPerBlock;
    // printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

#ifdef BENCHMARK
    cudaEvent_t start, stop;
    float milliseconds = 0;
    double average = 0;
    for (int i = 0; i < 100; i++) {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
#endif
        quantum_simulation_gpu<<<blocksPerGrid, threadsPerBlock>>>(U_gpu, a_gpu, output_gpu, qubit,
                                                                   a.size());
#ifdef BENCHMARK
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&milliseconds, start, stop);
        average += milliseconds;
    }
    cout << "Time taken on average: " << average / 100 << " ms" << endl;
    cout << "GFLOPs: " << FLOPs / (average / 100) / 1000 << endl;
#endif

    cudaMemcpy(output, output_gpu, a.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    // Print the output vector
    for (int i = 0; i < a.size(); i++) {
        printf("%.3f\n", output[i]);
    }

    cudaFree(U_gpu);
    cudaFree(a_gpu);
    cudaFree(output_gpu);
    free(U);
    free(output);
    return 0;
}