#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <string>
#include <vector>

using namespace std;

#define FRAGMENT_SIZE     (1 << 6) // 2^6
#define THREADS_PER_BLOCK (1 << 5) // 2^5

__global__ void quantum_simulation_gpu(float* U_0, float* U_1, float* U_2, float* U_3, float* U_4,
                                       float* U_5, float* a, float* output, size_t N,
                                       size_t* auxillary_array, size_t* offsets) {

    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid >= N / 2) {
        return;
    }

    __shared__ float a_shared[FRAGMENT_SIZE];

    // Load the fragment from global memory to shared memory
    size_t offset = offsets[blockIdx.x];

    size_t idx = threadIdx.x * 2;

    a_shared[idx] = a[auxillary_array[idx] + offset];
    a_shared[idx + 1] = a[auxillary_array[idx + 1] + offset];

    float* Us[6] = {U_0, U_1, U_2, U_3, U_4, U_5};
    __syncthreads();
    __syncwarp();

    register float x0, x1;
    for (size_t gate = 0; gate < 6; gate++) {
        float* U = Us[gate];
        size_t gate_offset = 1 << gate;

        if ((idx & gate_offset) == 0) {
            x0 = a_shared[idx];
            x1 = a_shared[idx + gate_offset];

            a_shared[idx] = U[0] * x0 + U[1] * x1;
            a_shared[idx + gate_offset] = U[2] * x0 + U[3] * x1;
        } else {
            x0 = a_shared[idx - gate_offset + 1];
            x1 = a_shared[idx + 1];

            a_shared[idx - gate_offset + 1] = U[0] * x0 + U[1] * x1;
            a_shared[idx + 1] = U[2] * x0 + U[3] * x1;
        }
        __syncthreads();
        __syncwarp();
    }

    // Store the fragment from shared memory to global memory
    output[auxillary_array[idx] + offset] = a_shared[idx];
    output[auxillary_array[idx + 1] + offset] = a_shared[idx + 1];
    return;
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
    float* U_0 = (float*)malloc(4 * sizeof(float));
    float* U_1 = (float*)malloc(4 * sizeof(float));
    float* U_2 = (float*)malloc(4 * sizeof(float));
    float* U_3 = (float*)malloc(4 * sizeof(float));
    float* U_4 = (float*)malloc(4 * sizeof(float));
    float* U_5 = (float*)malloc(4 * sizeof(float));
    vector<float> a;
    size_t qubit_0, qubit_1, qubit_2, qubit_3, qubit_4, qubit_5;

    // Create a list of Us
    vector<float*> Us = {U_0, U_1, U_2, U_3, U_4, U_5};

    // For loop to read in the U matrices
    for (int i = 0; i < 6; i++) {
        float* U = Us[i];
        for (int i = 0; i < 4; i++) {
            input_file >> U[i];
        }
    }

    // Read in the vector until we hit an empty line
    string line;
    getline(input_file, line);
    getline(input_file, line);

    while (getline(input_file, line) && !line.empty()) {
        a.push_back(stof(line));
    }

    // Read in the qubit
    input_file >> qubit_0;
    input_file >> qubit_1;
    input_file >> qubit_2;
    input_file >> qubit_3;
    input_file >> qubit_4;
    input_file >> qubit_5;

    float* output = (float*)malloc(a.size() * sizeof(float));

    // Copy memory to GPU
    float* a_gpu;
    cudaMalloc(&a_gpu, a.size() * sizeof(float));
    cudaMemcpy(a_gpu, a.data(), a.size() * sizeof(float), cudaMemcpyHostToDevice);

    float* output_gpu;
    cudaMalloc(&output_gpu, a.size() * sizeof(float));

    float *U_0_gpu, *U_1_gpu, *U_2_gpu, *U_3_gpu, *U_4_gpu, *U_5_gpu;
    cudaMalloc(&U_0_gpu, 4 * sizeof(float));
    cudaMalloc(&U_1_gpu, 4 * sizeof(float));
    cudaMalloc(&U_2_gpu, 4 * sizeof(float));
    cudaMalloc(&U_3_gpu, 4 * sizeof(float));
    cudaMalloc(&U_4_gpu, 4 * sizeof(float));
    cudaMalloc(&U_5_gpu, 4 * sizeof(float));

    cudaMemcpy(U_0_gpu, U_0, 4 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(U_1_gpu, U_1, 4 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(U_2_gpu, U_2, 4 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(U_3_gpu, U_3, 4 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(U_4_gpu, U_4, 4 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(U_5_gpu, U_5, 4 * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = THREADS_PER_BLOCK;
    int blocksPerGrid = (a.size() + threadsPerBlock - 1) / threadsPerBlock;

    size_t* auxillary_array = (size_t*)malloc(FRAGMENT_SIZE * sizeof(size_t));
    size_t* auxillary_array_gpu;
    cudaMalloc(&auxillary_array_gpu, FRAGMENT_SIZE * sizeof(size_t));

    size_t qubits[6] = {qubit_0, qubit_1, qubit_2, qubit_3, qubit_4, qubit_5};

    for (int i = 0; i < FRAGMENT_SIZE; i++) {
        int sum = 0;
        for (int j = 0; j < 6; j++) {
            if (i & (1 << j)) {
                sum += pow(2, qubits[j]);
            }
        }
        auxillary_array[i] = sum;
    }

    cudaMemcpy(auxillary_array_gpu, auxillary_array, FRAGMENT_SIZE * sizeof(size_t),
               cudaMemcpyHostToDevice);

    // length of offset is determined by log2(a.size()) - 6
    size_t offsets_len = (size_t)log2(a.size()) - 6;
    offsets_len = pow(2, offsets_len);
    size_t* offsets = (size_t*)calloc(offsets_len, sizeof(size_t));
    size_t* offsets_gpu;
    cudaMalloc(&offsets_gpu, offsets_len * sizeof(size_t));

    size_t bitmask = 0;
    // set the bitmask to 1 at the qubit indices
    for (int i = 0; i < 6; i++) {
        bitmask |= (1 << qubits[i]);
    }

    // Determine the bits that are set to 0 in the bitmask and store them in the offsets array
    int bitid = 0;
    for (int i = 0; i < a.size(); i++) {
        if ((i & bitmask) == 0) {
            offsets[bitid] = i;
            bitid++;
        }
    }

    cudaMemcpy(offsets_gpu, offsets, offsets_len * sizeof(size_t), cudaMemcpyHostToDevice);

    quantum_simulation_gpu<<<blocksPerGrid, threadsPerBlock>>>(
        U_0_gpu, U_1_gpu, U_2_gpu, U_3_gpu, U_4_gpu, U_5_gpu, a_gpu, output_gpu, a.size(),
        auxillary_array_gpu, offsets_gpu);

    cudaDeviceSynchronize();
    cudaMemcpy(output, output_gpu, a.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // Print the output vector
    for (int i = 0; i < a.size(); i++) {
        printf("%.3f\n", output[i]);
    }

    cudaFree(U_0_gpu);
    cudaFree(U_1_gpu);
    cudaFree(U_2_gpu);
    cudaFree(U_3_gpu);
    cudaFree(U_4_gpu);
    cudaFree(U_5_gpu);
    cudaFree(a_gpu);
    cudaFree(output_gpu);
    cudaFree(offsets_gpu);
    cudaFree(auxillary_array_gpu);
    free(U_0);
    free(U_1);
    free(U_2);
    free(U_3);
    free(U_4);
    free(U_5);
    free(output);
    free(offsets);
    free(auxillary_array);
    return 0;
}
