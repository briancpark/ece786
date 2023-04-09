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

__global__ void device_to_device_memcpy(float* a, float* b, int N) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid > N)
        return;

    b[tid] = a[tid];
    __syncthreads();
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
    __syncthreads();
    __syncwarp();
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

    int threadsPerBlock = 256;
    int blocksPerGrid = (a.size() + threadsPerBlock - 1) / threadsPerBlock;

    quantum_simulation_gpu<<<blocksPerGrid, threadsPerBlock>>>(U_0_gpu, a_gpu, output_gpu, qubit_0,
                                                               a.size());
    device_to_device_memcpy<<<blocksPerGrid, threadsPerBlock>>>(output_gpu, a_gpu, a.size());
    quantum_simulation_gpu<<<blocksPerGrid, threadsPerBlock>>>(U_1_gpu, a_gpu, output_gpu, qubit_1,
                                                               a.size());
    device_to_device_memcpy<<<blocksPerGrid, threadsPerBlock>>>(output_gpu, a_gpu, a.size());
    quantum_simulation_gpu<<<blocksPerGrid, threadsPerBlock>>>(U_2_gpu, a_gpu, output_gpu, qubit_2,
                                                               a.size());
    device_to_device_memcpy<<<blocksPerGrid, threadsPerBlock>>>(output_gpu, a_gpu, a.size());
    quantum_simulation_gpu<<<blocksPerGrid, threadsPerBlock>>>(U_3_gpu, a_gpu, output_gpu, qubit_3,
                                                               a.size());
    device_to_device_memcpy<<<blocksPerGrid, threadsPerBlock>>>(output_gpu, a_gpu, a.size());
    quantum_simulation_gpu<<<blocksPerGrid, threadsPerBlock>>>(U_4_gpu, a_gpu, output_gpu, qubit_4,
                                                               a.size());
    device_to_device_memcpy<<<blocksPerGrid, threadsPerBlock>>>(output_gpu, a_gpu, a.size());
    quantum_simulation_gpu<<<blocksPerGrid, threadsPerBlock>>>(U_5_gpu, a_gpu, output_gpu, qubit_5,
                                                               a.size());

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
    free(U_0);
    free(U_1);
    free(U_2);
    free(U_3);
    free(U_4);
    free(U_5);

    cudaFree(a_gpu);
    cudaFree(output_gpu);
    free(output);
    return 0;
}