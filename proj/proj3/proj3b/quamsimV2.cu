#include <bitset>
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
                                       float* U_5, float* a, float* output, size_t qubit0,
                                       size_t qubit1, size_t qubit2, size_t qubit3, size_t qubit4,
                                       size_t qubit5, size_t N, size_t* auxillary_array,
                                       size_t* offsets) {

    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int num_fragments = N / FRAGMENT_SIZE;

    if (tid >= N / 2) {
        return;
    }

    __shared__ float a_shared[FRAGMENT_SIZE];
    __shared__ float output_shared[FRAGMENT_SIZE];

    printf("num_fragments: %d\n", num_fragments);

    // Load the fragment from global memory to shared memory

    // todo; eneumeratie
    size_t offset = 0;
    if (blockIdx.x) {
        offset = 1 << offsets[blockIdx.x - 1];
    }
    printf("blockIdx.x: %d\n", blockIdx.x);
    printf("offset: %d\n", offset);

    for (int i = threadIdx.x * 2; i < FRAGMENT_SIZE; i += blockDim.x * 2) {
        a_shared[i] = a[i + offset];
        a_shared[i + 1] = a[i + 1 + offset];
    }
    __syncthreads();
    __syncwarp();
    if (tid % 32 == 0) {
        for (int i = 0; i < FRAGMENT_SIZE; i++) {
            printf("a_shared[%d] = %f\n", i, a_shared[i]);
        }
    }
    __syncthreads();
    __syncwarp();

    __syncthreads();
    size_t qid0 = 1 << qubit0;
    size_t qid1 = 1 << qubit1;
    size_t qid2 = 1 << qubit2;
    size_t qid3 = 1 << qubit3;
    size_t qid4 = 1 << qubit4;
    size_t qid5 = 1 << qubit5;

    // example: 0 2 3 4 5 6
    // binary  0 4 8 16 32 64

    __syncthreads();
    __syncwarp();
    float* Us[6] = {U_0, U_1, U_2, U_3, U_4, U_5};
    size_t qids[6] = {qid0, qid1, qid2, qid3, qid4, qid5};
    int i = threadIdx.x * 2;

    __syncthreads();
    __syncwarp();
    for (int gate = 0; gate < 6; gate++) {
        float* U = Us[gate];

        float x0 = a_shared[i];
        float x1 = a_shared[i + 1];
        __syncthreads();
        __syncwarp();
        a_shared[i] = U[0] * x0 + U[2] * x1;
        a_shared[i + 1] = U[1] * x0 + U[3] * x1;

        __syncthreads();
        __syncwarp();
    }

    __syncthreads();
    __syncwarp();
    // Store the fragment from shared memory to global memory
    output[tid * 2 + offset] = a_shared[i];
    output[(tid * 2) + 1 + offset] = a_shared[i + 1];

    __syncthreads();
    __syncwarp();
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

    int threadsPerBlock = 32;
    int blocksPerGrid = (a.size() + threadsPerBlock - 1) / threadsPerBlock;

    size_t* auxillary_array = (size_t*)malloc(64 * sizeof(size_t));
    size_t* auxillary_array_gpu;

    size_t qubits[6] = {qubit_0, qubit_1, qubit_2, qubit_3, qubit_4, qubit_5};

    // enumerate all possible combinations of the qubits indices and store them in the auxillary
    // array for example, if the qubits are 0, 2, 4, then the possible combinations are: 0, 1,
    // 4, 5, 16, 17, 20, 21 in decimal 0, 1, 100, 101, 10000, 10001, 10100, 10101 in binary
    for (int i = 0; i < 64; i++) {
        int sum = 0;
        for (int j = 0; j < 6; j++) {
            if (i & (1 << j)) {
                sum += pow(2, qubits[j]);
            }
        }
        auxillary_array[i] = sum;
    }

    cudaMalloc(&auxillary_array_gpu, 64 * sizeof(size_t));
    cudaMemcpy(auxillary_array_gpu, auxillary_array, 64 * sizeof(size_t), cudaMemcpyHostToDevice);

    // print the contents of the auxillary array in binary
    for (int i = 0; i < 64; i++) {
        // binary
        cout << bitset<6>(auxillary_array[i]) << endl;
    }

    // length of offset is determined by log2(a.size()) - 6
    size_t offsets_len = (size_t)log2(a.size()) - 6;
    size_t* offsets = (size_t*)malloc(offsets_len * sizeof(size_t));
    size_t* offsets_gpu;
    cudaMalloc(&offsets_gpu, offsets_len * sizeof(size_t));
    cout << "offsets_len: " << offsets_len << endl;

    size_t bitmask = 0;
    // set the bitmask to 1 at the qubit indices
    for (int i = 0; i < 6; i++) {
        bitmask |= (1 << qubits[i]);
        cout << "bitmask: " << bitset<16>(bitmask) << endl;
    }

    // Determine the bits that are set to 0 in the bitmask and store them in the offsets array
    int bitid = 0;
    cout << "log(a.size()): " << log(a.size()) << endl;
    for (int i = 0; i < log2(a.size()); i++) {
        cout << "bitmask: " << bitset<16>(bitmask) << endl;
        cout << "i: " << i << endl;
        cout << "bitmask | (1 << i): " << bitset<16>(bitmask | (1 << i)) << endl;
        if ((bitmask | (1 << i)) != bitmask) {
            offsets[bitid] = i;
            bitid++;
        }
    }

    // print the contents of the offsets array
    for (int i = 0; i < offsets_len; i++) {
        cout << "offsets[" << i << "]: " << offsets[i] << endl;
    }

    cudaMemcpy(offsets_gpu, offsets, offsets_len * sizeof(size_t), cudaMemcpyHostToDevice);

    quantum_simulation_gpu<<<blocksPerGrid, threadsPerBlock>>>(
        U_0_gpu, U_1_gpu, U_2_gpu, U_3_gpu, U_4_gpu, U_5_gpu, a_gpu, output_gpu, qubit_0, qubit_1,
        qubit_2, qubit_3, qubit_4, qubit_5, a.size(), auxillary_array_gpu, offsets_gpu);

    // err
    cudaError_t err;
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(err));
    }
    err = cudaMemcpy(output, output_gpu, a.size() * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(err));
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(err));
    }
    // Print the output vector
    for (int i = 0; i < a.size(); i++) {
        printf("%.3f\n", output[i]);
    }

    cudaFree(offsets_gpu);
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
