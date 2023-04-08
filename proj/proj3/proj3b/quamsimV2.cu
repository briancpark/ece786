#include <bitset>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <string>
#include <vector>
using namespace std;

__device__ void inline quantum_kernel(float* U, float* a, float* output, size_t qid, size_t N,
                                      int tid) {
    if (tid & qid)
        output[tid] = U[3] * a[tid] + U[2] * a[tid - qid];
    else
        output[tid] = U[1] * a[tid + qid] + U[0] * a[tid];
    __syncthreads();
    __syncwarp();
}

__global__ void quantum_simulation_gpu(float* U_0, float* U_1, float* U_2, float* U_3, float* U_4,
                                       float* U_5, float* a, float* output, size_t qubit0,
                                       size_t qubit1, size_t qubit2, size_t qubit3, size_t qubit4,
                                       size_t qubit5, size_t N, size_t* auxillary_array) {

    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    // if (tid >= N) {
    //     return;
    // }

    __shared__ float a_shared[64];
    __shared__ float output_shared[64];

    size_t qid0 = 1 << qubit0;
    size_t qid1 = 1 << qubit1;
    size_t qid2 = 1 << qubit2;
    size_t qid3 = 1 << qubit3;
    size_t qid4 = 1 << qubit4;
    size_t qid5 = 1 << qubit5;

    int tid1 = tid;
    // int tid0 = tid1 + 1;
    quantum_kernel(U_0, a, output, qid0, N, tid1);
    __syncthreads();
    __syncwarp();

    // copy output to a
    a[tid1] = output[tid1];
    // a[tid0] = output[tid0];
    __syncthreads();
    __syncwarp();
    quantum_kernel(U_1, a, output, qid1, N, tid1);

    __syncthreads();
    __syncwarp();

    // copy output to a
    a[tid1] = output[tid1];
    // a[tid0] = output[tid0];
    __syncthreads();
    __syncwarp();
    quantum_kernel(U_2, a, output, qid2, N, tid1);

    __syncthreads();
    __syncwarp();

    // copy output to a
    a[tid1] = output[tid1];
    // a[tid0] = output[tid0];
    __syncthreads();
    __syncwarp();
    quantum_kernel(U_3, a, output, qid3, N, tid1);

    __syncthreads();
    __syncwarp();

    // copy output to a
    a[tid1] = output[tid1];
    // a[tid0] = output[tid0];
    __syncthreads();
    __syncwarp();
    quantum_kernel(U_4, a, output, qid4, N, tid1);

    __syncthreads();
    __syncwarp();

    // copy output to a
    a[tid1] = output[tid1];
    // a[tid0] = output[tid0];
    __syncthreads();
    __syncwarp();
    quantum_kernel(U_5, a, output, qid5, N, tid1);

    __syncthreads();
    __syncwarp();

    // if (tid1 == 0) {
    //     // printf(
    //     // "qubit0 = %lu, qubit1 = %lu, qubit2 = %lu, qubit3 = %lu, qubit4 = %lu, qubit5 =
    //     %lu\n",
    //     // qubit0, qubit1, qubit2, qubit3, qubit4, qubit5);
    //     printf("qid0 = %lu, qid1 = %lu, qid2 = %lu, qid3 = %lu, qid4 = %lu, qid5 = %lu\n", qid0,
    //            qid1, qid2, qid3, qid4, qid5);
    // }
    // __shared__ float a_shared[64];
    // __shared__ float output_shared[64];

    // size_t t = N - 1;
    // size_t bid = blockDim.x * blockIdx.x;
    // if (tid1 == 0) {
    //     printf("t = %lu\n", t);
    // }
    // // set t's 1s to 0s based on qids
    // size_t bitmask = qid0 | qid1 | qid2 | qid3 | qid4 | qid5;
    // t &= ~bitmask;

    // if (tid1 == 0) {
    //     printf("bitmask = %lu\n", bitmask);
    //     printf("t = %lu\n", t);
    // }

    // printf("bid = %lu\n", bid);

    // // based on the bid, set the t bits to 1s
    // // THIS IS ONLY FOR 128
    // if (bid < 32) {
    //     t = 0;
    // } else {
    //     t = 2;
    // }
    // __syncthreads();
    // __syncwarp();

    // if (tid1 % 32 == 0) {
    //     for (int i = 0; i < 64; i++) {
    //         printf("auxillary_array[%d] = %lu\n", i, auxillary_array[i]);
    //         a_shared[i] = a[auxillary_array[i] + t];
    //     }
    // }

    // __syncthreads();
    // __syncwarp();

    // if (tid1 == 0) {
    //     for (int i = 0; i < 64; i++) {
    //         printf("a_shared[%d] = %f\n", i, a_shared[i]);
    //     }
    // }
    // __syncthreads();
    // __syncwarp();

    // // Apply the six single-qubit gates on each of the fragments and get the results. Similar to
    // // applying 6 gates on a 6-qubit circuit, you can split this process into 6 steps: applying
    // // the first gate to qubit 0, applying the second gate to qubit 1, applying the third gate
    // // to qubit 2, applying the fourth gate to qubit 3, applying the fifth gate to qubit 4, and
    // // applying the sixth gate to qubit 5. And for each of the steps, the computation is the
    // // same as in PA1, and here since we have 2^6 values that need to be calculated in each
    // // thread block, there will be 2^5 threads in each thread block, one thread will do one
    // // matrix multiplication

    // // The auxiliary array is used to store the indices of the 2^6 values that need to be
    // // calculated in each thread block.
    // // We can perform quantum simulation by using the auxiliary array to map the 2^6 values where
    // // they are stored consecutively

    // // Apply the first gate to qubit 0
    // __syncthreads();
    // __syncwarp();
    // size_t tid_0 = threadIdx.x * 2;
    // size_t tid_1 = threadIdx.x * 2 + 1;

    // if (qid0&)

    //     __syncthreads();
    // __syncwarp();

    // // copy output_shared back to a_shared
    // a_shared[tid_0] = output_shared[tid_0];
    // a_shared[tid_1] = output_shared[tid_1];

    // __syncthreads();
    // __syncwarp();

    // // if (tid1 % 32 == 0) {
    // //     for (int i = 0; i < 64; i++) {
    // //         printf("a_shared[%d] = %f\n", i, a_shared[i]);
    // //         printf("output_shared[%d] = %f\n", i, output_shared[i]);
    // //     }
    // // }

    // // printf("%f * %f = %f\n", U_1[0], a_shared[tid_0], U_1[0] * a_shared[tid_0]);
    // // printf("%f * %f = %f\n", U_1[1], a_shared[tid_1], U_1[1] * a_shared[tid_1]);
    // // printf("%f * %f = %f\n", U_1[2], a_shared[tid_0], U_1[2] * a_shared[tid_0]);
    // // printf("%f * %f = %f\n", U_1[3], a_shared[tid_1], U_1[3] * a_shared[tid_1]);

    // if (tid0 & qid0) {
    //     output_shared[tid_0] = U_1[0] * a_shared[tid_0] + U_1[1] * a_shared[tid_1];
    // }
    // output_shared[tid_0] = U_1[0] * a_shared[tid_0] + U_1[1] * a_shared[tid_1];
    // output_shared[tid_1] = U_1[2] * a_shared[tid_0] + U_1[3] * a_shared[tid_1];

    // // output_shared[tid_0] = U_2[0] * a_shared[tid_0] + U_2[1] * a_shared[tid_1];
    // // output_shared[tid_1] = U_2[2] * a_shared[tid_0] + U_2[3] * a_shared[tid_1];

    // __syncthreads();
    // __syncwarp();
    // // copy back output_shared to output
    // output[auxillary_array[tid_0] + t] = output_shared[tid_0];
    // output[auxillary_array[tid_1] + t] = output_shared[tid_1];
    // __syncthreads();
    // __syncwarp();
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
        // TODO (bcp) DEBUG
        // for (int i = 0; i < 4; i++) {
        //     cout << U[i] << " ";
        // }
        // cout << endl;
    }

    // Read in the vector until we hit an empty line
    string line;
    getline(input_file, line);
    getline(input_file, line);

    while (getline(input_file, line) && !line.empty()) {
        a.push_back(stof(line));
    }

    // Print the vector contents:
    // for (int i = 0; i < a.size(); i++) {
    //     printf("%.3f\n", a[i]);
    // }

    // Read in the qubit
    input_file >> qubit_0;
    input_file >> qubit_1;
    input_file >> qubit_2;
    input_file >> qubit_3;
    input_file >> qubit_4;
    input_file >> qubit_5;

    // TODO (bcp) DEBUG: Print the qubits:
    // cout << qubit_0 << endl;
    // cout << qubit_1 << endl;
    // cout << qubit_2 << endl;
    // cout << qubit_3 << endl;
    // cout << qubit_4 << endl;
    // cout << qubit_5 << endl;

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
    // cout << "Blocks per grid: " << blocksPerGrid << endl;
    // cout << "Threads per block: " << threadsPerBlock << endl;

    /*
    (const float* U_0, const float* U_1, const float* U_2,
    const float* U_3, const float* U_4, const float* U_5,
    const float* a, float* output, size_t qubit0, size_t
    qubit1, size_t qubit2, size_t qubit3, size_t qubit4, size_t qubit5, size_t N) {
    */

    // cout << qubit_0 << endl;
    // cout << qubit_1 << endl;
    // cout << qubit_2 << endl;
    // cout << qubit_3 << endl;
    // cout << qubit_4 << endl;
    // cout << qubit_5 << endl;

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
    // for (int i = 0; i < 64; i++) {
    //     // binary
    //     cout << bitset<6>(auxillary_array[i]) << endl;
    // }

    quantum_simulation_gpu<<<blocksPerGrid, threadsPerBlock>>>(
        U_0_gpu, U_1_gpu, U_2_gpu, U_3_gpu, U_4_gpu, U_5_gpu, a_gpu, output_gpu, qubit_0, qubit_1,
        qubit_2, qubit_3, qubit_4, qubit_5, a.size(), auxillary_array_gpu);

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
