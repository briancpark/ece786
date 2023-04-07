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
}

__device__ void inline quantum_kernel(float* U, float* a, float* output, size_t qubit, size_t N,
                                      int tid) {
    register size_t qid = 1 << qubit;

    if (tid & qid)
        output[tid] = U[2] * a[tid - qid] + U[3] * a[tid];
    else
        output[tid] = U[0] * a[tid] + U[1] * a[tid + qid];
}

__global__ void quantum_simulation_gpu(const float* U_0, const float* U_1, const float* U_2,
                                       const float* U_3, const float* U_4, const float* U_5,
                                       const float* a, float* output, int qubit0, int qubit1,
                                       int qubit2, int qubit3, int qubit4, int qubit5, int N) {

    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    // For an n-qubit quantum circuit that has an initial quantum state of 2^n values, split the 2^n
    // values into several independent fragments of size 2^6, then map each fragment to one GPU
    // thread block. For each thread block, load the 2^6 values of the fragment from global memory
    // to shared memory. (The reason for using 2^6 as the fragment size is because there are six
    // single-qubit gates applied to six different qubits.)

    extern __shared__ float a_shared[64];

    size_t qid0 = 1 << qubit0;
    size_t qid1 = 1 << qubit1;
    size_t qid2 = 1 << qubit2;
    size_t qid3 = 1 << qubit3;
    size_t qid4 = 1 << qubit4;
    size_t qid5 = 1 << qubit5;

    a_shared[tid * 2] = a[tid * 2];
    a_shared[(tid * 2) + 1] = a[(tid * 2) + 1];

    __syncthreads();

    // Apply the six single-qubit gates on each of the fragments and get the results. Similar to
    // applying 6 gates on a 6-qubit circuit, you can split this process into 6 steps: applying the
    // first gate to qubit 0, applying the second gate to qubit 1, applying the third gate to qubit
    // 2, applying the fourth gate to qubit 3, applying the fifth gate to qubit 4, and applying the
    // sixth gate to qubit 5. And for each of the steps, the computation is the same as in PA1, and
    // here since we have 2^6 values that need to be calculated in each thread block, there will be
    // 2^5 threads in each thread block, one thread will do one matrix multiplication

    __syncthreads();
    // call the kernel function
    quantum_kernel((float*)U_0, a_shared, output, qubit0, N, tid);
    __syncthreads();
    // copy shared memory back to global memory
    output[tid * 2] = a_shared[tid * 2];
    output[(tid * 2) + 1] = a_shared[(tid * 2) + 1];
    printf("shared[%d] = %f\n", tid, a_shared[tid]);
    printf("output[%d] = %f\n", tid, output[tid]);
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
    cout << "Blocks per grid: " << blocksPerGrid << endl;
    cout << "Threads per block: " << threadsPerBlock << endl;

    quantum_simulation_gpu<<<blocksPerGrid, threadsPerBlock>>>(
        U_0_gpu, U_1_gpu, U_2_gpu, U_3_gpu, U_4_gpu, U_5_gpu, a_gpu, output_gpu, qubit_0, qubit_1,
        qubit_2, qubit_3, qubit_4, qubit_5, a.size());

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