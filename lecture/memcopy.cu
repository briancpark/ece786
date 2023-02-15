#include <cuda_runtime.h>
#include <stdio.h>

size_t N;

__global__ void VecCopy(float* A, float* B, size_t N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        B[i] = A[i];
    }
}

int main() {
    float *A, *B;
    N = 1000000;
    cudaMallocManaged(&A, N * sizeof(float));
    cudaMallocManaged(&B, N * sizeof(float));
    for (int i = 0; i < N; i++) {
        A[i] = i;
        B[i] = 0;
    }
    // size_t threadsPerBlock = 256;

    VecCopy<<<1, N>>>(A, B, N);
    return 0;
}