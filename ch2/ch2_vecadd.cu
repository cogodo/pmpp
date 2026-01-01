#include <__clang_cuda_builtin_vars.h>
#include <cuda_runtime.h>

__global__ 
void vecAddKernel(float* A_d, float* B_d, float* C_d, int size) {
    // use the blockSize
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i < size) {
    C_d[i] = A_d[i] * B_d[i];
    }
}

void vecAdd(float* A, float* B, float* C, int n) {
    float *A_d, *B_d, *C_d;
    int size = n * sizeof(float);

    cudaMalloc((void **) &A_d, size);
    cudaMalloc((void **) &B_d, size);
    cudaMalloc((void **) &C_d, size);

    cudaMemcpy(A_d, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, size, cudaMemcpyHostToDevice);
    
    vecAddKernel<<<ceil(n/256.0), 256>>>(A_d, B_d, C_d, n);

    cudaMemcpy(C, C_d, size, cudaMemcpyDeviceToHost);
}

