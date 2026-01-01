#include <__clang_cuda_builtin_vars.h>
#include <__clang_cuda_runtime_wrapper.h>
#include <cuda_runtime.h>


__global__
void mmul_row(float* A, float* B, float* P, int Height, int Width) {
    //produce an output row per thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if(row < Height) {
        for(int k = 0; k < Width; ++k) {
            float P_val = 0;
            for(int j = 0; j < Height; ++j) {
                P_val += A[row*Width + j] * B[j*Width + k];
            }
            P[row*Width + k] = P_val;
        }
    }
}

__global__
void mmul_col(float* A, float* B, float* C, int Height, int Width) {
    //produce an output col per thread
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(col < Width) {
        for(int k = 0; k < Height; ++k) {
            float C_val = 0;
            for(int j = 0; j < Width; ++j) {
                C_val += A[k*Width + j] * B[j*Width + col];
            }
            C[k*Width + col] = C_val;
        }
    }
}