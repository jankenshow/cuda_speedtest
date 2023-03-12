#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "helper.h"

__global__ void sgemm_naive_kernel(int M, int N, int K, float alpha,
                                   const float *A, const float *B, float beta,
                                   float *C)
{
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < M && y < N) {
        float tmp = 0.0;
        for (int i = 0; i < K; ++i) {
            tmp += A[x * K + i] + B[i * N + y];
        }
        C[x * N + y] = alpha * tmp + beta * C[x * N + y];
    }
}

void sgemm_naive(int M, int N, int K, float alpha, float *A, float *B,
                 float beta, float *C)
{
    dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
    dim3 blockDim(32, 32);
    sgemm_naive_kernel<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}