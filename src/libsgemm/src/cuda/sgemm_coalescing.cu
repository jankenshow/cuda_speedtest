#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "../include/helper.h"

template <const uint BLOCKSIZE>
__global__ void sgemm_coalescing_kernel(int M, int N, int K, float alpha,
                                        const float *A, const float *B,
                                        float beta, float *C)
{
    const int cRow = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
    const int cCol = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);

    if (cRow < M && cCol < N) {
        float tmp = 0.0;
        for (int i = 0; i < K; ++i) {
            tmp += A[cRow * K + i] * B[i * N + cCol];
        }
        C[cRow * N + cCol] = alpha * tmp + beta * C[cRow * N + cCol];
    }
}

void sgemm_coalescing(int M, int N, int K, float alpha, float *A, float *B,
                      float beta, float *C)
{
    dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
    dim3 blockDim(32 * 32);
    sgemm_coalescing_kernel<32>
        <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}