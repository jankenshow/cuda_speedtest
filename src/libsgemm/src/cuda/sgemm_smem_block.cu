#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "../include/helper.h"

template <const int BLOCKSIZE>
__global__ void sgemm_smem_block_kernel(int M, int N, int K, float alpha,
                                        const float *A, const float *B,
                                        float beta, float *C)
{
    const uint cRow = blockIdx.x;
    const uint cCol = blockIdx.y;

    __shared__ float As[BLOCKSIZE];
    __shared__ float Bs[BLOCKSIZE];

    const uint threadCol = threadIdx.x % BLOCKSIZE;
    const uint threadRow = threadIdx.x / BLOCKSIZE;

    A += cRow * BLOCKSIZE * K;
    B += cCol * BLOCKSIZE;
    C += cRow * BLOCKSIZE * N + cCol * BLOCKSIZE;

    float tmp = 0.0;
    for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE) {
        As[threadRow * BLOCKSIZE + threadCol] = A[threadRow * K + threadCol];
        Bs[threadRow * BLOCKSIZE + threadCol] = B[threadRow * K + threadCol];

        __syncthreads();
        A += BLOCKSIZE;
        B += BLOCKSIZE * N;

        for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx) {
            tmp += As[threadRow * BLOCKSIZE + dotIdx] *
                   Bs[dotIdx * BLOCKSIZE + threadCol];
        }
        __syncthreads();
    }
    C[threadRow * N + threadCol] =
        alpha * tmp + beta * C[threadRow * N + threadCol];
}

void sgemm_smem_block(int M, int N, int K, float alpha, float *A, float *B,
                      float beta, float *C)
{
    dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
    dim3 blockDim(32 * 32);
    // L1 cache becomes useless, since we access GMEM only via SMEM, so we
    // carve out all of L1 to SMEM. This doesn't currently make a difference,
    // since occupancy is limited by reg and thread count, but it's good to do
    // anyway.
    cudaFuncSetAttribute(sgemm_smem_block_kernel<32>,
                         cudaFuncAttributePreferredSharedMemoryCarveout,
                         cudaSharedmemCarveoutMaxShared);
    sgemm_smem_block_kernel<32>
        <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}
