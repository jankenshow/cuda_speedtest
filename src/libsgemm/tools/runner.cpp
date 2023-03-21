#include <stdexcept>

#include "../include/cuda/sgemm_coalescing.h"
#include "../include/cuda/sgemm_naive.h"
#include "../include/cuda/sgemm_smem_block.h"
#include "../include/tools.h"

void run_kernel(int kernel_num, int M, int N, int K, float alpha, float *A,
                float *B, float beta, float *C)
{
    switch (kernel_num) {
        // case 0: cublasFP32(handle, M, N, K, alpha, A, B, beta, C); break;
        case 1: sgemm_naive(M, N, K, alpha, A, B, beta, C); break;
        case 2: sgemm_coalescing(M, N, K, alpha, A, B, beta, C); break;
        case 3: sgemm_smem_block(M, N, K, alpha, A, B, beta, C); break;
        // case 4: sgemm1DBlocktiling(M, N, K, alpha, A, B, beta, C); break;
        // case 5: sgemm2DBlocktiling(M, N, K, alpha, A, B, beta, C); break;
        // case 6: sgemmVectorize(M, N, K, alpha, A, B, beta, C); break;
        // case 7:
        //     sgemmResolveBankConflicts(M, N, K, alpha, A, B, beta, C);
        //     break;
        // case 8: sgemmResolveBankExtraCol(M, N, K, alpha, A, B, beta, C);
        // break;
        // case 9: sgemmAutotuned(M, N, K, alpha, A, B, beta, C); break;
        // case 10: sgemmWarptiling(M, N, K, alpha, A, B, beta, C); break;
        // case 11: sgemmDoubleBuffering(M, N, K, alpha, A, B, beta, C); break;
        default: throw std::invalid_argument("Unknown kernel number");
    }
}