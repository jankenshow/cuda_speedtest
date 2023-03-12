#include "../include/helper.h"

void sgemm_cpu(int M, int N, int K, float alpha, float *A, float *B,
               float beta, float *C)
{
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            // int out_idx = i * width + j;
            float temp = 0.0;
            for (int k = 0; k < K; k++) {
                temp += MMCOL(A, K, i, k) * MMCOL(B, K, k, j);
            }
            MMCOL(C, N, i, j) = alpha * temp + beta * MMCOL(C, N, i, j);
        }
    }
}
