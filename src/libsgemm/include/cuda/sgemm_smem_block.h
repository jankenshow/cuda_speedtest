#pragma onece

void sgemm_smem_block(int M, int N, int K, float alpha, float *A, float *B,
                      float beta, float *C);