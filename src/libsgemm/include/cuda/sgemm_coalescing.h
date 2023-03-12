#pragma onece

void sgemm_coalescing(int M, int N, int K, float alpha, float *A, float *B,
                      float beta, float *C);