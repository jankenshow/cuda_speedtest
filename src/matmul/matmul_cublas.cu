#include <stdio.h>
#include <cublas_v2.h>
#include "matrix_generator.h"

inline cublasStatus_t gpu_blas_mmul(const float *A, const float *B, float *C, const int m, const int n, const int k, cublasHandle_t* handle) {
    // int lda=m,ldb=k,ldc=m;
    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasStatus_t stat;

    // Do the actual multiplication
    stat = cublasSgemm(*handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, A, m, B, k, &beta, C, m);
    return stat;
}

void test(int size, int iterations) {
    // Allocate 3 arrays on CPU
    int x_height = size;
    int num_prod = size;
    int y_width = size;

    // // setup execution parameters
    // cudaSetDevice(0);
    // cudaDeviceProp deviceProp;
    // cudaGetDeviceProperties(&deviceProp, 0);
    // int block_size = (deviceProp.major < 2) ? 16 : 32;
    // dim3 threads(block_size, block_size);
    // dim3 grid(size / threads.x, size / threads.y);
    // printf("block_size=%d, grid_size=%d\n", block_size, size/threads.x);

    float *x = (float *)malloc(x_height * num_prod * sizeof(float));
    float *y = (float *)malloc(num_prod * y_width * sizeof(float));
    float *out = (float *)malloc(num_prod * num_prod * sizeof(float));
    randn_matrices(x, y, x_height, y_width, num_prod);
    zero_matrix(out, num_prod, num_prod);

    double d_sec_memcpy_cuda = 0;
    double d_sec_kernel_cuda = 0;
    double d_sec_total_cuda = 0;

    // Create a handle for CUBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasStatus_t stat;

    for (int num_iter=0; num_iter<(iterations+1); num_iter++) {

        cudaEvent_t start, memh2d_stop, kernel_stop, memd2h_stop;
        cudaEventCreate(&start);
        cudaEventCreate(&memh2d_stop);
        cudaEventCreate(&kernel_stop);
        cudaEventCreate(&memd2h_stop);
        cudaEventRecord(start);

        // Allocate 3 arrays on GPU
        float *d_x, *d_y, *d_out;
        cudaMalloc(&d_x, x_height * num_prod * sizeof(float));
        cudaMalloc(&d_y, num_prod * y_width * sizeof(float));
        cudaMalloc(&d_out, num_prod * num_prod * sizeof(float));
        
        // Optionally we can copy the data back on CPU and print the arrays
        cudaMemcpy(d_x, x, x_height * num_prod  * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_y, y, num_prod * y_width * sizeof(float), cudaMemcpyHostToDevice);

        cudaEventRecord(memh2d_stop);
        cudaEventSynchronize(memh2d_stop);

        // Multiply A and B on GPU
        stat = gpu_blas_mmul(d_x, d_y, d_out, x_height, y_width, num_prod, &handle);

        cudaEventRecord(kernel_stop);
        cudaEventSynchronize(kernel_stop);

        // Copy (and print) the result on host memory
        cudaMemcpy(out, d_out, x_height * y_width * sizeof(float), cudaMemcpyDeviceToHost);

        cudaEventRecord(memd2h_stop);
        cudaEventSynchronize(memd2h_stop);

        if (num_iter != 0) {
            float milisec = 0.0;
            cudaEventElapsedTime(&milisec, start, memd2h_stop);
            d_sec_total_cuda += milisec;
            cudaEventElapsedTime(&milisec, start, memh2d_stop);
            d_sec_memcpy_cuda += milisec;
            cudaEventElapsedTime(&milisec, memh2d_stop, kernel_stop);
            d_sec_kernel_cuda += milisec;
        }

        if (stat != CUBLAS_STATUS_SUCCESS) {
            printf ("CUBLAS failed\n");
        }

        cudaEventDestroy(start);
        cudaEventDestroy(memh2d_stop);
        cudaEventDestroy(kernel_stop);
        cudaEventDestroy(memd2h_stop);

        //Free GPU memory
        cudaFree(d_x);
        cudaFree(d_y);
        cudaFree(d_out);
    }

    // Destroy the handle
    cublasDestroy(handle);
    cudaDeviceReset();

    d_sec_memcpy_cuda /= iterations;
    d_sec_kernel_cuda /= iterations;
    d_sec_total_cuda /= iterations;

    printf("行列サイズ=%d\n", size);
    printf("計算結果=%f\n", out[x_height * y_width - 1]);
    printf("処理時間 : メモリ移動 = %lf\n", d_sec_memcpy_cuda);
    printf("処理時間 : カーネル実行 =%lf\n", d_sec_kernel_cuda);
    printf("処理時間 : トータル =%lf\n", d_sec_total_cuda);

    // Free CPU memory
    free(x);
    free(y);
    free(out);
}


int main() {
    int iterations = 10;
    int sizes[3] = {256, 1024, 4096};

    for (auto& size : sizes) {
        test(size, iterations);
    }

    return 0;
}