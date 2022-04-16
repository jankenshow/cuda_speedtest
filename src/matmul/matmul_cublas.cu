#include <stdio.h>
#include <cublas_v2.h>
#include "matrix_generator.h"
#include "time_utils.h"

inline cublasStatus_t gpu_blas_mmul(const float *A, const float *B, float *C, const int m, const int n, const int k, cublasHandle_t* handle) {
    // int lda=m,ldb=k,ldc=m;
    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasStatus_t stat;

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

    stopwatch sw;
    double d_sec_memcpy = 0;
    double d_sec_kernel = 0;
    double d_sec_total = 0;

    // Create a handle for CUBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);
    // cublasStatus_t stat;

    for (int num_iter=0; num_iter<(iterations+1); num_iter++) {

        if (num_iter != 0) {
            sw.start();
        }

        float *d_x, *d_y, *d_out;
        cudaMalloc(&d_x, x_height * num_prod * sizeof(float));
        cudaMalloc(&d_y, num_prod * y_width * sizeof(float));
        cudaMalloc(&d_out, num_prod * num_prod * sizeof(float));

        cudaMemcpy(d_x, x, x_height * num_prod  * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_y, y, num_prod * y_width * sizeof(float), cudaMemcpyHostToDevice);

        if (num_iter != 0) {
            cudaDeviceSynchronize();
            sw.lap();
        }

        gpu_blas_mmul(d_x, d_y, d_out, x_height, y_width, num_prod, &handle);
        // stat = gpu_blas_mmul(d_x, d_y, d_out, x_height, y_width, num_prod, &handle);

        if (num_iter != 0) {
            cudaDeviceSynchronize();
            sw.lap();
        }

        cudaMemcpy(out, d_out, x_height * y_width * sizeof(float), cudaMemcpyDeviceToHost);

        // 時間計測
        if (num_iter != 0) {
            cudaDeviceSynchronize();
            sw.stop();
            d_sec_memcpy += sw.get_lap(0);
            d_sec_kernel += sw.get_lap(1);
            d_sec_total += sw.get_total();
        }

        // if (stat != CUBLAS_STATUS_SUCCESS) {
        //     printf ("CUBLAS failed\n");
        // }

        sw.reset();

        cudaFree(d_x);
        cudaFree(d_y);
        cudaFree(d_out);
    }

    // Destroy the handle
    cublasDestroy(handle);
    cudaDeviceReset();

    d_sec_memcpy /= iterations;
    d_sec_kernel /= iterations;
    d_sec_total /= iterations;

    printf("行列サイズ=%d\n", size);
    printf("計算結果=%f\n", out[x_height * y_width - 1]);
    printf("処理時間 : メモリ移動 = %lf\n", d_sec_memcpy);
    printf("処理時間 : カーネル実行 =%lf\n", d_sec_kernel);
    printf("処理時間 : トータル =%lf\n", d_sec_total);

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