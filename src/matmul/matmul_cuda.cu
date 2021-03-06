#include <stdio.h>
#include <sys/time.h>
#include <string>
#include <random>
#include <cmath>
#include "matrix_generator.h"
#include "time_utils.h"


__global__ void matmul(float* out, float* x, float* y, int height, int width, int num_prod) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    if (idx < width && idy < height) {
        float tmp = 0.0;
        for (int k=0; k < num_prod; k++) {
            tmp += x[k + idy * num_prod] * y[idx + k * num_prod];
        }
        out[idx + idy * width] = tmp;
    }
}

void test(int size, int iterations, int thread_size) {
    int x_height = size;
    int num_prod = size;
    int y_width = size;

    float* x = new float[x_height * num_prod];
    float* y = new float[num_prod * y_width];
    float* out = new float[x_height * y_width];
    randn_matrices(x, y, x_height, y_width, num_prod);

    dim3 grid(std::ceil(x_height / thread_size), std::ceil(y_width / thread_size), 1);
    dim3 threads(thread_size, thread_size, 1);
    printf("grid_size=(%d, %d, %d)\n", grid.x, grid.y, grid.z);
    printf("thread_size=(%d, %d, %d)\n", threads.x, threads.y, threads.z);

    stopwatch sw;
    double d_sec_memcpy = 0;
    double d_sec_kernel = 0;
    double d_sec_total = 0;

    // double d_sec_memcpy_cuda = 0;
    // double d_sec_kernel_cuda = 0;
    // double d_sec_total_cuda = 0;

    for (int num_iter=0; num_iter<(iterations+1); num_iter++) {

        // 時間計測
        if (num_iter != 0) {
            sw.start();
        }

        // cudaEvent_t start, memh2d_stop, kernel_stop, memd2h_stop;
        // cudaEventCreate(&start);
        // cudaEventCreate(&memh2d_stop);
        // cudaEventCreate(&kernel_stop);
        // cudaEventCreate(&memd2h_stop);
        // cudaEventRecord(start);


        // メモリ移動 host -> device
        float* d_x = nullptr;
        float* d_y = nullptr;
        float* d_out = nullptr;

        cudaMalloc((void **)&d_x, sizeof(float) * x_height * num_prod);
        cudaMalloc((void **)&d_y, sizeof(float) * num_prod * y_width);
        cudaMalloc((void **)&d_out, sizeof(float) * x_height * y_width);

        cudaMemcpy(d_x, x, sizeof(float) * x_height * num_prod, cudaMemcpyHostToDevice);
        cudaMemcpy(d_y, y, sizeof(float) * num_prod * y_width, cudaMemcpyHostToDevice);


        // 時間計測
        if (num_iter != 0) {
            cudaDeviceSynchronize();
            sw.lap();
        }

        // cudaEventRecord(memh2d_stop);
        // cudaEventSynchronize(memh2d_stop);


        // kernel実行
        matmul<<<grid, threads>>>(d_out, d_x, d_y, x_height, y_width, num_prod);


        // 時間計測
        if (num_iter != 0) {
            cudaDeviceSynchronize();
            sw.lap();
        }

        // cudaEventRecord(kernel_stop);
        // cudaEventSynchronize(kernel_stop);


        // メモリ移動 device -> host
        cudaMemcpy(out, d_out, sizeof(float) * x_height * y_width, cudaMemcpyDeviceToHost);


        // 時間計測
        if (num_iter != 0) {
            cudaDeviceSynchronize();
            sw.stop();
            d_sec_memcpy += sw.get_lap(0);
            d_sec_kernel += sw.get_lap(1);
            d_sec_total += sw.get_total();
        }

        // cudaEventRecord(memd2h_stop);
        // cudaEventSynchronize(memd2h_stop);

        // if (num_iter != 0) {
        //     float milisec = 0.0;
        //     cudaEventElapsedTime(&milisec, start, memd2h_stop);
        //     d_sec_total_cuda += milisec;
        //     cudaEventElapsedTime(&milisec, start, memh2d_stop);
        //     d_sec_memcpy_cuda += milisec;
        //     cudaEventElapsedTime(&milisec, memh2d_stop, kernel_stop);
        //     d_sec_kernel_cuda += milisec;
        // }


        // 時間計測後処理
        sw.reset();
        // cudaEventDestroy(start);
        // cudaEventDestroy(memh2d_stop);
        // cudaEventDestroy(kernel_stop);
        // cudaEventDestroy(memd2h_stop);


        // kernel実行後処理
        // printf("計算結果 = %f\n", out[x_height * y_width - 1]);

        cudaFree(d_x);
        cudaFree(d_y);
        cudaFree(d_out);
    }

    cudaDeviceReset();

    d_sec_memcpy /= iterations;
    d_sec_kernel /= iterations;
    d_sec_total /= iterations;

    // d_sec_memcpy_cuda /= iterations;
    // d_sec_kernel_cuda /= iterations;
    // d_sec_total_cuda /= iterations;

    printf("行列サイズ=%d\n", size);
    printf("計算結果=%f\n", out[x_height * y_width - 1]);
    printf("処理時間 : メモリ移動 = %lf\n", d_sec_memcpy);
    printf("処理時間 : カーネル実行 =%lf\n", d_sec_kernel);
    printf("処理時間 : トータル =%lf\n", d_sec_total);
    // printf("処理時間 : メモリ移動 = %lf vs %lf\n", d_sec_memcpy, d_sec_memcpy_cuda);
    // printf("処理時間 : カーネル実行 =%lf vs %lf\n", d_sec_kernel, d_sec_kernel_cuda);
    // printf("処理時間 : トータル =%lf vs %lf\n", d_sec_total, d_sec_total_cuda);

    delete[] x;
    delete[] y;
    delete[] out;
}


int main(int argc, char** argv) {
    int iterations = 10;
    int sizes[3] = {256, 1024, 4096};

    int thread_size;
    if (argc != 2) {
        thread_size = 32;
    } else {
        std::size_t pos;
        thread_size = std::stoi((std::string)argv[1], &pos);
    }

    if (thread_size != 16 && thread_size != 32) {
        thread_size = 32;
    }

    for (auto& size : sizes) {
        test(size, iterations, thread_size);
    }

    return 0;
}