#include <cmath>
#include <numeric>
#include <stdio.h>
#include <sys/time.h>

__global__ void mandelbrot_kernel(float *out, float *r_vals, float *i_vals,
                                  int size_r, int size_i, int max_iter,
                                  float upper_bound)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    int ij  = idx + size_r * idy;
    if (idx < size_r && idy < size_i) {
        int   diverge_flag = 0;
        float a            = r_vals[idx];
        float b            = i_vals[idy];
        float z_r          = 0;
        float z_i          = 0;
        for (int i = 0; i < max_iter; i++) {
            z_r = (z_r * z_r) - (z_i * z_i) + a;
            z_i = 2 * z_r * z_i + b;
            if (sqrtf(z_r * z_r + z_i * z_i) > upper_bound) {
                out[ij]      = (float)i;
                diverge_flag = 1;
                break;
            }
        }
        if (diverge_flag == 0) {
            out[ij] = (float)max_iter;
        }
    }
}

int main()
{
    int thread_size = 32;

    // int max_iter = 255;
    int   max_iter    = 500;
    float upper_bound = 2.0;

    // int size_r = 16384;
    // int size_i = 16384;
    int   size_r    = 4096;
    int   size_i    = 4096;
    float range_min = -2.0;
    float range_max = 2.0;

    float range_step_r =
        (abs(range_max) + abs(range_min)) / (float)(size_r - 1);
    float range_step_i =
        (abs(range_max) + abs(range_min)) / (float)(size_i - 1);

    printf("サイズ = %d x %d, 最大イテレーション  = %d, 上限 = %f\n", size_r,
           size_i, max_iter, upper_bound);

    float *r_vals = new float[size_r];
    float *i_vals = new float[size_i];
    float *out    = new float[size_r * size_i];

    r_vals[0] = range_min;
    for (int i = 1; i < size_r; i++) {
        r_vals[i] = r_vals[i - 1] + range_step_r;
    }

    i_vals[0] = range_min;
    for (int i = 1; i < size_i; i++) {
        i_vals[i] = i_vals[i - 1] + range_step_i;
    }

    cudaEvent_t start, memh2d_stop, kernel_stop, memd2h_stop;
    cudaEventCreate(&start);
    cudaEventCreate(&memh2d_stop);
    cudaEventCreate(&kernel_stop);
    cudaEventCreate(&memd2h_stop);
    cudaEventRecord(start);

    float *d_r_vals = nullptr;
    float *d_i_vals = nullptr;
    float *d_out    = nullptr;

    cudaMalloc((void **)&d_r_vals, sizeof(float) * size_r);
    cudaMalloc((void **)&d_i_vals, sizeof(float) * size_i);
    cudaMalloc((void **)&d_out, sizeof(float) * size_r * size_i);

    cudaMemcpy(d_r_vals, r_vals, sizeof(float) * size_r,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_i_vals, i_vals, sizeof(float) * size_i,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_out, out, sizeof(float) * size_r * size_i,
               cudaMemcpyHostToDevice);

    cudaEventRecord(memh2d_stop);
    cudaEventSynchronize(memh2d_stop);

    dim3 grid(std::ceil(size_r / thread_size), std::ceil(size_i / thread_size),
              1);
    dim3 threads(thread_size, thread_size, 1);

    mandelbrot_kernel<<<grid, threads>>>(d_out, d_r_vals, d_i_vals, size_r,
                                         size_i, max_iter, upper_bound);
    // cudaDeviceSynchronize();

    cudaEventRecord(kernel_stop);
    cudaEventSynchronize(kernel_stop);

    cudaMemcpy(out, d_out, sizeof(float) * size_r * size_i,
               cudaMemcpyDeviceToHost);
    printf("計算結果=%f\n", out[size_r * size_i / 2 + size_r / 2]);
    float sum = std::accumulate(&out[0], &out[size_r * size_i - 1], 0);
    printf("処理回数=%f\n", sum);

    cudaEventRecord(memd2h_stop);
    cudaEventSynchronize(memd2h_stop);

    float milisec = 0.0;
    cudaEventElapsedTime(&milisec, start, memh2d_stop);
    printf("処理時間 : メモリ移動 = %lf\n", milisec / 1000);
    cudaEventElapsedTime(&milisec, memh2d_stop, kernel_stop);
    printf("処理時間 : カーネル実行 =%lf\n", milisec / 1000);
    cudaEventElapsedTime(&milisec, start, memd2h_stop);
    printf("処理時間 : トータル =%lf\n", milisec / 1000);

    // 時間計測後処理
    cudaEventDestroy(start);
    cudaEventDestroy(memh2d_stop);
    cudaEventDestroy(kernel_stop);
    cudaEventDestroy(memd2h_stop);

    cudaFree(d_r_vals);
    cudaFree(d_i_vals);
    cudaFree(d_out);

    cudaDeviceReset();

    delete[] r_vals;
    delete[] i_vals;
    delete[] out;
}