/* C source code is found in dgemm_example.c */

#define minimum(x,y) (((x) < (y)) ? (x) : (y))

#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <random>
#include "mkl.h"


inline void randn_matrices (float* x, float* y, float* out, int x_height, int y_width, int num_prod) {
    std::random_device rnd;
    std::default_random_engine eng(rnd());
    std::uniform_real_distribution<float> distr(-1, 1);

    for (int i=0; i < x_height * num_prod; i++) {
        x[i] = (float)distr(eng);
    }

    for (int i=0; i < num_prod * y_width; i++) {
        y[i] = (float)distr(eng);
    }

    for (int i=0; i < num_prod * num_prod; i++) {
        out[i] = 0.0;
    }
}


int test(int size, int iterations) {
    int x_height = size;
    int num_prod = size;
    int y_width = size;

    float *x = (float *)mkl_malloc(x_height * num_prod * sizeof(float), 32);
    float *y = (float *)mkl_malloc(num_prod * y_width * sizeof(float), 32);
    float *out = (float *)mkl_malloc(num_prod * num_prod * sizeof(float), 32);
    if (x == NULL || y == NULL || out == NULL) {
        printf( "\n ERROR: Can't allocate memory for matrices. Aborting... \n\n");
        mkl_free(x);
        mkl_free(y);
        mkl_free(out);
        return 1;
    }
    randn_matrices(x, y, out, x_height, y_width, num_prod);

    float alpha, beta;
    alpha = 1.0; beta = 0.0;


    struct timespec start_time, end_time;
    unsigned int sec;
    int nsec;
    double d_sec = 0;

    for (int num_iter=0; num_iter<(iterations+1); num_iter++) {
        clock_gettime(CLOCK_REALTIME, &start_time);

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
            x_height, y_width, num_prod, alpha, x, num_prod, y, y_width, beta, out, y_width);

        clock_gettime(CLOCK_REALTIME, &end_time);
        sec = end_time.tv_sec - start_time.tv_sec;
        nsec = end_time.tv_nsec - start_time.tv_nsec;

        if (num_iter != 0) {
            d_sec += (double)sec + (double)nsec / (1000 * 1000 * 1000);
        }
    }

    d_sec /= iterations;
    printf("行列サイズ=%d\n", size);
    printf("計算結果=%f\n", out[x_height * y_width - 1]);
    printf("処理時間=%lf\n", d_sec);
    
    mkl_free(x);
    mkl_free(y);
    mkl_free(out);

    return 0;
}

int main() {
    int iterations = 10;
    int sizes[3] = {256, 1024, 4096};

    for (auto& size : sizes) {
        test(size, iterations);
    }

    return 0;
}