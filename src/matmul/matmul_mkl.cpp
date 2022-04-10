/* C source code is found in dgemm_example.c */

#define minimum(x,y) (((x) < (y)) ? (x) : (y))

#include <stdio.h>
#include "mkl.h"
#include "matrix_generator.h"
#include "time_utils.h"


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
    randn_matrices(x, y, x_height, y_width, num_prod);
    zero_matrix(out, num_prod, num_prod);

    float alpha, beta;
    alpha = 1.0; beta = 0.0;


    stopwatch sw;
    double d_sec = 0;

    for (int num_iter=0; num_iter<(iterations+1); num_iter++) {
        if (num_iter != 0) {
            sw.start();
        }

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
            x_height, y_width, num_prod, alpha, x, num_prod, y, y_width, beta, out, y_width);

        if (num_iter != 0) {
            sw.stop();
        }
    }

    d_sec = sw.get_total() / iterations;
    printf("行列サイズ=%d\n", size);
    printf("計算結果=%f\n", out[x_height * y_width - 1]);
    printf("処理時間=%lf\n\n", d_sec);
    
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