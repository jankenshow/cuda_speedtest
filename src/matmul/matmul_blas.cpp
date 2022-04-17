#include <stdio.h>
#include <cblas.h>
#include "matrix_generator.h"
#include "time_utils.h"


int test(int size, int iterations) {
    int x_height = size;
    int num_prod = size;
    int y_width = size;

    float *x = (float *)malloc(x_height * num_prod * sizeof(float));
    float *y = (float *)malloc(num_prod * y_width * sizeof(float));
    float *out = (float *)malloc(num_prod * num_prod * sizeof(float));
    if (x == NULL || y == NULL || out == NULL) {
        printf( "\n ERROR: Can't allocate memory for matrices. Aborting... \n\n");
        free(x);
        free(y);
        free(out);
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

        // sgemm("N", "N", &x_height, &y_width, &num_prod, &alpha, x, &num_prod, y, &y_width, &beta, out, &y_width);
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

    free(x);
    free(y);
    free(out);

    return 0;
}

int main() {
    int iterations = 10;
    int sizes[3] = {256, 1024, 4096};

    // omp_set_num_threads(16);
    openblas_set_num_threads(8);
    int max_threads = openblas_get_num_threads();
    printf("thread数=%d\n\n", max_threads);

    for (auto& size : sizes) {
        test(size, iterations);
    }

    return 0;
}