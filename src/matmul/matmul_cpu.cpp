#define MMCOL(x, dj, i, j) ((x)[(i)*(dj)+(j)])

#include <stdio.h>
#include "time_utils.h"
#include "matrix_generator.h"

void matmul(float* out, float* x, float* y, int height, int width, int num_prod) {
    for (int i=0; i<height; i++) {
        for (int j=0; j<width; j++) {
            // int out_idx = i * width + j;
            float temp = 0.0;
            for (int k=0; k<num_prod; k++) {
                temp += MMCOL(x, num_prod, i, k) * MMCOL(y, num_prod, k, j);
            }
            MMCOL(out, width, i, j) = temp;
        }
    }
}

void test(int size, int iterations) {
    int x_height = size;
    int num_prod = size;
    int y_width = size;

    float* x = new float[x_height * num_prod];
    float* y = new float[num_prod * y_width];
    float* out = new float[x_height * y_width];
    randn_matrices(x, y, x_height, y_width, num_prod);

    stopwatch sw;
    double d_sec;

    for (int num_iter=0; num_iter<iterations+1; num_iter++) {
        if (num_iter != 0) {
            sw.start();
        }

        matmul(out, x, y, x_height, y_width, num_prod);
        // printf("計算結果=%f\n", out[x_height * y_width - 1]);

        if (num_iter != 0) {
            sw.stop();
        }
    }
    d_sec = sw.get_total();
    d_sec /= iterations;
    printf("行列サイズ=%d\n", size);
    printf("計算結果=%f\n", out[x_height * y_width - 1]);
    printf("処理時間=%lf\n", d_sec);

    delete[] x;
    delete[] y;
    delete[] out;
}

int main() {
    int iterations = 10;
    int sizes[3] = {256, 1024, 4096};

    for (auto& size : sizes) {
        test(size, iterations);
    }

    return 0;
}