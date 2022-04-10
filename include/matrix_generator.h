#ifndef MATRIX_GEN_H
#define MATRIX_GEN_H

#include <random>

inline void randn_matrices (float* x, float* y, int x_height, int y_width, int num_prod) {
    static std::random_device rnd;
    static std::default_random_engine eng(rnd());
    static std::uniform_real_distribution<float> distr(-1, 1);

    for (int i=0; i < x_height * num_prod; i++) {
        x[i] = distr(eng);
    }

    for (int i=0; i < num_prod * y_width; i++) {
        y[i] = distr(eng);
    }
}

#endif