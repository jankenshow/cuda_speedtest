#include <chrono>
#include <cmath>
#include <complex>
#include <iostream>
#include <vector>

// constexpr double xmin     = -1.75f;
// constexpr double xmax     = 0.75f;
// constexpr double ymin     = -1.25f;
// constexpr double ymax     = 1.25f;
constexpr float xmin     = -1.75f;
constexpr float xmax     = 0.75f;
constexpr float ymin     = -1.25f;
constexpr float ymax     = 1.25f;
constexpr int   height   = 4096;
constexpr int   width    = 4096;
constexpr int   max_iter = 500;

template <typename T> int mandelbrot_kernel(const std::complex<T> &c)
{
    std::complex<T> z = c;
    for (int i = 0; i < max_iter; i++) {
        z = z * z + c;
        if (std::abs(z) > 2) {
            return i;
        }
    }

    return max_iter;
}

template <typename T>
std::vector<std::vector<int>>
compute_mandelbrot(std::vector<std::vector<int>> &image)
{
    T x, y;

    T dx = (xmax - xmin) / width;
    T dy = (ymax - ymin) / height;

#pragma omp parallel for collapse(2)
    for (int j = 0; j < height; j++) {
        for (int i = 0; i < width; i++) {
            x = xmin + i * dy;
            y = ymin + j * dy;

            image[j][i] = mandelbrot_kernel<T>(std::complex<T>(x, y));
        }
    }

    return image;
}

int main()
{
    std::vector<std::vector<int>> image(height, std::vector<int>(width));

    auto t0 = std::chrono::high_resolution_clock::now();
    image   = compute_mandelbrot<float>(image);
    auto t1 = std::chrono::high_resolution_clock::now();

    auto elapsed_time =
        std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

    std::cout << "elapsed time: " << double(elapsed_time) / 1000 << "ms"
              << std::endl;

    return 0;
}