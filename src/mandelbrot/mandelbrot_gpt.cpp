#include <chrono>
#include <immintrin.h>
#include <iostream>
#include <omp.h>
#include <thread>
#include <vector>

#include <opencv2/opencv.hpp>

constexpr int   MAX_ITER = 1000;
constexpr float X_MIN    = -2.0f;
constexpr float X_MAX    = 1.0f;
constexpr float Y_MIN    = -1.5f;
constexpr float Y_MAX    = 1.5f;
constexpr int   WIDTH    = 4096;
constexpr int   HEIGHT   = 4096;

void mandelbrot(std::vector<int> &image, int startY, int endY)
{
    float dx = (X_MAX - X_MIN) / WIDTH;
    float dy = (Y_MAX - Y_MIN) / HEIGHT;

#pragma omp parallel for
    for (int y = startY; y < endY; y++) {
        for (int x = 0; x < WIDTH; x += 8) {
            __m256 mx = _mm256_set_ps(x + 7, x + 6, x + 5, x + 4, x + 3, x + 2,
                                      x + 1, x);
            __m256 my = _mm256_set1_ps(y);
            __m256 real  = _mm256_add_ps(_mm256_mul_ps(mx, _mm256_set1_ps(dx)),
                                         _mm256_set1_ps(X_MIN));
            __m256 imag  = _mm256_add_ps(_mm256_mul_ps(my, _mm256_set1_ps(dy)),
                                         _mm256_set1_ps(Y_MIN));
            __m256 creal = real;
            __m256 cimag = imag;
            __m256i iter = _mm256_setzero_si256();
            for (int i = 0; i < MAX_ITER; ++i) {
                __m256 real2 = _mm256_mul_ps(real, real);
                __m256 imag2 = _mm256_mul_ps(imag, imag);
                __m256 temp  = _mm256_add_ps(real2, imag2);
                __m256 mask =
                    _mm256_cmp_ps(temp, _mm256_set1_ps(4.0f), _CMP_LT_OQ);
                if (_mm256_testz_ps(mask, mask)) {
                    break;
                }
                iter = _mm256_add_epi32(iter, _mm256_castps_si256(mask));
                imag = _mm256_add_ps(_mm256_mul_ps(_mm256_mul_ps(real, imag),
                                                   _mm256_set1_ps(2.0f)),
                                     cimag);
                real = _mm256_add_ps(real2, _mm256_sub_ps(cimag, imag2));
            }
            iter      = _mm256_min_epi32(iter, _mm256_set1_epi32(255));
            __m256i t = _mm256_permute4x64_epi64(iter, 0xd8);
            _mm_store_si128((__m128i *)&image[y * WIDTH + x],
                            _mm256_castsi256_si128(t));
        }
    }
}

void save_image(const std::vector<int> &image, const std::string &path)
{
    cv::Mat img(HEIGHT, WIDTH, CV_8U);

    for (int i = 0; i < HEIGHT; ++i) {
        for (int j = 0; j < WIDTH; ++j) {
            img.at<uchar>(i, j) = image[i * WIDTH + j];
        }
    }

    cv::imwrite(path, img);
}

int main()
{
    std::vector<int>         image(WIDTH * HEIGHT);
    int                      numThreads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;
    int                      rowsPerThread = HEIGHT / numThreads;

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < numThreads; ++i) {
        int startY = i * rowsPerThread;
        int endY   = (i == numThreads - 1) ? HEIGHT : (i + 1) * rowsPerThread;
        threads.emplace_back(mandelbrot, std::ref(image), startY, endY);
    }

    for (auto &thread : threads) {
        thread.join();
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    auto elapsed_time =
        std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    std::cout << "elapsed time: " << double(elapsed_time) / 1000 << "ms"
              << std::endl;
    save_image(image, "output.png");

    return 0;
}
