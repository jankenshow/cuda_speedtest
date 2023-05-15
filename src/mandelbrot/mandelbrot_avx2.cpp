#include <chrono>
#include <immintrin.h>
#include <iostream>
#include <omp.h>
#include <vector>

#include <opencv2/opencv.hpp>

constexpr double xmin     = -1.75f;
constexpr double xmax     = 0.75f;
constexpr double ymin     = -1.25f;
constexpr double ymax     = 1.25f;
constexpr int    height   = 4096;
constexpr int    width    = 4096;
constexpr int    max_iter = 500;

class ComplexSIMD {
  public:
    ComplexSIMD(__m256 real, __m256 imag) : real_(real), imag_(imag) {}

    ComplexSIMD operator+(const ComplexSIMD &other) const
    {
        __m256 real_res = _mm256_add_ps(real(), other.real());
        __m256 imag_res = _mm256_add_ps(imag(), other.imag());
        return ComplexSIMD(real_res, imag_res);
    }

    ComplexSIMD operator*(const ComplexSIMD &other) const
    {
        __m256 real_res = _mm256_fmsub_ps(real(), other.real(),
                                          _mm256_mul_ps(imag(), other.imag()));
        __m256 imag_res = _mm256_fmadd_ps(real(), other.imag(),
                                          _mm256_mul_ps(imag(), other.real()));
        return ComplexSIMD(real_res, imag_res);
    }

    __m256 abs() const
    {
        return _mm256_sqrt_ps(
            _mm256_fmadd_ps(real(), real(), _mm256_mul_ps(imag(), imag())));
    }

    __m256 real() const { return real_; }

    __m256 imag() const { return imag_; }

  private:
    __m256 real_;
    __m256 imag_;
};

__m256i mandelbrot_kernel(ComplexSIMD &c)
{
    ComplexSIMD z    = c;
    __m256i     iter = _mm256_setzero_si256();

    for (int i = 0; i < max_iter; i++) {
        z           = z * z + c;
        __m256 mask = _mm256_cmp_ps(z.abs(), _mm256_set1_ps(2.0f), _CMP_LT_OS);
        __m256i mask_i = _mm256_castps_si256(mask);
        iter           = _mm256_sub_epi32(iter, mask_i);

        if (_mm256_movemask_ps(mask) == 0) {
            break;
        }
    }

    return iter;
}

std::vector<std::vector<int>>
compute_mandelbrot(std::vector<std::vector<int>> &image)
{
    double dx = (xmax - xmin) / width;
    double dy = (ymax - ymin) / height;

#pragma omp parallel for collapse(2)
    for (int j = 0; j < height; j++) {
        for (int i = 0; i < width; i += 8) {
            __m256 c_real = _mm256_setr_ps(
                xmin + (i + 0) * dx, xmin + (i + 1) * dx, xmin + (i + 2) * dx,
                xmin + (i + 3) * dx, xmin + (i + 4) * dx, xmin + (i + 5) * dx,
                xmin + (i + 6) * dx, xmin + (i + 7) * dx);
            __m256 c_imag = _mm256_set1_ps(ymin + j * dy);

            ComplexSIMD c(c_real, c_imag);
            __m256i     iter_counts = mandelbrot_kernel(c);

            int result[8];
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(result),
                                iter_counts);
            for (int k = 0; k < 8; k++) {
                image[j][i + k] = result[k];
            }
        }
    }

    return image;
}

void save_image(const std::vector<std::vector<int>> &image,
                const std::string                   &path)
{
    int height = image.size();
    int width  = image[0].size();

    cv::Mat img(height, width, CV_8U);

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            img.at<uchar>(i, j) = image[i][j];
        }
    }

    cv::imwrite(path, img);
}

int main()
{
    std::vector<std::vector<int>> image(height, std::vector<int>(width));

    auto t0 = std::chrono::high_resolution_clock::now();
    image   = compute_mandelbrot(image);
    auto t1 = std::chrono::high_resolution_clock::now();

    auto elapsed_time =
        std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    std::cout << "elapsed time: " << double(elapsed_time) / 1000 << "ms"
              << std::endl;
    save_image(image, "output.png");

    return 0;
}