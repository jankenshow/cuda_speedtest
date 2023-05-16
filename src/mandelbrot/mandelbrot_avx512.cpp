#include <chrono>
#include <immintrin.h>
#include <iostream>
#include <omp.h>
#include <vector>

#include <opencv2/opencv.hpp>

constexpr float xmin     = -1.75f;
constexpr float xmax     = 0.75f;
constexpr float ymin     = -1.25f;
constexpr float ymax     = 1.25f;
constexpr int   height   = 4096;
constexpr int   width    = 4096;
constexpr int   max_iter = 500;

class ComplexSIMD {
  public:
    ComplexSIMD(__m512 real, __m512 imag) : real_(real), imag_(imag) {}

    ComplexSIMD operator+(const ComplexSIMD &other) const
    {
        __m512 real_res = _mm512_add_ps(real(), other.real());
        __m512 imag_res = _mm512_add_ps(imag(), other.imag());
        return ComplexSIMD(real_res, imag_res);
    }

    ComplexSIMD operator*(const ComplexSIMD &other) const
    {
        __m512 real_res = _mm512_fmsub_ps(real(), other.real(),
                                          _mm512_mul_ps(imag(), other.imag()));
        __m512 imag_res = _mm512_fmadd_ps(real(), other.imag(),
                                          _mm512_mul_ps(imag(), other.real()));
        return ComplexSIMD(real_res, imag_res);
    }

    __m512 abs() const
    {
        __m512 imag2 = _mm512_mul_ps(imag(), imag());
        return _mm512_sqrt_ps(_mm512_fmadd_ps(real(), real(), imag2));
    }

    __m512 real() const { return real_; }

    __m512 imag() const { return imag_; }

  private:
    __m512 real_;
    __m512 imag_;
};

__m512i mandelbrot_kernel(ComplexSIMD &c)
{
    ComplexSIMD z    = c;
    __m512i     iter = _mm512_setzero_si512();
    __m512i     ones = _mm512_set1_epi32(1);

    for (int i = 0; i < max_iter; i++) {
        z              = z * z + c;
        __mmask16 mask = _mm512_cmplt_ps_mask(z.abs(), _mm512_set1_ps(2.0f));
        iter           = _mm512_maskz_add_epi32(mask, iter, ones);

        if (_cvtmask16_u32(mask) == 0) {
            break;
        }
    }

    return iter;
}

std::vector<std::vector<int>>
compute_mandelbrot(std::vector<std::vector<int>> &image)
{
    float dx = (xmax - xmin) / width;
    float dy = (ymax - ymin) / height;

#pragma omp parallel for collapse(2)
    for (int j = 0; j < height; j++) {
        for (int i = 0; i < width; i += 16) {
            __m512 c_real = _mm512_setr_ps(
                xmin + (i + 0) * dx, xmin + (i + 1) * dx, xmin + (i + 2) * dx,
                xmin + (i + 3) * dx, xmin + (i + 4) * dx, xmin + (i + 5) * dx,
                xmin + (i + 6) * dx, xmin + (i + 7) * dx, xmin + (i + 8) * dx,
                xmin + (i + 9) * dx, xmin + (i + 10) * dx,
                xmin + (i + 11) * dx, xmin + (i + 12) * dx,
                xmin + (i + 13) * dx, xmin + (i + 14) * dx,
                xmin + (i + 15) * dx);
            __m512 c_imag = _mm512_set1_ps(ymin + j * dy);

            ComplexSIMD c(c_real, c_imag);
            __m512i     iter_counts = mandelbrot_kernel(c);

            int result[16];
            _mm512_storeu_si512(reinterpret_cast<void *>(result), iter_counts);
            for (int k = 0; k < 16; k++) {
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