#include <chrono>
#include <cmath>
#include <complex>
#include <iostream>
#include <vector>

#include <opencv2/opencv.hpp>

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

    for (int j = 0; j < height; j++) {
        for (int i = 0; i < width; i++) {
            x = xmin + i * dy;
            y = ymin + j * dy;

            image[j][i] = mandelbrot_kernel<T>(std::complex<T>(x, y));
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
    image   = compute_mandelbrot<float>(image);
    auto t1 = std::chrono::high_resolution_clock::now();

    auto elapsed_time =
        std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    std::cout << "elapsed time: " << double(elapsed_time) / 1000 << "ms"
              << std::endl;
    save_image(image, "output.png");

    return 0;
}