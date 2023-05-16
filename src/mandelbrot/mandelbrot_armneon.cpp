// For arm64 macbook. Instructins are based on Armv8.4a.
#include <arm_neon.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>

constexpr float xmin           = -1.75;
constexpr float xmax           = 0.75;
constexpr float ymin           = -1.25;
constexpr float ymax           = 1.25;
constexpr int   width          = 4096;
constexpr int   height         = 4096;
constexpr int   max_iterations = 500;

uint32x4_t mandelbrot_neon(float32x4_t real, float32x4_t imag)
{
    float32x4_t z_real     = real;
    float32x4_t z_imag     = imag;
    uint32x4_t  iterations = vdupq_n_u32(0);

    for (int i = 0; i < max_iterations; ++i) {
        float32x4_t z_real2 = vmulq_f32(z_real, z_real);
        float32x4_t z_imag2 = vmulq_f32(z_imag, z_imag);

        uint32x4_t mask =
            vcltq_f32(vaddq_f32(z_real2, z_imag2), vdupq_n_f32(4.0f));
        if (vmaxvq_u32(mask) == 0) {
            break;
        }

        iterations = vsubq_u32(iterations, mask);
        z_imag     = vaddq_f32(
            vmulq_f32(z_real, vmulq_f32(z_imag, vdupq_n_f32(2.0f))), imag);
        z_real = vaddq_f32(vsubq_f32(z_real2, z_imag2), real);
    }

    // return vreinterpretq_u8_u32(iterations);
    return iterations;
}

std::vector<std::vector<int>>
compute_mandelbrot(std::vector<std::vector<int>> &image)
{
    float dx = (xmax - xmin) / (float)(width - 1);
    float dy = (ymax - ymin) / (float)(height - 1);

#pragma omp parallel for collapse(2)
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; x += 4) {
            float32x4_t real   = {xmin + (x * dx), xmin + ((x + 1) * dx),
                                  xmin + ((x + 2) * dx), xmin + ((x + 3) * dx)};
            float32x4_t imag   = vdupq_n_f32(ymin + (y * dy));
            uint32x4_t  values = mandelbrot_neon(real, imag);
            uint32_t    iterations[4];
            vst1q_u32(iterations, values);

            for (int k = 0; k < 4; k++) {
                image[y][x + k] = (int)iterations[k];
            }
        }
    }
    return image;
}

int save_ppm(std::vector<std::vector<int>> &image)
{
    std::ofstream output("output.ppm", std::ios::binary);
    if (!output.is_open()) {
        std::cerr << "Could not open the output file." << std::endl;
        return 1;
    }
    output << "P6\n" << width << " " << height << "\n255\n";
    for (const auto &row : image) {
        for (const auto &pixel : row) {
            uint8_t color =
                static_cast<uint8_t>(((float)pixel / max_iterations) * 255);
            output.write(reinterpret_cast<const char *>(&color), 1);    // R
            output.write(reinterpret_cast<const char *>(&color), 1);    // G
            output.write(reinterpret_cast<const char *>(&color), 1);    // B
        }
    }
    output.close();

    std::cout << "The PPM image has been successfully saved as output.ppm"
              << std::endl;
    return 0;
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

    // save_ppm(image);
    return 0;
}
