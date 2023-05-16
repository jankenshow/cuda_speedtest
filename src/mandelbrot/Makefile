mandelbrot_cpp: clean
	g++ -Ofast -march=native mandelbrot.cpp -o build/mandelbrot `pkg-config opencv4 --cflags --libs`

mandelbrot_omp: clean
	g++ -Ofast -fopenmp -march=native mandelbrot_omp.cpp -o build/mandelbrot `pkg-config opencv4 --cflags --libs`

mandelbrot_avx2: clean
	g++ -Ofast -fopenmp -march=native -mavx2 mandelbrot_avx2.cpp -o build/mandelbrot `pkg-config opencv4 --cflags --libs`

mandelbrot_avx512: clean
	g++ -Ofast -fopenmp -march=native -mavx512f mandelbrot_avx512.cpp -o build/mandelbrot `pkg-config opencv4 --cflags --libs`

mandelbrot_gpt: clean
	g++ -Ofast -fopenmp -march=native -mavx2 mandelbrot_gpt.cpp -o build/mandelbrot `pkg-config opencv4 --cflags --libs`

mandelbrot_cuda: clean
	nvcc -O3 mandelbrot_cuda.cu -o build/mandelbrot

# For arm64 macbook. Instructins are based on Armv8.4a.
mandelbrot_armneon: clean
	clang++ -std=c++17 -O3 -Xclang -fopenmp -L/opt/homebrew/opt/libomp/lib -I/opt/homebrew/opt/libomp/include -lomp -o build/mandelbrot mandelbrot_armneon.cpp

.PHONY: clean
clean:
	rm -rf build && mkdir build

run:
	OMP_NUM_THREADS=128 ./build/mandelbrot