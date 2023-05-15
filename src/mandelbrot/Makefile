mandelbrot_cpp: clean
	g++ -Ofast -march=native mandelbrot.cpp -o build/mandelbrot `pkg-config opencv4 --cflags --libs`

mandelbrot_omp: clean
	g++ -Ofast -fopenmp -march=native mandelbrot_omp.cpp -o build/mandelbrot `pkg-config opencv4 --cflags --libs`

mandelbrot_simd: clean
	g++ -Ofast -fopenmp -march=native -mavx2 mandelbrot_simd.cpp -o build/mandelbrot `pkg-config opencv4 --cflags --libs`

mandelbrot_gpt: clean
	g++ -Ofast -fopenmp -march=native -mavx2 mandelbrot_gpt.cpp -o build/mandelbrot `pkg-config opencv4 --cflags --libs`

mandelbrot_cuda: clean
	nvcc -O3 mandelbrot_cuda.cu -o build/mandelbrot

.PHONY: clean
clean:
	rm -rf build && mkdir build

run:
	OMP_NUM_THREADS=128 ./build/mandelbrot