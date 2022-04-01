matmul_cpu:
	g++ -std=c++17 -O3 src/matmul/matmul_cpu.cpp -o build/matmul_cpu

matmul_mkl:
	g++ -O3 src/matmul/matmul_mkl.cpp -o build/matmul_mkl -I/opt/intel/mkl/include -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -ldl -lpthread -lm

matmul_cuda:
	nvcc -O3 src/matmul/matmul_cuda.cu -o build/matmul_cuda

matmul_cublas:
	nvcc -O3 src/matmul/matmul_cublas.cu -o build/matmul_cublas -lcublas

mandelbrot_cuda:
	nvcc -O3 src/mandelbrot/mandelbrot_cuda.cu -o build/mandelbrot_cuda