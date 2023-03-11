all: matmul_cpu matmul_blas matmul_mkl matmul_mkl_double matmul_cuda matmul_cublas

time_utils:
	g++ --std=c++17 -O3 time_utils.cpp -o build/time_utils -I./include

matmul_cpu:
	g++ -std=c++17 -O3 matmul_cpu.cpp time_utils.cpp \
		-o build/matmul_cpu -I./include

matmul_blas:
	g++ -std=c++17 -O3 matmul_blas.cpp time_utils.cpp \
		-o build/matmul_blas -I./include -lopenblas -lpthread

matmul_mkl:
	g++ -O3 -fopenmp matmul_mkl.cpp time_utils.cpp \
		-o build/matmul_mkl -I/opt/intel/mkl/include -I./include -Wl,--no-as-needed \
		-lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_sequential -lmkl_core -ldl -lpthread -lm -ldl

matmul_mkl_double:
	g++ -O3 -fopenmp matmul_mkl_double.cpp time_utils.cpp \
		-o build/matmul_mkl_double -I/opt/intel/mkl/include -I./include -Wl,--no-as-needed \
		-lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_sequential -lmkl_core -ldl -lpthread -lm -ldl

matmul_cuda:
	nvcc -O3 matmul_cuda.cu time_utils.cpp -o build/matmul_cuda -I./include

matmul_cublas:
	nvcc -O3 matmul_cublas.cu time_utils.cpp -o build/matmul_cublas -I./include -lcublas

.PHONY: clean
clean:
	rm -rf build && mkdir build