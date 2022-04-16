time_utils:
	g++ --std=c++17 -O3 src/time_utils.cpp -o build/time_utils -I./include

matmul_cpu:
	g++ -std=c++17 -O3 src/matmul/matmul_cpu.cpp src/time_utils.cpp \
		-o build/matmul_cpu -I./include

matmul_mkl:
	g++ -O3 -fopenmp src/matmul/matmul_mkl.cpp src/time_utils.cpp \
		-o build/matmul_mkl -I/opt/intel/mkl/include -I./include -Wl,--no-as-needed \
		-lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_sequential -lmkl_core -ldl -lpthread -lm -ldl

matmul_mkl_double:
	g++ -O3 -fopenmp src/matmul/matmul_mkl_double.cpp src/time_utils.cpp \
		-o build/matmul_mkl_double -I/opt/intel/mkl/include -I./include -Wl,--no-as-needed \
		-lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_sequential -lmkl_core -ldl -lpthread -lm -ldl

matmul_cuda:
	nvcc -O3 src/matmul/matmul_cuda.cu -o build/matmul_cuda

matmul_cublas:
	nvcc -O3 src/matmul/matmul_cublas.cu -o build/matmul_cublas -I./include -lcublas
	# g++ -O3 src/matmul/matmul_cublas.cu -o build/matmul_cublas -I./include \
	# -lcublas_static -lculibos -lcudart_static -lpthread -ldl -I/usr/local/cuda/include -L /usr/local/cuda/lib64

mandelbrot_cuda:
	nvcc -O3 src/mandelbrot/mandelbrot_cuda.cu -o build/mandelbrot_cuda