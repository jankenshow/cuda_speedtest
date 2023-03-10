ディレクトリ構造は以下を予定。

```
.
├── build
├── install
├── CMakeLists.txt
├── Makefile # TODO
├── python
│   ├── matmul_numpy_mkl.ipynb
│   ├── matmul_on_cpu.ipynb
│   ├── matmul_on_gpu.ipynb
│   └── mean_shift.ipynb
├── README.md
└── src
    ├── libsgemm # TODO
    │   ├── CMakeLists.txt
    │   ├── include
    │   │   ├── utils
    │   │   │   ├── matrix_generator.h
    │   │   │   └── time_utils.h
    │   │   ├── cpu
    │   │   └── cuda
    │   │       ├── sgemm_global_mem_coalesce.h
    │   │       ├── sgemm_naive.h
    │   │       └── sgemm_shared_mem_block.h
    │   ├── src
    │   │   ├── include # プライベートヘッダー (空ディレクトリの予定)
    │   │   ├── utils
    │   │   │   ├── matrix_generator.cpp
    │   │   │   └── time_utils.cpp
    │   │   ├── cpu # cpu上のsgemm実施
    │   │   └── cuda
    │   │       ├── sgemm_global_mem_coalesce.cu
    │   │       ├── sgemm_naive.cu
    │   │       └── sgemm_shared_mem_block.cu
    │   └── tools
    ├── mandelbrot # cmake対象外 (気が向いたらやる)
    │   ├── Makefile # TODO
    │   └── mandelbrot_cuda.cu
    ├── matmul # cmake対象外　# TODO (matrix_generator, time_utils含めて)
    │   ├── Makefile
    │   ├── matmul_cublas.cu
    │   ├── matmul_cuda.cu
    │   ├── matmul_openacc.cpp
    │   ├── matmul_blas.cpp
    │   ├── matmul_cpu.cpp
    │   ├── matmul_mkl.cpp
    │   └── matmul_mkl_double.cpp
    └── sgemm
        └── runner.cpp　# TODO
```