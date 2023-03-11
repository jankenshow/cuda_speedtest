# 概要

- cuda の書き方での速度比較 (`sgemm`)
- cuda vs cpuで速度比較 (`matmul`, 一部gemmを利用) 
- mandelbrotをCUDAで実行

# インストール

## sgemm

```
$ mkdir build && cd build
$ cmake ..
$ make
```

必要であれば　`make`実行後に`make install`をしても良い。

## matmul

```
$ cd src/matmul
$ make
```

## mandelbrot

```
$ cd src/mandelbrot
$ make
```

# 実行

## sgemm


## matmul


## mandelbrot


# ディレクトリ構造

以下を予定。

```
.
├── build
├── install
├── CMakeLists.txt # TODO
├── Makefile # TODO
├── python
│   ├── matmul_numpy_mkl.ipynb
│   ├── matmul_on_cpu.ipynb
│   ├── matmul_on_gpu.ipynb
│   └── mean_shift.ipynb
├── README.md # TODO
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
    ├── mandelbrot 
    │   ├── Makefile
    │   └── mandelbrot_cuda.cu
    ├── matmul
    │   ├── include
    │   │   ├── matrix_generator.h
    │   │   └── time_utils.h
    │   ├── Makefile
    │   ├── matmul_cublas.cu
    │   ├── matmul_cuda.cu
    │   ├── matmul_openacc.cpp
    │   ├── matmul_blas.cpp
    │   ├── matmul_cpu.cpp
    │   ├── matmul_mkl.cpp
    │   ├── matmul_mkl_double.cpp
    │   └── time_utils.cpp
    └── sgemm
        └── runner.cpp　# TODO
```