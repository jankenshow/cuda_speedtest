# 概要  

- cuda の書き方での速度比較 (`sgemm`)  
- cuda vs cpuで速度比較 (`matmul`, 一部gemmを利用)  
- mandelbrotをCUDAで実行  

# インストール  

cmakeでビルドできるのは`sgemm`のみ  
残り二つは、各ディレクトリで`nvcc`を利用してビルド。  

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
├── CMakeLists.txt
├── Makefile
├── python
│   ├── matmul_numpy_mkl.ipynb
│   ├── matmul_on_cpu.ipynb
│   ├── matmul_on_gpu.ipynb
│   └── mean_shift.ipynb
├── README.md
└── src
    ├── libsgemm
    │   ├── CMakeLists.txt
    │   ├── include
    │   │   ├── utils
    │   │   │   ├── matrix_generator.h
    │   │   │   └── time_utils.h
    │   │   ├── cpu
    │   │   │   └── sgemm_cpu.h
    │   │   ├── cuda
    │   │   │   ├── sgemm_coalescing.h
    │   │   │   ├── sgemm_naive.h
    │   │   │   └── sgemm_smem_block.h
    │   │   ├── kernels.h # TODO
    │   │   └── tools.h
    │   ├── src
    │   │   ├── include # プライベートヘッダー
    │   │   ├── utils
    │   │   │   ├── matrix_generator.cpp
    │   │   │   └── time_utils.cpp
    │   │   ├── cpu # cpu上のsgemm実施
    │   │   └── cuda
    │   │       ├── sgemm_coalescing.cu
    │   │       ├── sgemm_naive.cu
    │   │       └── sgemm_smem_block.cu
    │   └── tools
    │       └── runner.cpp
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
        └── run.cpp　# TODO
```

# TODO  

- `run.cpp` の整理  
    - 速度測定用の実装(`libsgemm/tools/time_kernel.cu`などを作成)
    - run.cppの実装 
        - 速度測定用の実装を呼び出す
        - 結果出力
- cuda kernelsの拡充  
- cublasによるsgemmの追加  
- 結果をまとめる  

# (優先度:低) 整理しておきたいこと  

include方法は "CMakeLists.txtの修正" を含む  

- `libsgemm`内のプライベートヘッダファイルを美しくincludeしたい(#include "../include/helper.h" vs "helper.h")  
    - 現状のままで良いかも
- `time_utils.cpp`に関して、interfaceヘッダファイルのinclude方法 -> publicにすれば良い？ or ファイル単体を指定したtarget_include~  
- `matrix_generator.h`の分割  
- プロジェクト内のソースコードについて、libsgemmの外(`run.cpp`など)から、libsgemmのヘッダファイルをincludeする方法  
    - 下記のinstallディレクトリ構造を変更して対応するなど。
- install先で ヘッダファイルをディレクトリごとに分ける
    - (`cpu`, `cuda`, `utils`) + generalなincludeファイル(`kernels.h`)を作成するなど。  

- vscodeでCUDAの記法に対するエラーが発生するので、その対応  
- vscodeでCMake extension設定  