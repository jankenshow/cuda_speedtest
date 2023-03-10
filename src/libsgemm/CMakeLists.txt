cmake_minimum_required(VERSION 3.20)
project(libsgemm LANGUAGES CXX CUDA)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(CUDA REQUIRED)
include(CheckLanguage)
check_language(CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CUDA_COMPUTE_CAPABILITY 75)

add_compile_options("$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G;-src-in-ptx>")

set(SOURCES
    src/sgemm_naive.cu
    src/sgemm_global_mem_coalesce.cu
    src/sgemm_shared_mem_block.cu
)

set(HEADERS
)

if(CUDA_FOUND)
    message("CUDA version : " ${CUDA_VERSION})
    CUDA_ADD_LIBRARY(libsgemm STATIC
        SOURCES
        HEADERS
    )
end_if()