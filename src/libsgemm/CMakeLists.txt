cmake_minimum_required(VERSION 3.21)
project(${LIB_NAME} LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CUDA_COMPUTE_CAPABILITY 75)
add_compile_options("$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G;-src-in-ptx>")

find_package(CUDA REQUIRED)
include(CheckLanguage)
check_language(CUDA)

set(LIB_TYPE "STATIC")
if(${LIB_TYPE} MATCHES "STATIC")
  set(LIB_TYPE_ARC "ARCHIVE")
else()
  set(LIB_TYPE_ARC "LIBRARY")
endif()

file(GLOB_RECURSE LIB_SOURCES
    RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    *.cu
    *.cpp
)
file(GLOB LIB_PRIVATE_HEADERS 
    RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    src/include/*.h
    src/include/**/*.h
)
file(GLOB LIB_HEADERS
    RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    include/**/*.h
)

message(LIB_SOURCES "${LIB_SOURCES}")
message(LIB_PRIVATE_HEADERS "${LIB_PRIVATE_HEADERS}")
message(LIB_HEADERS "${LIB_HEADERS}")

if(CUDA_FOUND)
    # message("CUDA version : " ${CUDA_VERSION})
    cuda_add_library(${LIB_NAME} STATIC
        ${LIB_SOURCES}
        ${LIB_PRIVATE_HEADERS}
    )
    target_include_directories(${LIB_NAME} INTERFACE include)
    # target_include_directories(${LIB_NAME} PRIVATE src/include)
    set_target_properties(${LIB_NAME} PROPERTIES 
        CUDA_ARCHITECTURES 
        ${CUDA_COMPUTE_CAPABILITY})
    # cuda_add_cublas_to_target(${LIB_NAME})
endif()

# install library
install(TARGETS ${LIB_NAME} ${LIB_TYPE_ARC} DESTINATION "${INSTALL_DIR}/lib")
install(FILES ${LIB_HEADERS} DESTINATION "${INSTALL_DIR}/include")
