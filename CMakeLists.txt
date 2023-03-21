cmake_minimum_required(VERSION 3.21)
project(sgemm_test)

set(CMAKE_CXX_STANDARD 20)
set(INSTALL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/install")

# Add sub project (library)
set(LIB_NAME libsgemm)
set(LIB_BASE_DIR src/libsgemm)

add_subdirectory(${LIB_BASE_DIR})

# set executable sources
set(SOURCES
    src/sgemm/run.cpp
)
# no headers for executable sources.
# set(HEADERS)

# find_package(CUDA REQUIRED)
# include(CheckLanguage)
# check_language(CUDA)

set(EXEC_FILE_NAME run)
add_executable(${EXEC_FILE_NAME}
    ${SOURCES}
    ${HEADERS}
)
# if(CUDA_FOUND)
#     cuda_add_executable(${EXEC_FILE_NAME} ${SOURCES} ${HEADERS})
# endif()

# build executable scripts
target_include_directories(${EXEC_FILE_NAME} PUBLIC ${LIB_BASE_DIR})
target_link_libraries(${EXEC_FILE_NAME} ${LIB_NAME})

# install binary
install(TARGETS ${EXEC_FILE_NAME} RUNTIME DESTINATION "${INSTALL_DIR}/bin")
