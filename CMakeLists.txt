cmake_minimum_required(VERSION 3.21)
project(sgemm_test)

# set executable sources
set(SOURCES
    src/sgemm/runner.cpp
)
# no headers for executable sources.
# set(HEADERS)

set(EXEC_FILE_NAME run)
add_executable(${EXEC_FILE_NAME}
    ${SOURCES}
    ${HEADERS}
)

# Add sub project (library)
set(LIB_TYPE "STATIC")
if(${LIB_TYPE} MATCHES "STATIC")
  set(LIB_TYPE_ARC "ARCHIVE")
else()
  set(LIB_TYPE_ARC "LIBRARY")
endif()

set(LIB_NAME libsgemm)
set(LIB_BASE_DIR src/libsgemm)
set(LIB_HEADERS CudaVecDouble.h)
set(LIB_SOURCES CudaVecDouble.cu)

add_subdirectory(${LIB_BASE_DIR})

# build executable scripts
target_include_directories(${EXEC_FILE_NAME} PUBLIC ${LIB_BASE_DIR})
target_link_libraries(${EXEC_FILE_NAME} ${LIB_NAME})

# install library
set(INSTALL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/install")

install(TARGETS ${LIB_NAME} ${LIB_TYPE_ARC} DESTINATION "${INSTALL_DIR}/lib")
install(FILES ${LIB_BASE_DIR}/${LIB_HEADERS} DESTINATION "${INSTALL_DIR}/include")

# install binary
install(TARGETS ${EXEC_FILE_NAME} RUNTIME DESTINATION "${INSTALL_DIR}/bin")
