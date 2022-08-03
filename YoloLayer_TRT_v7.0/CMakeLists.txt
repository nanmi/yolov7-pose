cmake_minimum_required(VERSION 3.10)

project(plugin_build_example)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_BUILD_TYPE Release)

# cuda/cudnn
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# # tensorrt
set(TENSORRT_INCLUDE_DIR /usr/local/TensorRT-8.4.1.5/include/)
set(TENSORRT_LIBRARY_DIR /usr/local/TensorRT-8.4.1.5/lib/)
include_directories(${TENSORRT_INCLUDE_DIR})
link_directories(${TENSORRT_LIBRARY_DIR})

cuda_add_library(yolo SHARED ${PROJECT_SOURCE_DIR}/yololayer.cu
                                                    )
target_link_libraries(yolo nvinfer ${CUDA_LIBRARIES})

add_definitions(-O2 -pthread)

