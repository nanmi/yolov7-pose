cmake_minimum_required(VERSION 3.10)

project(yolov7_pose)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -g -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")
add_definitions(-DAPI_EXPORTS)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_BUILD_TYPE Debug)

# cuda
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# tensorrt
set(TENSORRT_INCLUDE_DIR /usr/local/TensorRT-8.4.1.5/include/)
set(TENSORRT_LIBRARY_DIR /usr/local/TensorRT-8.4.1.5/lib/)
include_directories(${TENSORRT_INCLUDE_DIR})
link_directories(${TENSORRT_LIBRARY_DIR})

# OpenCV
find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})


cuda_add_executable(yolov7_pose yolov7_pose.cpp yololayer.cu)
target_link_libraries(yolov7_pose nvinfer ${CUDA_LIBRARIES} ${OpenCV_LIBS})

add_definitions(-O2 -pthread)

