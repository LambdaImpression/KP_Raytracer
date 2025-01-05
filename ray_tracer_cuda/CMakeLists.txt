cmake_minimum_required(VERSION 3.30)
project(ray_tracer_cuda CUDA)

set(CMAKE_CUDA_STANDARD 20)

include_directories(${CMAKE_SOURCE_DIR}/include)

add_executable(ray_tracer_cuda main.cu)

set_target_properties(ray_tracer_cuda PROPERTIES
                      CUDA_SEPARABLE_COMPILATION ON)
