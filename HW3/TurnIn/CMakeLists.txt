cmake_minimum_required(VERSION 3.5)
project(HW3)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES
    jacobi.cu
    kernel.cu
    kernel.h
    kernel1.cu
    kernel1.h
    Makefile)

add_executable(HW3 ${SOURCE_FILES})