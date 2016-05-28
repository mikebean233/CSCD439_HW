cmake_minimum_required(VERSION 3.5)
project(HW5)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES main.cpp)
add_executable(HW5 ${SOURCE_FILES})