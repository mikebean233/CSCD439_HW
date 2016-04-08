cmake_minimum_required(VERSION 3.5)
project(HW1)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES
    src/main.c
    src/main.h
    src/linkedList.h
    src/linkedList.c
    )

add_executable(HW1 ${SOURCE_FILES})