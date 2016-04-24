cmake_minimum_required(VERSION 3.5)
project(HW2)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
set(SOURCE_FILES pgmUtility.c)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
add_executable(HW2 ${SOURCE_FILES})