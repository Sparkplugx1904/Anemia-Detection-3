cmake_minimum_required(VERSION 3.22)
project("anedet_native")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -flto -D__ARM_NEON")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Ofast -flto")

add_library(
    anedet_native
    SHARED
    native_bridge.cpp
    segment_processor.cpp
    image_utils.cpp
)

find_library(log-lib log)
target_link_libraries(
    anedet_native
    ${log-lib}
)
