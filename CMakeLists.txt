cmake_minimum_required(VERSION 3.26)
project(ngp-baseline-tcnn VERSION 1.0.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_executable(ngp-baseline-tcnn main.cpp)


include(FetchContent)
FetchContent_Declare(
        tiny-cuda-nn
        GIT_REPOSITORY https://github.com/NVlabs/tiny-cuda-nn.git
        GIT_TAG master
)
FetchContent_MakeAvailable(tiny-cuda-nn)
target_link_libraries(ngp-baseline-tcnn PRIVATE tiny-cuda-nn)

include(FetchContent)
FetchContent_Declare(
        dataset
        GIT_REPOSITORY https://github.com/modular-ngp/dataset.git
        GIT_TAG master
)
FetchContent_MakeAvailable(dataset)
target_link_libraries(ngp-baseline-tcnn PRIVATE dataset::dataset)
