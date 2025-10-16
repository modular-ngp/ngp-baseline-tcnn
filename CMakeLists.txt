cmake_minimum_required(VERSION 3.26)
project(ngp-baseline-tcnn VERSION 1.0.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 75;86 CACHE STRING "CUDA archs" FORCE)
set(TCNN_CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}" CACHE STRING "" FORCE)
list(GET CMAKE_CUDA_ARCHITECTURES 0 _min_arch)
set(TCNN_MIN_GPU_ARCH "${_min_arch}" CACHE STRING "" FORCE)

option(INSTANTNGP_BUILD_TESTS "Build Instant-NGP tests" OFF)

include(FetchContent)
set(TCNN_BUILD_BENCHMARK OFF CACHE BOOL "" FORCE)
set(TCNN_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(TCNN_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(TCNN_BUILD_WITH_RTC ON CACHE BOOL "" FORCE)
FetchContent_Declare(
        tiny-cuda-nn
        GIT_REPOSITORY https://github.com/NVlabs/tiny-cuda-nn.git
        GIT_TAG master
)
FetchContent_MakeAvailable(tiny-cuda-nn)
#target_link_libraries(ngp-baseline-tcnn PRIVATE tiny-cuda-nn)

FetchContent_Declare(
        dataset
        GIT_REPOSITORY https://github.com/modular-ngp/dataset.git
        GIT_TAG master
)
FetchContent_MakeAvailable(dataset)
#target_link_libraries(ngp-baseline-tcnn PRIVATE dataset::dataset)

add_library(instantngp_core STATIC
        src/core/config.cpp
)
target_include_directories(instantngp_core PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_compile_features(instantngp_core PUBLIC cxx_std_23)

add_executable(ngp-baseline-tcnn
        src/main.cpp
)
target_compile_features(ngp-baseline-tcnn PRIVATE cxx_std_23)
target_link_libraries(ngp-baseline-tcnn PRIVATE instantngp_core)
