cmake_minimum_required(VERSION 3.26)
project(ngp-baseline-tcnn VERSION 1.0.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

# Modern CUDA architectures - 75 (Turing), 86 (Ampere), 89 (Ada Lovelace)
set(CMAKE_CUDA_ARCHITECTURES 75;86;89 CACHE STRING "CUDA archs" FORCE)
set(TCNN_CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}" CACHE STRING "" FORCE)
list(GET CMAKE_CUDA_ARCHITECTURES 0 _min_arch)
set(TCNN_MIN_GPU_ARCH "${_min_arch}" CACHE STRING "" FORCE)

# Enable modern CUDA features
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_RESOLVE_DEVICE_SYMBOLS ON)

option(INSTANTNGP_BUILD_TESTS "Build Instant-NGP tests" OFF)

include(FetchContent)

# 虽然 project() 中已声明 LANGUAGES CUDA，但仍需要 find_package(CUDAToolkit)
# 原因：
# 1. LANGUAGES CUDA 只启用 nvcc 编译器，不导入 CUDA 库目标
# 2. find_package(CUDAToolkit) 提供现代 CMake 导入目标：CUDA::cudart, CUDA::nvrtc 等
# 3. 我们需要显式链接 CUDA Runtime 库来使用 cudaMallocAsync 等 API
find_package(CUDAToolkit 13.0 REQUIRED)

# Tiny-CUDA-NN configuration
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

# Dataset library
FetchContent_Declare(
        dataset
        GIT_REPOSITORY https://github.com/modular-ngp/dataset.git
        GIT_TAG master
)
FetchContent_MakeAvailable(dataset)

add_library(instantngp_core STATIC
        src/core/config.cpp
        src/io/hostpack_loader.cpp
        include/instantngp/core/device_buffer.hpp
        include/instantngp/core/cuda_utils.hpp
)
target_include_directories(instantngp_core PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_compile_features(instantngp_core PUBLIC cxx_std_23)
target_link_libraries(instantngp_core
        PUBLIC
        # CUDA Runtime API - 必须 PUBLIC，因为 device_buffer.hpp 是 header-only 模板
        # 用户代码会直接调用 cudaMallocAsync, cudaMemcpyAsync 等函数
        CUDA::cudart
        PRIVATE
        dataset::dataset
        tiny-cuda-nn)

# Modern CUDA compile options
target_compile_options(instantngp_core PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        --use_fast_math
        --extended-lambda
        >
)

add_executable(ngp-baseline-tcnn
        src/main.cpp
)
target_compile_features(ngp-baseline-tcnn PRIVATE cxx_std_23)
target_link_libraries(ngp-baseline-tcnn PRIVATE instantngp_core)
