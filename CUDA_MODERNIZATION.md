# CUDA 13.0 现代化升级说明

## 升级概述

本次升级将项目从旧版 CUDA API 全面升级到 CUDA 13.0 的现代化写法，采用最新的最佳实践。

## 主要改进

### 1. CMakeLists.txt 升级

**改进内容：**
- ✅ 明确要求 CUDA Toolkit 13.0+
- ✅ 设置 CUDA C++20 标准
- ✅ 启用 CUDA 分离编译 (CMAKE_CUDA_SEPARABLE_COMPILATION)
- ✅ 启用设备符号解析 (CMAKE_CUDA_RESOLVE_DEVICE_SYMBOLS)
- ✅ 添加现代 CUDA 架构支持：75 (Turing), 86 (Ampere), 89 (Ada Lovelace)
- ✅ 添加现代编译选项：--expt-relaxed-constexpr, --use_fast_math, --extended-lambda
- ✅ 正确链接 CUDA::cudart 和 CUDA::cuda_driver

### 2. device_buffer.hpp 现代化

**旧版问题：**
- ❌ 使用同步的 cudaMalloc/cudaFree
- ❌ 使用同步的 cudaMemcpy
- ❌ 没有流（stream）支持
- ❌ 性能较差，阻塞 CPU

**新版改进：**
- ✅ 使用 `cudaMallocAsync` 替代 `cudaMalloc` - 异步内存分配
- ✅ 使用 `cudaFreeAsync` 替代 `cudaFree` - 异步内存释放
- ✅ 使用 `cudaMemcpyAsync` 替代 `cudaMemcpy` - 异步内存拷贝
- ✅ 完整的 CUDA 流支持，允许并发操作
- ✅ 添加 `copy_to()` 方法用于设备到主机的拷贝
- ✅ 更好的错误处理和同步机制
- ✅ 包含 `<cuda_runtime.h>` 和 `<cuda.h>` 现代头文件

**性能提升：**
- 异步操作允许 CPU 和 GPU 并行工作
- 支持多流并发，提高吞吐量
- 减少 CPU-GPU 同步开销

### 3. cuda_utils.hpp - 全新现代 CUDA 工具类

创建了完整的现代 CUDA 辅助工具库，包含：

**CudaStream 类：**
```cpp
- RAII 管理的 CUDA 流
- 使用 cudaStreamNonBlocking 标志实现并发
- 支持移动语义，不可复制
- 自动同步和销毁
```

**CudaEvent 类：**
```cpp
- RAII 管理的 CUDA 事件
- 用于精确计时和同步
- elapsed_time() 方法测量性能
```

**CudaMemoryPool 类：**
```cpp
- 现代内存池支持（CUDA 11.2+）
- 减少内存分配开销
- 提高性能和内存利用率
```

**错误处理：**
```cpp
- 使用 C++23 std::expected 进行错误处理
- check_cuda_error() 带 std::source_location
- 类型安全的错误传播
```

**设备管理函数：**
```cpp
- get_device_properties() - 查询设备属性
- set_device() / get_device() - 设备选择
- device_synchronize() - 设备同步
- get_device_count() - 获取设备数量
```

### 4. cuda_examples.hpp - 最佳实践示例

创建了完整的使用示例，展示：
- 使用流进行异步操作
- 使用事件进行性能计时
- 查询和显示设备信息
- 多流并发操作
- 现代错误处理模式

## 关键技术特性

### 异步操作 (Async Operations)
```cpp
// 旧版（同步，阻塞）
cudaMalloc(&ptr, size);
cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);

// 新版（异步，非阻塞）
cudaMallocAsync(&ptr, size, stream);
cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, stream);
```

### 流并发 (Stream Concurrency)
```cpp
CudaStream stream1, stream2;
buffer1.copy_from(data1, stream1);  // 并发执行
buffer2.copy_from(data2, stream2);  // 并发执行
```

### 现代错误处理
```cpp
// 使用 C++23 std::expected
auto result = buffer.resize(1024);
if (!result) {
    std::cerr << result.error().message << std::endl;
}
```

### RAII 资源管理
所有 CUDA 资源（流、事件、内存）都使用 RAII 模式自动管理，确保：
- 无内存泄漏
- 异常安全
- 自动清理

## 性能优势

1. **异步操作**：CPU 和 GPU 可以并行工作
2. **流并发**：多个操作可以同时在不同流中执行
3. **内存池**：减少分配开销，提高性能
4. **非阻塞流**：避免与默认流的隐式同步

## 兼容性

- **最低要求**：CUDA 13.0
- **GPU 架构**：支持 Turing (75), Ampere (86), Ada Lovelace (89)
- **C++ 标准**：C++23
- **CUDA C++ 标准**：C++20

## 迁移建议

如果您的代码中有使用旧的 DeviceBuffer：

**旧代码：**
```cpp
DeviceBuffer<float> buffer;
buffer.resize(1024);
buffer.copy_from(host_data);
```

**新代码（无流）：**
```cpp
DeviceBuffer<float> buffer;
buffer.resize(1024);  // 使用默认流
buffer.copy_from(host_data);  // 使用默认流
```

**新代码（带流，推荐）：**
```cpp
CudaStream stream;
DeviceBuffer<float> buffer;
buffer.resize(1024, stream);
buffer.copy_from(host_data, stream);
```

## 编译说明

确保您的系统已安装 CUDA 13.0 或更高版本：
```bash
# 检查 CUDA 版本
nvcc --version

# 使用 CMake 配置
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

## 总结

本次升级完全遵循 CUDA 13.0 的现代最佳实践：
- ✅ 移除所有旧式同步 API
- ✅ 采用异步操作提升性能
- ✅ 使用 RAII 确保资源安全
- ✅ 利用 C++23/C++20 现代特性
- ✅ 完整的类型安全和错误处理
- ✅ 支持最新 GPU 架构

这些改进将显著提升您的 CUDA 应用的性能、可维护性和安全性。

