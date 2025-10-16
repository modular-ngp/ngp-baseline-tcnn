// Modern CUDA 13.0 Example Usage
// This file demonstrates best practices for CUDA 13.0

#include "instantngp/core/device_buffer.hpp"
#include "instantngp/core/cuda_utils.hpp"

#include <vector>
#include <iostream>

namespace instantngp::examples {

// Example: Using modern DeviceBuffer with streams
void example_device_buffer_with_streams() {
    using namespace instantngp::core;

    // Create a CUDA stream for async operations
    CudaStream stream;

    // Create host data
    std::vector<float> host_data(1024, 3.14f);

    // Create device buffer and copy data asynchronously
    DeviceBuffer<float> device_buffer;
    auto result = device_buffer.copy_from(host_data, stream.get());

    if (!result) {
        std::cerr << "Failed to copy data: " << result.error().message << std::endl;
        return;
    }

    // The buffer is now on the device and ready to use
    std::cout << "Buffer size: " << device_buffer.size() << " elements" << std::endl;
    std::cout << "Buffer bytes: " << device_buffer.bytes() << " bytes" << std::endl;
}

// Example: Using CUDA events for timing
void example_timing_with_events() {
    using namespace instantngp::core;

    CudaStream stream;
    CudaEvent start, stop;

    // Record start event
    start.record(stream.get());

    // Do some work (example: allocate and copy data)
    DeviceBuffer<float> buffer;
    std::vector<float> data(1000000, 1.0f);
    buffer.copy_from(data, stream.get());

    // Record stop event
    stop.record(stream.get());
    stop.synchronize();

    // Calculate elapsed time
    float ms = stop.elapsed_time(start);
    std::cout << "Operation took: " << ms << " ms" << std::endl;
}

// Example: Query device properties
void example_device_info() {
    using namespace instantngp::core;

    auto device_count = get_device_count();
    if (!device_count) {
        std::cerr << "Failed to get device count" << std::endl;
        return;
    }

    std::cout << "Number of CUDA devices: " << device_count.value() << std::endl;

    for (int i = 0; i < device_count.value(); ++i) {
        auto props = get_device_properties(i);
        if (props) {
            const auto& p = props.value();
            std::cout << "\nDevice " << i << ": " << p.name << std::endl;
            std::cout << "  Compute capability: " << p.major << "." << p.minor << std::endl;
            std::cout << "  Total memory: " << (p.totalGlobalMem / (1024 * 1024)) << " MB" << std::endl;
            std::cout << "  Multiprocessors: " << p.multiProcessorCount << std::endl;
            std::cout << "  Max threads per block: " << p.maxThreadsPerBlock << std::endl;
            std::cout << "  Memory clock rate: " << (p.memoryClockRate / 1000) << " MHz" << std::endl;
            std::cout << "  Memory bus width: " << p.memoryBusWidth << " bits" << std::endl;
            std::cout << "  L2 cache size: " << (p.l2CacheSize / 1024) << " KB" << std::endl;
        }
    }
}

// Example: Multiple streams for concurrent operations
void example_concurrent_streams() {
    using namespace instantngp::core;

    constexpr size_t num_streams = 4;
    constexpr size_t data_size = 1000000;

    // Create multiple streams
    std::vector<CudaStream> streams(num_streams);
    std::vector<DeviceBuffer<float>> buffers(num_streams);

    // Launch operations on different streams
    for (size_t i = 0; i < num_streams; ++i) {
        std::vector<float> data(data_size, static_cast<float>(i));
        buffers[i].copy_from(data, streams[i].get());
    }

    // Synchronize all streams
    for (auto& stream : streams) {
        stream.synchronize();
    }

    std::cout << "Processed " << num_streams << " streams concurrently" << std::endl;
}

// Example: Error handling with std::expected
void example_error_handling() {
    using namespace instantngp::core;

    auto result = set_device(0);
    if (!result) {
        std::cerr << "Failed to set device: "
                  << result.error().message
                  << " at " << result.error().location.file_name()
                  << ":" << result.error().location.line() << std::endl;
        return;
    }

    std::cout << "Successfully set device" << std::endl;
}

} // namespace instantngp::examples

