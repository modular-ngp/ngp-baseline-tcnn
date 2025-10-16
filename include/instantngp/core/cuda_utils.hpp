#pragma once

#include <cuda_runtime.h>
#include <cuda.h>

#include <expected>
#include <string>
#include <source_location>

namespace instantngp::core {

// Modern CUDA error handling with C++23 features
struct CudaError {
    cudaError_t code;
    std::string message;
    std::source_location location;
};

// RAII wrapper for CUDA streams
class CudaStream {
public:
    CudaStream() {
        // Use cudaStreamCreateWithFlags for better performance
        // cudaStreamNonBlocking allows operations in this stream to run concurrently
        // with operations in stream 0 (the default stream)
        cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);
    }

    ~CudaStream() {
        if (stream_ != nullptr) {
            cudaStreamSynchronize(stream_);
            cudaStreamDestroy(stream_);
        }
    }

    // Non-copyable
    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;

    // Movable
    CudaStream(CudaStream&& other) noexcept : stream_(other.stream_) {
        other.stream_ = nullptr;
    }

    CudaStream& operator=(CudaStream&& other) noexcept {
        if (this != &other) {
            if (stream_ != nullptr) {
                cudaStreamSynchronize(stream_);
                cudaStreamDestroy(stream_);
            }
            stream_ = other.stream_;
            other.stream_ = nullptr;
        }
        return *this;
    }

    [[nodiscard]] cudaStream_t get() const noexcept { return stream_; }
    [[nodiscard]] operator cudaStream_t() const noexcept { return stream_; }

    void synchronize() const {
        if (stream_ != nullptr) {
            cudaStreamSynchronize(stream_);
        }
    }

private:
    cudaStream_t stream_{nullptr};
};

// RAII wrapper for CUDA events
class CudaEvent {
public:
    CudaEvent() {
        cudaEventCreateWithFlags(&event_, cudaEventDefault);
    }

    ~CudaEvent() {
        if (event_ != nullptr) {
            cudaEventDestroy(event_);
        }
    }

    // Non-copyable
    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;

    // Movable
    CudaEvent(CudaEvent&& other) noexcept : event_(other.event_) {
        other.event_ = nullptr;
    }

    CudaEvent& operator=(CudaEvent&& other) noexcept {
        if (this != &other) {
            if (event_ != nullptr) {
                cudaEventDestroy(event_);
            }
            event_ = other.event_;
            other.event_ = nullptr;
        }
        return *this;
    }

    [[nodiscard]] cudaEvent_t get() const noexcept { return event_; }
    [[nodiscard]] operator cudaEvent_t() const noexcept { return event_; }

    void record(cudaStream_t stream = nullptr) const {
        cudaEventRecord(event_, stream);
    }

    void synchronize() const {
        if (event_ != nullptr) {
            cudaEventSynchronize(event_);
        }
    }

    [[nodiscard]] float elapsed_time(const CudaEvent& start) const {
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start.get(), event_);
        return ms;
    }

private:
    cudaEvent_t event_{nullptr};
};

// Helper to check CUDA errors with modern C++23 features
inline std::expected<void, CudaError> check_cuda_error(
    cudaError_t code,
    const std::source_location& location = std::source_location::current()
) {
    if (code != cudaSuccess) {
        return std::unexpected(CudaError{
            .code = code,
            .message = std::string(cudaGetErrorString(code)),
            .location = location
        });
    }
    return {};
}

// Get device properties using modern CUDA API
inline std::expected<cudaDeviceProp, CudaError> get_device_properties(int device = 0) {
    cudaDeviceProp prop;
    if (auto result = check_cuda_error(cudaGetDeviceProperties(&prop, device)); !result) {
        return std::unexpected(result.error());
    }
    return prop;
}

// Set device using modern API
inline std::expected<void, CudaError> set_device(int device) {
    return check_cuda_error(cudaSetDevice(device));
}

// Get current device
inline std::expected<int, CudaError> get_device() {
    int device = 0;
    if (auto result = check_cuda_error(cudaGetDevice(&device)); !result) {
        return std::unexpected(result.error());
    }
    return device;
}

// Synchronize device
inline std::expected<void, CudaError> device_synchronize() {
    return check_cuda_error(cudaDeviceSynchronize());
}

// Reset device (useful for profiling and cleanup)
inline std::expected<void, CudaError> device_reset() {
    return check_cuda_error(cudaDeviceReset());
}

// Get number of CUDA devices
inline std::expected<int, CudaError> get_device_count() {
    int count = 0;
    if (auto result = check_cuda_error(cudaGetDeviceCount(&count)); !result) {
        return std::unexpected(result.error());
    }
    return count;
}

// Memory pool support (CUDA 11.2+, optimized in CUDA 13.0)
class CudaMemoryPool {
public:
    CudaMemoryPool() = default;

    static std::expected<CudaMemoryPool, CudaError> create(int device = 0) {
        CudaMemoryPool pool;
        cudaMemPoolProps props = {};
        props.allocType = cudaMemAllocationTypePinned;
        props.handleTypes = cudaMemHandleTypeNone;
        props.location.type = cudaMemLocationTypeDevice;
        props.location.id = device;

        if (auto result = check_cuda_error(cudaMemPoolCreate(&pool.pool_, &props)); !result) {
            return std::unexpected(result.error());
        }
        return pool;
    }

    ~CudaMemoryPool() {
        if (pool_ != nullptr) {
            cudaMemPoolDestroy(pool_);
        }
    }

    // Non-copyable
    CudaMemoryPool(const CudaMemoryPool&) = delete;
    CudaMemoryPool& operator=(const CudaMemoryPool&) = delete;

    // Movable
    CudaMemoryPool(CudaMemoryPool&& other) noexcept : pool_(other.pool_) {
        other.pool_ = nullptr;
    }

    CudaMemoryPool& operator=(CudaMemoryPool&& other) noexcept {
        if (this != &other) {
            if (pool_ != nullptr) {
                cudaMemPoolDestroy(pool_);
            }
            pool_ = other.pool_;
            other.pool_ = nullptr;
        }
        return *this;
    }

    [[nodiscard]] cudaMemPool_t get() const noexcept { return pool_; }

private:
    cudaMemPool_t pool_{nullptr};
};

} // namespace instantngp::core

