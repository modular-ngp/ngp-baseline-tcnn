#pragma once

#include <cuda_runtime.h>
#include <cuda.h>

#include <cstddef>
#include <expected>
#include <span>
#include <string>
#include <utility>

namespace instantngp::core {

enum class DeviceBufferErrorCode { None = 0, InvalidArgument, AllocationFailed, CopyFailed, StreamError };

struct DeviceBufferError {
    DeviceBufferErrorCode code{DeviceBufferErrorCode::None};
    std::string message{};
};

template <typename T>
class DeviceBuffer {
public:
    using value_type = T;
    using Result = std::expected<void, DeviceBufferError>;

    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept {
        swap(other);
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            reset();
            swap(other);
        }
        return *this;
    }

    ~DeviceBuffer() {
        reset();
    }

    [[nodiscard]] std::size_t size() const noexcept { return count_; }
    [[nodiscard]] std::size_t bytes() const noexcept { return count_ * sizeof(T); }
    [[nodiscard]] bool empty() const noexcept { return device_ptr_ == nullptr; }

    [[nodiscard]] T* data() noexcept { return device_ptr_; }
    [[nodiscard]] const T* data() const noexcept { return device_ptr_; }

    Result resize(std::size_t count, cudaStream_t stream = nullptr) {
        if (count == count_) {
            return {};
        }
        if (count == 0U) {
            reset();
            return {};
        }
        reset();

        void* allocation{nullptr};
        const std::size_t total_bytes = count * sizeof(T);

        // Use modern async allocation API for better performance with streams
        cudaError_t status = cudaMallocAsync(&allocation, total_bytes, stream);
        if (status != cudaSuccess) {
            return std::unexpected(DeviceBufferError{
                    .code = DeviceBufferErrorCode::AllocationFailed,
                    .message = std::string{"cudaMallocAsync failed: "} + cudaGetErrorString(status)});
        }

        // Synchronize to ensure allocation is complete before use
        if (stream != nullptr) {
            status = cudaStreamSynchronize(stream);
            if (status != cudaSuccess) {
                cudaFreeAsync(allocation, stream);
                return std::unexpected(DeviceBufferError{
                        .code = DeviceBufferErrorCode::StreamError,
                        .message = std::string{"cudaStreamSynchronize failed: "} + cudaGetErrorString(status)});
            }
        }

        device_ptr_ = static_cast<T*>(allocation);
        count_ = count;
        stream_ = stream;
        return {};
    }

    Result copy_from(std::span<const T> data, cudaStream_t stream = nullptr) {
        if (data.empty()) {
            reset();
            return {};
        }
        if (data.data() == nullptr) {
            return std::unexpected(DeviceBufferError{
                    .code = DeviceBufferErrorCode::InvalidArgument,
                    .message = "DeviceBuffer::copy_from received null data pointer"});
        }
        if (count_ != data.size()) {
            auto resized = resize(data.size(), stream);
            if (!resized.has_value()) {
                return std::unexpected(resized.error());
            }
        }

        // Use modern async memcpy for better performance
        cudaError_t status = cudaMemcpyAsync(
            device_ptr_,
            data.data(),
            data.size_bytes(),
            cudaMemcpyHostToDevice,
            stream
        );

        if (status != cudaSuccess) {
            return std::unexpected(DeviceBufferError{
                    .code = DeviceBufferErrorCode::CopyFailed,
                    .message = std::string{"cudaMemcpyAsync failed: "} + cudaGetErrorString(status)});
        }

        // Synchronize to ensure copy is complete
        if (stream != nullptr) {
            status = cudaStreamSynchronize(stream);
            if (status != cudaSuccess) {
                return std::unexpected(DeviceBufferError{
                        .code = DeviceBufferErrorCode::StreamError,
                        .message = std::string{"cudaStreamSynchronize failed after copy: "} + cudaGetErrorString(status)});
            }
        }

        return {};
    }

    Result copy_to(std::span<T> data, cudaStream_t stream = nullptr) const {
        if (empty()) {
            return std::unexpected(DeviceBufferError{
                    .code = DeviceBufferErrorCode::InvalidArgument,
                    .message = "DeviceBuffer::copy_to called on empty buffer"});
        }
        if (data.size() < count_) {
            return std::unexpected(DeviceBufferError{
                    .code = DeviceBufferErrorCode::InvalidArgument,
                    .message = "DeviceBuffer::copy_to destination span too small"});
        }

        cudaError_t status = cudaMemcpyAsync(
            data.data(),
            device_ptr_,
            bytes(),
            cudaMemcpyDeviceToHost,
            stream
        );

        if (status != cudaSuccess) {
            return std::unexpected(DeviceBufferError{
                    .code = DeviceBufferErrorCode::CopyFailed,
                    .message = std::string{"cudaMemcpyAsync failed: "} + cudaGetErrorString(status)});
        }

        // Synchronize to ensure copy is complete
        if (stream != nullptr) {
            status = cudaStreamSynchronize(stream);
            if (status != cudaSuccess) {
                return std::unexpected(DeviceBufferError{
                        .code = DeviceBufferErrorCode::StreamError,
                        .message = std::string{"cudaStreamSynchronize failed after copy: "} + cudaGetErrorString(status)});
            }
        }

        return {};
    }

    void reset() noexcept {
        if (device_ptr_ != nullptr) {
            // Use async free for modern CUDA
            cudaFreeAsync(device_ptr_, stream_);
            device_ptr_ = nullptr;
            count_ = 0;
            stream_ = nullptr;
        }
    }

private:
    void swap(DeviceBuffer& other) noexcept {
        std::swap(device_ptr_, other.device_ptr_);
        std::swap(count_, other.count_);
        std::swap(stream_, other.stream_);
    }

    T* device_ptr_{nullptr};
    std::size_t count_{0};
    cudaStream_t stream_{nullptr};
};

} // namespace instantngp::core
