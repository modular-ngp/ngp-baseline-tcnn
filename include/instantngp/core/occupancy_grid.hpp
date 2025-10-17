#pragma once

#include "instantngp/core/device_buffer.hpp"
#include "instantngp/core/expected_compat.hpp"
#include "instantngp/io/hostpack_loader.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <string>

namespace instantngp::core {

struct OccupancyGridConfig {
    std::uint32_t resolution{128};
    float ema_decay{0.95F};
    float threshold{0.01F};
};

struct OccupancyError { std::string message; };

class OccupancyGrid {
public:
    OccupancyGrid() = default;
    static instantngp::expected<OccupancyGrid, OccupancyError> create(const io::SceneBounds& bounds,
                                                               const OccupancyGridConfig& cfg,
                                                               cudaStream_t stream = nullptr);

    OccupancyGrid(const OccupancyGrid&) = delete;
    OccupancyGrid& operator=(const OccupancyGrid&) = delete;
    OccupancyGrid(OccupancyGrid&& other) noexcept { swap(other); }
    OccupancyGrid& operator=(OccupancyGrid&& other) noexcept { if (this!=&other){ reset(); swap(other);} return *this; }
    ~OccupancyGrid() { reset(); }

    [[nodiscard]] std::uint32_t resolution() const noexcept { return resolution_; }
    [[nodiscard]] const io::SceneBounds& bounds() const noexcept { return bounds_; }
    [[nodiscard]] float voxel_size_x() const noexcept { return voxel_size_x_; }
    [[nodiscard]] float voxel_size_y() const noexcept { return voxel_size_y_; }
    [[nodiscard]] float voxel_size_z() const noexcept { return voxel_size_z_; }

    [[nodiscard]] core::DeviceBuffer<float>& ema() noexcept { return ema_; }
    [[nodiscard]] const core::DeviceBuffer<float>& ema() const noexcept { return ema_; }
    [[nodiscard]] core::DeviceBuffer<std::uint8_t>& bitfield() noexcept { return bitfield_; }
    [[nodiscard]] const core::DeviceBuffer<std::uint8_t>& bitfield() const noexcept { return bitfield_; }

    // Global decay of EMA values
    instantngp::expected<void, OccupancyError> decay(float factor, cudaStream_t stream = nullptr);

    // Accumulate max sigma per cell from samples and update EMA and bitfield.
    // positions_device: (N x 3) float
    // sigmas_device: (N) float
    instantngp::expected<void, OccupancyError> update(cudaStream_t stream,
                                               const float* positions_device,
                                               const float* sigmas_device,
                                               std::uint32_t n_samples,
                                               float alpha,
                                               float threshold);

private:
    void reset() noexcept;
    void swap(OccupancyGrid& other) noexcept;

    io::SceneBounds bounds_{};
    std::uint32_t resolution_{0};
    float voxel_size_x_{0.0F};
    float voxel_size_y_{0.0F};
    float voxel_size_z_{0.0F};
    core::DeviceBuffer<float> ema_{};          // resolution^3
    core::DeviceBuffer<std::uint8_t> bitfield_{}; // ceil(res^3 / 8)
    // scratch buffer for per-cell max sigmas during an update
    core::DeviceBuffer<float> tmp_max_{};
};

} // namespace instantngp::core
