#include "instantngp/core/occupancy_grid.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>

namespace instantngp::core {

namespace {

__device__ inline std::uint32_t flatten3D(std::uint32_t x, std::uint32_t y, std::uint32_t z, std::uint32_t res) {
    return (z * res + y) * res + x;
}

__device__ inline std::uint32_t clampu(std::uint32_t v, std::uint32_t lo, std::uint32_t hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

__device__ inline std::uint32_t float_to_uint(float v, float minv, float maxv, std::uint32_t res) {
    const float t = (v - minv) / (maxv - minv);
    const float u = fminf(fmaxf(t, 0.0f), 0.999999f);
    return static_cast<std::uint32_t>(u * res);
}

__device__ inline float atomicMaxFloat(float* addr, float val) {
    int* address_as_i = (int*)addr;
    int old = *address_as_i, assumed;
    if (__int_as_float(old) >= val) return __int_as_float(old);
    do {
        assumed = old;
        old = atomicCAS(address_as_i, assumed, __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__global__ void kernel_clear(float* data, std::size_t n, float value) {
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] = value;
}

__global__ void kernel_decay(float* ema, std::size_t n, float factor) {
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) ema[i] *= factor;
}

__global__ void kernel_scatter_max(
    const float* positions, // rows=3, cols=n
    const float* sigmas,    // length n
    std::uint32_t n,
    std::uint32_t res,
    float minx, float miny, float minz,
    float maxx, float maxy, float maxz,
    float* tmp_max // length res^3
) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    // positions in rows-first (3 x n)
    const float px = positions[0 + i * 3];
    const float py = positions[1 + i * 3];
    const float pz = positions[2 + i * 3];

    const std::uint32_t gx = clampu(float_to_uint(px, minx, maxx, res), 0u, res - 1u);
    const std::uint32_t gy = clampu(float_to_uint(py, miny, maxy, res), 0u, res - 1u);
    const std::uint32_t gz = clampu(float_to_uint(pz, minz, maxz, res), 0u, res - 1u);
    const std::uint32_t idx = flatten3D(gx, gy, gz, res);

    atomicMaxFloat(&tmp_max[idx], sigmas[i]);
}

__global__ void kernel_update_ema_and_bits(
    float* ema, const float* tmp_max, std::size_t n_cells, float alpha, float threshold, std::uint8_t* bitfield
) {
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_cells) return;
    const float m = tmp_max[i];
    float e = ema[i];
    e = e * (1.0f - alpha) + alpha * m;
    ema[i] = e;
    const bool occ = e >= threshold;
    const std::size_t byte_idx = i >> 3; // /8
    const std::uint8_t bit = 1u << (i & 7u);
    if (occ) {
        atomicOr((unsigned int*)&bitfield[byte_idx], bit);
    } else {
        atomicAnd((unsigned int*)&bitfield[byte_idx], (std::uint8_t)~bit);
    }
}

} // namespace

instantngp::expected<OccupancyGrid, OccupancyError> OccupancyGrid::create(const io::SceneBounds& bounds,
                                                                   const OccupancyGridConfig& cfg,
                                                                   cudaStream_t stream) {
    OccupancyGrid grid;
    grid.bounds_ = bounds;
    grid.resolution_ = cfg.resolution;
    const std::size_t n_cells = static_cast<std::size_t>(cfg.resolution) * cfg.resolution * cfg.resolution;
    const std::size_t n_bytes = (n_cells + 7) / 8;
    grid.voxel_size_x_ = (bounds.max.x - bounds.min.x) / static_cast<float>(cfg.resolution);
    grid.voxel_size_y_ = (bounds.max.y - bounds.min.y) / static_cast<float>(cfg.resolution);
    grid.voxel_size_z_ = (bounds.max.z - bounds.min.z) / static_cast<float>(cfg.resolution);

    if (auto r = grid.ema_.resize(n_cells, stream); !r) {
        return instantngp::unexpected(OccupancyError{.message = "Failed to allocate EMA buffer"});
    }
    if (auto r = grid.bitfield_.resize(n_bytes, stream); !r) {
        return instantngp::unexpected(OccupancyError{.message = "Failed to allocate bitfield buffer"});
    }
    if (auto r = grid.tmp_max_.resize(n_cells, stream); !r) {
        return instantngp::unexpected(OccupancyError{.message = "Failed to allocate tmp_max buffer"});
    }

    const dim3 blk{256};
    const dim3 grd_ema{static_cast<unsigned>((n_cells + blk.x - 1) / blk.x)};
    kernel_clear<<<grd_ema, blk, 0, stream>>>(grid.ema_.data(), n_cells, 0.0f);
    const dim3 grd_bits{static_cast<unsigned>((n_bytes + blk.x - 1) / blk.x)};
    // Clear bitfield via cudaMemsetAsync
    cudaMemsetAsync(grid.bitfield_.data(), 0, n_bytes, stream);
    return grid;
}

instantngp::expected<void, OccupancyError> OccupancyGrid::decay(float factor, cudaStream_t stream) {
    const std::size_t n_cells = ema_.size();
    const dim3 blk{256};
    const dim3 grd{static_cast<unsigned>((n_cells + blk.x - 1) / blk.x)};
    kernel_decay<<<grd, blk, 0, stream>>>(ema_.data(), n_cells, factor);
    return {};
}

instantngp::expected<void, OccupancyError> OccupancyGrid::update(
    cudaStream_t stream,
    const float* positions_device,
    const float* sigmas_device,
    std::uint32_t n_samples,
    float alpha,
    float threshold
) {
    const std::size_t n_cells = ema_.size();
    const std::size_t n_bytes = bitfield_.size();
    // Clear tmp_max
    const dim3 blk{256};
    const dim3 grd_tmp{static_cast<unsigned>((n_cells + blk.x - 1) / blk.x)};
    kernel_clear<<<grd_tmp, blk, 0, stream>>>(tmp_max_.data(), n_cells, 0.0f);

    // Scatter maxima
    const dim3 grd_scatter{static_cast<unsigned>((n_samples + blk.x - 1) / blk.x)};
    kernel_scatter_max<<<grd_scatter, blk, 0, stream>>>(
        positions_device,
        sigmas_device,
        n_samples,
        resolution_,
        bounds_.min.x, bounds_.min.y, bounds_.min.z,
        bounds_.max.x, bounds_.max.y, bounds_.max.z,
        tmp_max_.data());

    // Update EMA and bits
    const dim3 grd_update{static_cast<unsigned>((n_cells + blk.x - 1) / blk.x)};
    kernel_update_ema_and_bits<<<grd_update, blk, 0, stream>>>(
        ema_.data(), tmp_max_.data(), n_cells, alpha, threshold, bitfield_.data());
    return {};
}

void OccupancyGrid::reset() noexcept {
    ema_.reset();
    bitfield_.reset();
    tmp_max_.reset();
    resolution_ = 0;
    bounds_ = {};
    voxel_size_x_ = voxel_size_y_ = voxel_size_z_ = 0.0F;
}

void OccupancyGrid::swap(OccupancyGrid& other) noexcept {
    std::swap(bounds_, other.bounds_);
    std::swap(resolution_, other.resolution_);
    std::swap(voxel_size_x_, other.voxel_size_x_);
    std::swap(voxel_size_y_, other.voxel_size_y_);
    std::swap(voxel_size_z_, other.voxel_size_z_);
    std::swap(ema_, other.ema_);
    std::swap(bitfield_, other.bitfield_);
    std::swap(tmp_max_, other.tmp_max_);
}

} // namespace instantngp::core
