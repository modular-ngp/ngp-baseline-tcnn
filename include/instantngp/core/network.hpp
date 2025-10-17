#pragma once

#include "instantngp/core/config.hpp"
#include "instantngp/core/expected_compat.hpp"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <tiny-cuda-nn/network_with_input_encoding.h>

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>

namespace instantngp::core {

struct NetworkError {
    std::string message;
};

// Typed config for Phase 2 network core
struct GridEncodingConfig {
    std::uint32_t n_levels{16};
    std::uint32_t n_features_per_level{2};
    std::uint32_t log2_hashmap_size{19};
    std::uint32_t base_resolution{16};
    float per_level_scale{2.0F};
    std::string type{"Hash"}; // Hash | Dense | Tiled
};

struct MlpHeadConfig {
    std::string otype{"FullyFusedMLP"}; // FullyFusedMLP | CutlassMLP
    std::uint32_t n_neurons{64};
    std::uint32_t n_hidden_layers{2};
    std::string activation{"ReLU"};
    std::string output_activation{"None"};
};

struct DirEncodingConfig {
    // Directional encoding for color head
    std::string otype{"SphericalHarmonics"};
    std::uint32_t degree{4};
};

struct NetworkOptions {
    GridEncodingConfig pos_encoding{};
    MlpHeadConfig density_head{}; // outputs: density_features + 1
    MlpHeadConfig color_head{};   // outputs: 3
    DirEncodingConfig dir_encoding{};
    std::uint32_t density_n_features{16}; // features from density head forwarded to color head
};

// Parse typed options from the internal JSON Document
instantngp::expected<NetworkOptions, NetworkError> parse_network_options(const config::Document& doc);

// Wrapper holding two heads as required by Phase 2.
class NetworkCore {
public:
    using ComputeT = __half;

    NetworkCore() = default;
    static instantngp::expected<NetworkCore, NetworkError> create(const NetworkOptions& opts);

    NetworkCore(const NetworkCore&) = delete;
    NetworkCore& operator=(const NetworkCore&) = delete;
    NetworkCore(NetworkCore&&) noexcept;
    NetworkCore& operator=(NetworkCore&&) noexcept;
    ~NetworkCore();

    [[nodiscard]] const NetworkOptions& options() const noexcept { return options_; }

    // Inference helpers (no allocations besides temporary GPUMatrix internal to tcnn)
    // positions: (3 x N) device layout (rows=dims, cols=batch)
    // out_density: (N) device pointer, optional can be nullptr
    // out_features: (N x density_n_features) row-major on device, optional
    instantngp::expected<void, NetworkError> infer_density(cudaStream_t stream,
                                                    const float* positions_device,
                                                    std::uint32_t n,
                                                    float* out_density_device,
                                                    float* out_features_device) const;

    // directions: (3 x N), features: (density_n_features x N)
    // out_rgb: (3 x N)
    instantngp::expected<void, NetworkError> infer_color(cudaStream_t stream,
                                                  const float* directions_device,
                                                  const float* features_device,
                                                  std::uint32_t n,
                                                  float* out_rgb_device) const;

    // Accessors to underlying tcnn models (for training / checkpoint)
    std::shared_ptr<tcnn::NetworkWithInputEncoding<ComputeT>> density_model() const { return density_model_; }
    std::shared_ptr<tcnn::NetworkWithInputEncoding<ComputeT>> color_model() const { return color_model_; }

private:
    explicit NetworkCore(NetworkOptions opts,
                         std::shared_ptr<tcnn::NetworkWithInputEncoding<ComputeT>> density,
                         std::shared_ptr<tcnn::NetworkWithInputEncoding<ComputeT>> color);

    void reset() noexcept;
    void swap(NetworkCore& other) noexcept;

    NetworkOptions options_{};
    std::shared_ptr<tcnn::NetworkWithInputEncoding<ComputeT>> density_model_{};
    std::shared_ptr<tcnn::NetworkWithInputEncoding<ComputeT>> color_model_{};
};

} // namespace instantngp::core
