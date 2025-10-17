// Phase 2: Encoding & Network Core

#include "instantngp/core/network.hpp"

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/encoding.h>
#include <tiny-cuda-nn/network.h>
#include <tiny-cuda-nn/network_with_input_encoding.h>

#include <utility>

using tcnn_json = tcnn::json;

namespace instantngp::core {

namespace {

template <typename T>
std::string to_string_num(T v) {
    if constexpr (std::is_floating_point_v<T>) {
        return std::to_string(v);
    } else {
        return std::to_string(static_cast<long long>(v));
    }
}

} // namespace

instantngp::expected<NetworkOptions, NetworkError> parse_network_options(const config::Document& doc) {
    NetworkOptions out{};

    const config::Value& root = doc.root();
    if (!root.is_object()) {
        return instantngp::unexpected(NetworkError{.message = "Config root must be an object"});
    }

    const auto* network = root.find("network");
    if (!network || !network->is_object()) {
        // Defaults
        return out;
    }

    const auto* density_features = network->find("density_n_features");
    if (density_features && density_features->is_number()) {
        out.density_n_features = static_cast<std::uint32_t>(density_features->as_number());
    }

    // pos_encoding
    if (const auto* enc = network->find("pos_encoding"); enc && enc->is_object()) {
        if (const auto* v = enc->find("n_levels"); v && v->is_number()) out.pos_encoding.n_levels = static_cast<std::uint32_t>(v->as_number());
        if (const auto* v = enc->find("n_features_per_level"); v && v->is_number()) out.pos_encoding.n_features_per_level = static_cast<std::uint32_t>(v->as_number());
        if (const auto* v = enc->find("log2_hashmap_size"); v && v->is_number()) out.pos_encoding.log2_hashmap_size = static_cast<std::uint32_t>(v->as_number());
        if (const auto* v = enc->find("base_resolution"); v && v->is_number()) out.pos_encoding.base_resolution = static_cast<std::uint32_t>(v->as_number());
        if (const auto* v = enc->find("per_level_scale"); v && v->is_number()) out.pos_encoding.per_level_scale = static_cast<float>(v->as_number());
        if (const auto* v = enc->find("type"); v && v->is_string()) out.pos_encoding.type = v->as_string();
    }

    auto parse_head = [](const config::Value* obj, MlpHeadConfig& dst) {
        if (!obj || !obj->is_object()) return;
        if (const auto* v = obj->find("otype"); v && v->is_string()) dst.otype = v->as_string();
        if (const auto* v = obj->find("n_neurons"); v && v->is_number()) dst.n_neurons = static_cast<std::uint32_t>(v->as_number());
        if (const auto* v = obj->find("n_hidden_layers"); v && v->is_number()) dst.n_hidden_layers = static_cast<std::uint32_t>(v->as_number());
        if (const auto* v = obj->find("activation"); v && v->is_string()) dst.activation = v->as_string();
        if (const auto* v = obj->find("output_activation"); v && v->is_string()) dst.output_activation = v->as_string();
    };

    parse_head(network->find("density_head"), out.density_head);
    parse_head(network->find("color_head"), out.color_head);

    if (const auto* dir = network->find("dir_encoding"); dir && dir->is_object()) {
        if (const auto* v = dir->find("otype"); v && v->is_string()) out.dir_encoding.otype = v->as_string();
        if (const auto* v = dir->find("degree"); v && v->is_number()) out.dir_encoding.degree = static_cast<std::uint32_t>(v->as_number());
    }

    return out;
}

static tcnn_json build_grid_json(const GridEncodingConfig& cfg) {
    tcnn_json j;
    j["otype"] = "Grid";
    j["type"] = cfg.type;
    j["n_levels"] = cfg.n_levels;
    j["n_features_per_level"] = cfg.n_features_per_level;
    j["log2_hashmap_size"] = cfg.log2_hashmap_size;
    j["base_resolution"] = cfg.base_resolution;
    j["per_level_scale"] = cfg.per_level_scale;
    return j;
}

static tcnn_json build_mlp_json(const MlpHeadConfig& cfg, std::uint32_t input_dims, std::uint32_t output_dims) {
    tcnn_json j;
    j["otype"] = cfg.otype;
    j["n_input_dims"] = input_dims;  // will be overridden by NetworkWithInputEncoding constructor
    j["n_output_dims"] = output_dims; // same
    j["n_neurons"] = cfg.n_neurons;
    j["n_hidden_layers"] = cfg.n_hidden_layers;
    j["activation"] = cfg.activation;
    j["output_activation"] = cfg.output_activation;
    return j;
}

static tcnn_json build_dir_composite_json(const DirEncodingConfig& dir, std::uint32_t feat_dims) {
    tcnn_json identity;
    identity["otype"] = "Identity";
    identity["n_dims_to_encode"] = feat_dims;
    identity["dims_to_encode_begin"] = 0;

    tcnn_json nested_dir;
    nested_dir["otype"] = dir.otype;
    nested_dir["degree"] = dir.degree;
    nested_dir["n_dims_to_encode"] = 3;
    nested_dir["dims_to_encode_begin"] = feat_dims;

    tcnn_json comp;
    comp["otype"] = "Composite";
    comp["nested"] = tcnn_json::array({identity, nested_dir});
    return comp;
}

NetworkCore::NetworkCore(NetworkOptions opts,
                         std::shared_ptr<tcnn::NetworkWithInputEncoding<ComputeT>> density,
                         std::shared_ptr<tcnn::NetworkWithInputEncoding<ComputeT>> color)
    : options_{std::move(opts)}, density_model_{std::move(density)}, color_model_{std::move(color)} {}

NetworkCore::~NetworkCore() { reset(); }

NetworkCore::NetworkCore(NetworkCore&& other) noexcept { swap(other); }

NetworkCore& NetworkCore::operator=(NetworkCore&& other) noexcept { if (this != &other) { reset(); swap(other);} return *this; }

void NetworkCore::reset() noexcept {
    density_model_.reset();
    color_model_.reset();
}

void NetworkCore::swap(NetworkCore& other) noexcept {
    std::swap(options_, other.options_);
    std::swap(density_model_, other.density_model_);
    std::swap(color_model_, other.color_model_);
}

instantngp::expected<NetworkCore, NetworkError> NetworkCore::create(const NetworkOptions& opts) {
    try {
        // Density: position grid encoding + MLP -> (feat + 1)
        const auto grid_json = build_grid_json(opts.pos_encoding);
        const std::uint32_t density_out = opts.density_n_features + 1;
        const auto density_mlp = build_mlp_json(opts.density_head, /*in=*/0, density_out);

        auto density_model = std::make_shared<tcnn::NetworkWithInputEncoding<ComputeT>>(3u, density_out, grid_json, density_mlp);

        // Color: composite encoding (features identity + dir encoding) + MLP -> 3
        const auto dir_comp = build_dir_composite_json(opts.dir_encoding, opts.density_n_features);
        const auto color_mlp = build_mlp_json(opts.color_head, /*in=*/0, /*out=*/3);
        auto color_model = std::make_shared<tcnn::NetworkWithInputEncoding<ComputeT>>(opts.density_n_features + 3u, 3u, dir_comp, color_mlp);

        return NetworkCore{opts, std::move(density_model), std::move(color_model)};
    } catch (const std::exception& e) {
        return instantngp::unexpected(NetworkError{.message = std::string{"Failed to create NetworkCore: "} + e.what()});
    }
}

instantngp::expected<void, NetworkError> NetworkCore::infer_density(
    cudaStream_t stream,
    const float* positions_device,
    std::uint32_t n,
    float* out_density_device,
    float* out_features_device
) const {
    try {
        if (!density_model_) {
            return instantngp::unexpected(NetworkError{.message = "Density model not initialized"});
        }
        // Expect positions_device laid out as rows=3, cols=n
        tcnn::GPUMatrixDynamic<float> in(const_cast<float*>(positions_device), /*rows=*/3u, /*cols=*/n);
        const std::uint32_t out_dims = options_.density_n_features + 1u;

        if (out_density_device || out_features_device) {
            // Caller supplies a single packed buffer (out_dims x n). If only one pointer is provided, it must be to the packed buffer.
            tcnn::GPUMatrixDynamic<ComputeT> out(nullptr, out_dims, n);
            if (out_density_device && !out_features_device) {
                out.set((ComputeT*)out_density_device, out_dims, n);
            } else if (out_features_device && !out_density_device) {
                out.set((ComputeT*)out_features_device, out_dims, n);
            } else {
                // Both provided separately is not supported in this thin wrapper.
                return instantngp::unexpected(NetworkError{.message = "Provide either a single packed output buffer or nullptrs"});
            }
            density_model_->inference_mixed_precision(stream, in, out, true);
        } else {
            // No outputs provided; compute into temporary and discard (useful for warmup)
            tcnn::GPUMatrixDynamic<ComputeT> out(out_dims, n, stream);
            density_model_->inference_mixed_precision(stream, in, out, true);
        }
        return {};
    } catch (const std::exception& e) {
        return instantngp::unexpected(NetworkError{.message = std::string{"infer_density failed: "} + e.what()});
    }
}

instantngp::expected<void, NetworkError> NetworkCore::infer_color(
    cudaStream_t stream,
    const float* directions_device,
    const float* features_device,
    std::uint32_t n,
    float* out_rgb_device
) const {
    try {
        if (!color_model_) {
            return instantngp::unexpected(NetworkError{.message = "Color model not initialized"});
        }

        // Expect directions and features already arranged as rows-first (dims x N)
        const std::uint32_t in_dims = options_.density_n_features + 3u;
        tcnn::GPUMatrixDynamic<float> input(nullptr, in_dims, n);
        // We require caller to pass a pre-concatenated buffer at features_device for full input.
        input.set(const_cast<float*>(features_device), in_dims, n);

        tcnn::GPUMatrixDynamic<ComputeT> out(nullptr, /*rows=*/3u, /*cols=*/n);
        if (out_rgb_device) {
            out.set((ComputeT*)out_rgb_device, 3u, n);
        } else {
            out = tcnn::GPUMatrixDynamic<ComputeT>(3u, n, stream);
        }

        color_model_->inference_mixed_precision(stream, input, out, true);
        return {};
    } catch (const std::exception& e) {
        return instantngp::unexpected(NetworkError{.message = std::string{"infer_color failed: "} + e.what()});
    }
}

} // namespace instantngp::core
