#pragma once

#include "instantngp/core/json_writer.hpp"
#include "instantngp/core/network.hpp"

#include <tiny-cuda-nn/trainer.h>

#include <filesystem>
#include <expected>
#include <fstream>
#include <string>

namespace instantngp::core {

struct CheckpointError { std::string message; };

struct CheckpointMetadataConfig {
    std::uint64_t step{0};
    NetworkOptions network{};
};

// Save a tiny-cuda-nn snapshot (parameters, optionally optimizer) to JSON
template <typename T, typename PARAMS_T, typename COMPUTE_T=T>
std::expected<void, CheckpointError> save_tcnn_snapshot(const std::filesystem::path& path,
                                                        tcnn::Trainer<T, PARAMS_T, COMPUTE_T>& trainer,
                                                        bool with_optimizer = false) {
    try {
        auto j = trainer.serialize(with_optimizer);
        std::ofstream out(path, std::ios::binary);
        if (!out) return std::unexpected(CheckpointError{.message = "Failed to open snapshot path"});
        out << j.dump();
        return {};
    } catch (const std::exception& e) {
        return std::unexpected(CheckpointError{.message = std::string{"Snapshot save failed: "} + e.what()});
    }
}

// Load a tiny-cuda-nn snapshot from JSON
template <typename T, typename PARAMS_T, typename COMPUTE_T=T>
std::expected<void, CheckpointError> load_tcnn_snapshot(const std::filesystem::path& path,
                                                        tcnn::Trainer<T, PARAMS_T, COMPUTE_T>& trainer) {
    try {
        std::ifstream in(path, std::ios::binary);
        if (!in) return std::unexpected(CheckpointError{.message = "Failed to open snapshot for read"});
        tcnn::json j;
        in >> j;
        trainer.deserialize(j);
        return {};
    } catch (const std::exception& e) {
        return std::unexpected(CheckpointError{.message = std::string{"Snapshot load failed: "} + e.what()});
    }
}

// Write a compact metadata JSON using the internal writer
inline std::expected<void, CheckpointError> save_metadata(const std::filesystem::path& path,
                                                          const CheckpointMetadataConfig& meta) {
    std::ofstream out(path, std::ios::binary);
    if (!out) return std::unexpected(CheckpointError{.message = "Failed to open metadata path"});
    JsonWriter w{out, true};
    w.begin_object();
    w.key("step"); w.value(meta.step);
    w.key("network"); {
        JsonWriter::Scope s{w};
        w.key("density_n_features"); w.value(meta.network.density_n_features);
        w.key("pos_encoding"); {
            JsonWriter::Scope s2{w};
            w.key("type"); w.value(meta.network.pos_encoding.type);
            w.key("n_levels"); w.value(meta.network.pos_encoding.n_levels);
            w.key("n_features_per_level"); w.value(meta.network.pos_encoding.n_features_per_level);
            w.key("log2_hashmap_size"); w.value(meta.network.pos_encoding.log2_hashmap_size);
            w.key("base_resolution"); w.value(meta.network.pos_encoding.base_resolution);
            w.key("per_level_scale"); w.value(meta.network.pos_encoding.per_level_scale);
        }
        w.key("density_head"); {
            JsonWriter::Scope s2{w};
            w.key("otype"); w.value(meta.network.density_head.otype);
            w.key("n_neurons"); w.value(meta.network.density_head.n_neurons);
            w.key("n_hidden_layers"); w.value(meta.network.density_head.n_hidden_layers);
            w.key("activation"); w.value(meta.network.density_head.activation);
            w.key("output_activation"); w.value(meta.network.density_head.output_activation);
        }
        w.key("color_head"); {
            JsonWriter::Scope s2{w};
            w.key("otype"); w.value(meta.network.color_head.otype);
            w.key("n_neurons"); w.value(meta.network.color_head.n_neurons);
            w.key("n_hidden_layers"); w.value(meta.network.color_head.n_hidden_layers);
            w.key("activation"); w.value(meta.network.color_head.activation);
            w.key("output_activation"); w.value(meta.network.color_head.output_activation);
        }
        w.key("dir_encoding"); {
            JsonWriter::Scope s2{w};
            w.key("otype"); w.value(meta.network.dir_encoding.otype);
            w.key("degree"); w.value(meta.network.dir_encoding.degree);
        }
    }
    w.end_object();
    return {};
}

} // namespace instantngp::core

