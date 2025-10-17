#include "instantngp/core/config.hpp"
#include "instantngp/core/math.hpp"
#include "instantngp/core/network.hpp"
#include "instantngp/core/occupancy_grid.hpp"
#include "instantngp/io/hostpack_loader.hpp"

#include <cstdlib>
#include <filesystem>
#include <utility>

int main(int argc, char** argv) {
    std::optional<instantngp::config::Document> config_doc;
    if (argc > 1) {
        const std::filesystem::path config_path{argv[1]};
        auto document = instantngp::config::Document::from_file(config_path.string());
        if (!document.has_value()) {
            return EXIT_FAILURE;
        }
        config_doc = std::move(document.value());
    }

    if (argc > 2) {
        auto loader_result = instantngp::io::HostpackLoader::create(argv[2]);
        if (!loader_result.has_value()) {
            return EXIT_FAILURE;
        }
        auto loader = std::move(loader_result.value());
        // Phase 2: instantiate network + occupancy grid
        instantngp::core::NetworkOptions net_opts{};
        if (config_doc.has_value()) {
            if (auto parsed = instantngp::core::parse_network_options(*config_doc); parsed.has_value()) {
                net_opts = parsed.value();
            }
        }
        auto net = instantngp::core::NetworkCore::create(net_opts);
        if (!net.has_value()) {
            return EXIT_FAILURE;
        }
        auto grid = instantngp::core::OccupancyGrid::create(loader.bounds(), instantngp::core::OccupancyGridConfig{});
        if (!grid.has_value()) {
            return EXIT_FAILURE;
        }
        if (!loader.frames().empty()) {
            [[maybe_unused]] const auto& first_frame = loader.frames().front();
            [[maybe_unused]] const auto image = loader.image_view(first_frame.frame_index);
            [[maybe_unused]] const auto rays = loader.build_ray_batch(first_frame.frame_index);
        }
    }

    constexpr auto identity = instantngp::math::Mat3x4::identity();
    [[maybe_unused]] constexpr auto translated = instantngp::math::make_translation({1.0F, 0.0F, 0.0F});
    [[maybe_unused]] const auto origin = instantngp::math::Vec3{};
    [[maybe_unused]] const auto point = instantngp::math::transform_point(identity, origin);

    return EXIT_SUCCESS;
}
