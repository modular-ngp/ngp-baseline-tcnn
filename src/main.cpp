#include "instantngp/core/config.hpp"
#include "instantngp/core/math.hpp"
#include "instantngp/io/hostpack_loader.hpp"

#include <cstdlib>
#include <filesystem>
#include <utility>

int main(int argc, char** argv) {
    if (argc > 1) {
        const std::filesystem::path config_path{argv[1]};
        auto document = instantngp::config::Document::from_file(config_path.string());
        if (!document.has_value()) {
            return EXIT_FAILURE;
        }
    }

    if (argc > 2) {
        auto loader_result = instantngp::io::HostpackLoader::create(argv[2]);
        if (!loader_result.has_value()) {
            return EXIT_FAILURE;
        }
        auto loader = std::move(loader_result.value());
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
