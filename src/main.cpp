#include "instantngp/core/config.hpp"
#include "instantngp/core/math.hpp"

#include <cstdlib>
#include <filesystem>

int main(int argc, char** argv) {
    if (argc > 1) {
        const std::filesystem::path config_path{argv[1]};
        auto document = instantngp::config::Document::from_file(config_path.string());
        if (!document.has_value()) {
            return EXIT_FAILURE;
        }
    }

    constexpr auto identity = instantngp::math::Mat3x4::identity();
    [[maybe_unused]] constexpr auto translated = instantngp::math::make_translation({1.0F, 0.0F, 0.0F});
    [[maybe_unused]] const auto origin = instantngp::math::Vec3{};
    [[maybe_unused]] const auto point = instantngp::math::transform_point(identity, origin);

    return EXIT_SUCCESS;
}
