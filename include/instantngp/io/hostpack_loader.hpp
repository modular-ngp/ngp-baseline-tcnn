#pragma once

#include "instantngp/core/device_buffer.hpp"
#include "instantngp/core/math.hpp"

#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace instantngp::io {

enum class PixelFormat { Rgba8, Rgba32f };
enum class SceneColorSpace { Linear, Srgb };

struct SceneBounds {
    math::Vec3 min;
    math::Vec3 max;
};

struct FrameMetadata {
    std::size_t frame_index{};
    std::uint32_t camera_index{};
    std::uint32_t width{};
    std::uint32_t height{};
    float fx{};
    float fy{};
    float cx{};
    float cy{};
    float time{};
    math::Mat3x4 world_from_camera{};
    math::Vec3 camera_origin{};

    [[nodiscard]] std::size_t pixel_count() const noexcept {
        return static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    }
};

struct ImageView {
    const std::byte* data{nullptr};
    std::uint32_t width{};
    std::uint32_t height{};
    std::uint32_t row_stride{};
    std::uint32_t pixel_stride{};
    std::uint32_t roi_x{};
    std::uint32_t roi_y{};
    std::uint32_t roi_w{};
    std::uint32_t roi_h{};
};

struct RayBatch {
    std::vector<math::Vec3> origins;
    std::vector<math::Vec3> directions;
    std::uint32_t width{};
    std::uint32_t height{};
};

enum class LoaderErrorCode {
    None = 0,
    OpenFailed,
    InvalidCameraIndex,
    UnsupportedPixelFormat,
    UnsupportedColorSpace,
    MissingCameraData
};

struct LoaderError {
    LoaderErrorCode code{LoaderErrorCode::None};
    std::string message;
};

class HostpackLoader {
public:
    HostpackLoader() = default;
    static std::expected<HostpackLoader, LoaderError> create(std::string_view pack_path);

    HostpackLoader(const HostpackLoader&) = delete;
    HostpackLoader& operator=(const HostpackLoader&) = delete;

    HostpackLoader(HostpackLoader&& other) noexcept;
    HostpackLoader& operator=(HostpackLoader&& other) noexcept;

    ~HostpackLoader();

    [[nodiscard]] std::span<const FrameMetadata> frames() const noexcept {
        return {frame_metadata_.data(), frame_metadata_.size()};
    }

    [[nodiscard]] SceneBounds bounds() const noexcept { return bounds_; }
    [[nodiscard]] SceneColorSpace color_space() const noexcept { return color_space_; }
    [[nodiscard]] PixelFormat pixel_format() const noexcept { return pixel_format_; }
    [[nodiscard]] const std::string& pack_path() const noexcept { return pack_path_; }

    [[nodiscard]] ImageView image_view(std::size_t frame_index) const;
    [[nodiscard]] RayBatch build_ray_batch(std::size_t frame_index) const;

    core::DeviceBuffer<std::byte>::Result upload_image(std::size_t frame_index, core::DeviceBuffer<std::byte>& buffer) const;

private:
    HostpackLoader(void* handle, std::string path, PixelFormat pixel_format, SceneColorSpace color_space,
                   SceneBounds bounds, std::vector<FrameMetadata> metadata, std::vector<ImageView> images);

    void reset() noexcept;
    void swap(HostpackLoader& other) noexcept;

    void* handle_{nullptr};
    std::string pack_path_;
    PixelFormat pixel_format_{PixelFormat::Rgba8};
    SceneColorSpace color_space_{SceneColorSpace::Linear};
    SceneBounds bounds_{};
    std::vector<FrameMetadata> frame_metadata_;
    std::vector<ImageView> image_views_;
};

} // namespace instantngp::io
