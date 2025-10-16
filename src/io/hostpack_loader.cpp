#include "instantngp/io/hostpack_loader.hpp"

#include "dataset.h"

#include <algorithm>
#include <span>
#include <string>
#include <optional>
#include <stdexcept>

namespace instantngp::io {

namespace {

std::optional<PixelFormat> to_pixel_format(dataset::PixelFormat format) noexcept {
    switch (format) {
        case dataset::PixelFormat::RGBA8:
            return PixelFormat::Rgba8;
        case dataset::PixelFormat::RGBA32F:
            return PixelFormat::Rgba32f;
        default:
            return std::nullopt;
    }
}

std::optional<SceneColorSpace> to_color_space(dataset::ColorSpace color_space) noexcept {
    switch (color_space) {
        case dataset::ColorSpace::Linear:
            return SceneColorSpace::Linear;
        case dataset::ColorSpace::SRGB:
            return SceneColorSpace::Srgb;
        default:
            return std::nullopt;
    }
}

math::Mat3x4 load_transform(const float* data) {
    math::Mat3x4 transform{};
    for (std::size_t row = 0; row < 3; ++row) {
        for (std::size_t col = 0; col < 4; ++col) {
            transform(row, col) = data[row * 4 + col];
        }
    }
    return transform;
}

} // namespace

HostpackLoader::HostpackLoader(void* handle, std::string path, PixelFormat pixel_format, SceneColorSpace color_space,
                               SceneBounds bounds, std::vector<FrameMetadata> metadata, std::vector<ImageView> images)
    : handle_{handle},
      pack_path_{std::move(path)},
      pixel_format_{pixel_format},
      color_space_{color_space},
      bounds_{bounds},
      frame_metadata_{std::move(metadata)},
      image_views_{std::move(images)} {}

std::expected<HostpackLoader, LoaderError> HostpackLoader::create(std::string_view pack_path) {
    const std::string path{pack_path};
    dataset::PackHandle handle = dataset::open_hostpack(path);
    if (handle == nullptr) {
        const dataset::Error error = dataset::last_error();
        return std::unexpected(LoaderError{
                .code = LoaderErrorCode::OpenFailed,
                .message = "Failed to open hostpack '" + path + "' (dataset error code " + std::to_string(static_cast<int>(error)) + ")"});
    }

    const auto close_on_failure = [&]() {
        dataset::close_hostpack(handle);
    };

    SceneBounds bounds{};
    float raw_min[3]{};
    float raw_max[3]{};
    dataset::scene_aabb(handle, raw_min, raw_max);
    bounds.min = math::Vec3{raw_min[0], raw_min[1], raw_min[2]};
    bounds.max = math::Vec3{raw_max[0], raw_max[1], raw_max[2]};

    const auto pixel_format_opt = to_pixel_format(dataset::pack_pixel_format(handle));
    if (!pixel_format_opt.has_value()) {
        close_on_failure();
        return std::unexpected(LoaderError{
                .code = LoaderErrorCode::UnsupportedPixelFormat,
                .message = "Unsupported pixel format in hostpack"});
    }

    const auto color_space_opt = to_color_space(dataset::scene_color_space(handle));
    if (!color_space_opt.has_value()) {
        close_on_failure();
        return std::unexpected(LoaderError{
                .code = LoaderErrorCode::UnsupportedColorSpace,
                .message = "Unsupported color space in hostpack"});
    }

    const std::size_t frame_count = dataset::frame_count(handle);
    const dataset::CameraSOAView camera_view = dataset::camera_soa(handle);
    if (camera_view.count == 0) {
        close_on_failure();
        return std::unexpected(LoaderError{
                .code = LoaderErrorCode::MissingCameraData,
                .message = "Hostpack contains no camera entries"});
    }

    auto ensure_camera_array = [&](const float* ptr, std::string_view name) -> std::expected<const float*, LoaderError> {
        if (ptr == nullptr) {
            return std::unexpected(LoaderError{
                    .code = LoaderErrorCode::MissingCameraData,
                    .message = std::string{name} + " array missing from hostpack"});
        }
        return ptr;
    };

    auto fx_ptr_result = ensure_camera_array(camera_view.fx, "fx");
    if (!fx_ptr_result.has_value()) {
        close_on_failure();
        return std::unexpected(fx_ptr_result.error());
    }
    auto fy_ptr_result = ensure_camera_array(camera_view.fy, "fy");
    if (!fy_ptr_result.has_value()) {
        close_on_failure();
        return std::unexpected(fy_ptr_result.error());
    }
    auto cx_ptr_result = ensure_camera_array(camera_view.cx, "cx");
    if (!cx_ptr_result.has_value()) {
        close_on_failure();
        return std::unexpected(cx_ptr_result.error());
    }
    auto cy_ptr_result = ensure_camera_array(camera_view.cy, "cy");
    if (!cy_ptr_result.has_value()) {
        close_on_failure();
        return std::unexpected(cy_ptr_result.error());
    }
    if (camera_view.T3x4 == nullptr) {
        close_on_failure();
        return std::unexpected(LoaderError{
                .code = LoaderErrorCode::MissingCameraData,
                .message = "Camera transforms (T3x4) missing from hostpack"});
    }

    std::vector<FrameMetadata> metadata;
    std::vector<ImageView> images;
    metadata.reserve(frame_count);
    images.reserve(frame_count);

    for (std::size_t frame_index = 0; frame_index < frame_count; ++frame_index) {
        const std::size_t camera_index = dataset::frame_camera_index(handle, frame_index);
        if (camera_index >= camera_view.count) {
            close_on_failure();
            return std::unexpected(LoaderError{
                    .code = LoaderErrorCode::InvalidCameraIndex,
                    .message = "Camera index out of range for frame " + std::to_string(frame_index)});
        }

        auto image = dataset::image_view(handle, frame_index);

        FrameMetadata frame{};
        frame.frame_index = frame_index;
        frame.camera_index = static_cast<std::uint32_t>(camera_index);
        frame.width = image.width;
        frame.height = image.height;
        frame.fx = fx_ptr_result.value()[camera_index];
        frame.fy = fy_ptr_result.value()[camera_index];
        frame.cx = cx_ptr_result.value()[camera_index];
        frame.cy = cy_ptr_result.value()[camera_index];
        frame.time = camera_view.time != nullptr ? static_cast<float>(camera_view.time[camera_index]) : 0.0F;
        frame.world_from_camera = load_transform(camera_view.T3x4 + camera_index * 12U);
        frame.camera_origin = math::Vec3{
                frame.world_from_camera(0, 3),
                frame.world_from_camera(1, 3),
                frame.world_from_camera(2, 3)};

        ImageView view{};
        view.data = reinterpret_cast<const std::byte*>(image.data);
        view.width = image.width;
        view.height = image.height;
        view.row_stride = image.row_stride;
        view.pixel_stride = image.pixel_stride;
        view.roi_x = image.roi_x;
        view.roi_y = image.roi_y;
        view.roi_w = image.roi_w;
        view.roi_h = image.roi_h;

        metadata.push_back(frame);
        images.push_back(view);
    }

    return HostpackLoader{
            handle,
            path,
            *pixel_format_opt,
            *color_space_opt,
            bounds,
            std::move(metadata),
            std::move(images)};
}

HostpackLoader::HostpackLoader(HostpackLoader&& other) noexcept {
    swap(other);
}

HostpackLoader& HostpackLoader::operator=(HostpackLoader&& other) noexcept {
    if (this != &other) {
        reset();
        swap(other);
    }
    return *this;
}

HostpackLoader::~HostpackLoader() {
    reset();
}

ImageView HostpackLoader::image_view(std::size_t frame_index) const {
    if (frame_index >= image_views_.size()) {
        throw std::out_of_range("Frame index out of range in image_view");
    }
    return image_views_[frame_index];
}

RayBatch HostpackLoader::build_ray_batch(std::size_t frame_index) const {
    if (frame_index >= frame_metadata_.size()) {
        throw std::out_of_range("Frame index out of range in build_ray_batch");
    }
    const FrameMetadata& frame = frame_metadata_[frame_index];
    RayBatch rays{};
    rays.width = frame.width;
    rays.height = frame.height;
    rays.origins.resize(frame.pixel_count(), frame.camera_origin);
    rays.directions.resize(frame.pixel_count());

    const float fx = frame.fx;
    const float fy = frame.fy;
    const float cx = frame.cx;
    const float cy = frame.cy;
    const math::Mat3x4& world_from_camera = frame.world_from_camera;

    for (std::uint32_t y = 0; y < frame.height; ++y) {
        for (std::uint32_t x = 0; x < frame.width; ++x) {
            const float pixel_x = (static_cast<float>(x) + 0.5F - cx) / fx;
            const float pixel_y = (static_cast<float>(y) + 0.5F - cy) / fy;
            math::Vec3 direction_camera{pixel_x, pixel_y, 1.0F};
            direction_camera = math::normalize(direction_camera);
            const math::Vec3 direction_world = math::normalize(math::transform_direction(world_from_camera, direction_camera));

            const std::size_t index = static_cast<std::size_t>(y) * frame.width + x;
            rays.directions[index] = direction_world;
        }
    }

    return rays;
}

core::DeviceBuffer<std::byte>::Result HostpackLoader::upload_image(
        std::size_t frame_index, core::DeviceBuffer<std::byte>& buffer) const {
    if (frame_index >= image_views_.size()) {
        return std::unexpected(core::DeviceBufferError{
                .code = core::DeviceBufferErrorCode::InvalidArgument,
                .message = "Frame index out of range in upload_image"});
    }
    const ImageView view = image_views_[frame_index];
    const std::size_t row_bytes = view.row_stride;
    const std::size_t total_bytes = row_bytes * view.height;
    const auto* data_ptr = view.data;
    if (data_ptr == nullptr || total_bytes == 0) {
        buffer.reset();
        return {};
    }
    const std::span<const std::byte> span{data_ptr, total_bytes};
    return buffer.copy_from(span);
}

void HostpackLoader::reset() noexcept {
    if (handle_ != nullptr) {
        dataset::close_hostpack(static_cast<dataset::PackHandle>(handle_));
        handle_ = nullptr;
    }
}

void HostpackLoader::swap(HostpackLoader& other) noexcept {
    std::swap(handle_, other.handle_);
    std::swap(pack_path_, other.pack_path_);
    std::swap(pixel_format_, other.pixel_format_);
    std::swap(color_space_, other.color_space_);
    std::swap(bounds_, other.bounds_);
    std::swap(frame_metadata_, other.frame_metadata_);
    std::swap(image_views_, other.image_views_);
}

} // namespace instantngp::io
