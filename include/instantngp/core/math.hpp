#pragma once

#include <array>
#include <cmath>
#include <concepts>
#include <limits>

namespace instantngp::math {

struct Vec3 {
    float x{0.0F};
    float y{0.0F};
    float z{0.0F};

    constexpr Vec3() = default;
    constexpr Vec3(float x_, float y_, float z_) : x{x_}, y{y_}, z{z_} {}

    [[nodiscard]] constexpr Vec3 operator+() const noexcept { return *this; }
    [[nodiscard]] constexpr Vec3 operator-() const noexcept { return Vec3{-x, -y, -z}; }

    constexpr Vec3& operator+=(const Vec3& rhs) noexcept {
        x += rhs.x;
        y += rhs.y;
        z += rhs.z;
        return *this;
    }

    constexpr Vec3& operator-=(const Vec3& rhs) noexcept {
        x -= rhs.x;
        y -= rhs.y;
        z -= rhs.z;
        return *this;
    }

    constexpr Vec3& operator*=(float scalar) noexcept {
        x *= scalar;
        y *= scalar;
        z *= scalar;
        return *this;
    }

    constexpr Vec3& operator/=(float scalar) noexcept {
        x /= scalar;
        y /= scalar;
        z /= scalar;
        return *this;
    }
};

[[nodiscard]] constexpr Vec3 operator+(Vec3 lhs, const Vec3& rhs) noexcept {
    lhs += rhs;
    return lhs;
}

[[nodiscard]] constexpr Vec3 operator-(Vec3 lhs, const Vec3& rhs) noexcept {
    lhs -= rhs;
    return lhs;
}

[[nodiscard]] constexpr Vec3 operator*(Vec3 lhs, float scalar) noexcept {
    lhs *= scalar;
    return lhs;
}

[[nodiscard]] constexpr Vec3 operator*(float scalar, Vec3 rhs) noexcept {
    rhs *= scalar;
    return rhs;
}

[[nodiscard]] constexpr Vec3 operator/(Vec3 lhs, float scalar) noexcept {
    lhs /= scalar;
    return lhs;
}

[[nodiscard]] constexpr float dot(const Vec3& a, const Vec3& b) noexcept {
    return (a.x * b.x) + (a.y * b.y) + (a.z * b.z);
}

[[nodiscard]] constexpr Vec3 cross(const Vec3& a, const Vec3& b) noexcept {
    return Vec3{
            (a.y * b.z) - (a.z * b.y),
            (a.z * b.x) - (a.x * b.z),
            (a.x * b.y) - (a.y * b.x)};
}

[[nodiscard]] inline float length(const Vec3& v) noexcept {
    return std::sqrt(dot(v, v));
}

[[nodiscard]] inline Vec3 normalize(const Vec3& v) noexcept {
    const float len = length(v);
    if (len <= std::numeric_limits<float>::epsilon()) {
        return Vec3{};
    }
    return v / len;
}

struct Mat3x4 {
    std::array<float, 12> elements{}; // row-major: r0[0..3], r1[4..7], r2[8..11]

    [[nodiscard]] constexpr float operator()(std::size_t row, std::size_t col) const noexcept {
        return elements[row * 4 + col];
    }

    constexpr float& operator()(std::size_t row, std::size_t col) noexcept {
        return elements[row * 4 + col];
    }

    [[nodiscard]] static constexpr Mat3x4 identity() noexcept {
        Mat3x4 m{};
        m(0, 0) = 1.0F; m(0, 1) = 0.0F; m(0, 2) = 0.0F; m(0, 3) = 0.0F;
        m(1, 0) = 0.0F; m(1, 1) = 1.0F; m(1, 2) = 0.0F; m(1, 3) = 0.0F;
        m(2, 0) = 0.0F; m(2, 1) = 0.0F; m(2, 2) = 1.0F; m(2, 3) = 0.0F;
        return m;
    }
};

[[nodiscard]] constexpr Mat3x4 make_translation(const Vec3& t) noexcept {
    Mat3x4 m = Mat3x4::identity();
    m(0, 3) = t.x;
    m(1, 3) = t.y;
    m(2, 3) = t.z;
    return m;
}

[[nodiscard]] constexpr Vec3 transform_point(const Mat3x4& m, const Vec3& v) noexcept {
    return Vec3{
            (m(0, 0) * v.x) + (m(0, 1) * v.y) + (m(0, 2) * v.z) + m(0, 3),
            (m(1, 0) * v.x) + (m(1, 1) * v.y) + (m(1, 2) * v.z) + m(1, 3),
            (m(2, 0) * v.x) + (m(2, 1) * v.y) + (m(2, 2) * v.z) + m(2, 3)};
}

[[nodiscard]] constexpr Vec3 transform_direction(const Mat3x4& m, const Vec3& v) noexcept {
    return Vec3{
            (m(0, 0) * v.x) + (m(0, 1) * v.y) + (m(0, 2) * v.z),
            (m(1, 0) * v.x) + (m(1, 1) * v.y) + (m(1, 2) * v.z),
            (m(2, 0) * v.x) + (m(2, 1) * v.y) + (m(2, 2) * v.z)};
}

[[nodiscard]] constexpr Mat3x4 compose(const Mat3x4& lhs, const Mat3x4& rhs) noexcept {
    Mat3x4 result{};
    for (std::size_t row = 0; row < 3; ++row) {
        for (std::size_t col = 0; col < 4; ++col) {
            float value = 0.0F;
            for (std::size_t k = 0; k < 3; ++k) {
                value += lhs(row, k) * rhs(k, col);
            }
            if (col == 3) {
                value += lhs(row, 3);
            }
            result(row, col) = value;
        }
    }
    return result;
}

} // namespace instantngp::math
