#pragma once

// Compatibility header for std::expected
// CUDA 13.0 only supports C++20, which doesn't have std::expected

#include <utility>
#include <type_traits>

#if __cplusplus >= 202302L && !defined(__CUDACC__)
// C++23 with native std::expected support (non-CUDA code)
#include <expected>
namespace instantngp {
    template<typename T, typename E>
    using expected = std::expected<T, E>;

    template<typename E>
    using unexpected = std::unexpected<E>;
}
#else
// C++20 or CUDA code - use simplified implementation to avoid NVCC compiler bugs
namespace instantngp {

    template<typename E>
    class unexpected {
    public:
        constexpr unexpected(const E& e) : error_(e) {}
        constexpr unexpected(E&& e) : error_(std::move(e)) {}

        constexpr const E& error() const & noexcept { return error_; }
        constexpr E& error() & noexcept { return error_; }
        constexpr const E&& error() const && noexcept { return std::move(error_); }
        constexpr E&& error() && noexcept { return std::move(error_); }

    private:
        E error_;
    };

    template<typename T, typename E>
    class expected {
    public:
        using value_type = T;
        using error_type = E;

        // Construct with value (for void specialization, this is default constructor)
        constexpr expected() requires std::is_void_v<T> : has_value_(true) {}

        constexpr expected(const T& val) requires (!std::is_void_v<T>)
            : has_value_(true), value_(val) {}

        constexpr expected(T&& val) requires (!std::is_void_v<T>)
            : has_value_(true), value_(std::move(val)) {}

        // Construct with error
        constexpr expected(const unexpected<E>& e)
            : has_value_(false), error_(e.error()) {}

        constexpr expected(unexpected<E>&& e)
            : has_value_(false), error_(std::move(e.error())) {}

        // Destructor
        ~expected() {
            if (has_value_) {
                if constexpr (!std::is_void_v<T>) {
                    value_.~T();
                }
            } else {
                error_.~E();
            }
        }

        // Copy/move constructors
        expected(const expected& other) : has_value_(other.has_value_) {
            if (has_value_) {
                if constexpr (!std::is_void_v<T>) {
                    new (&value_) T(other.value_);
                }
            } else {
                new (&error_) E(other.error_);
            }
        }

        expected(expected&& other) noexcept : has_value_(other.has_value_) {
            if (has_value_) {
                if constexpr (!std::is_void_v<T>) {
                    new (&value_) T(std::move(other.value_));
                }
            } else {
                new (&error_) E(std::move(other.error_));
            }
        }

        // Assignment operators
        expected& operator=(const expected& other) {
            if (this != &other) {
                this->~expected();
                new (this) expected(other);
            }
            return *this;
        }

        expected& operator=(expected&& other) noexcept {
            if (this != &other) {
                this->~expected();
                new (this) expected(std::move(other));
            }
            return *this;
        }

        // Observers
        constexpr bool has_value() const noexcept { return has_value_; }
        constexpr explicit operator bool() const noexcept { return has_value_; }

        constexpr const T& value() const & requires (!std::is_void_v<T>) { return value_; }
        constexpr T& value() & requires (!std::is_void_v<T>) { return value_; }
        constexpr const T&& value() const && requires (!std::is_void_v<T>) { return std::move(value_); }
        constexpr T&& value() && requires (!std::is_void_v<T>) { return std::move(value_); }

        constexpr const E& error() const & { return error_; }
        constexpr E& error() & { return error_; }
        constexpr const E&& error() const && { return std::move(error_); }
        constexpr E&& error() && { return std::move(error_); }

        constexpr const T& operator*() const & requires (!std::is_void_v<T>) { return value_; }
        constexpr T& operator*() & requires (!std::is_void_v<T>) { return value_; }
        constexpr const T&& operator*() const && requires (!std::is_void_v<T>) { return std::move(value_); }
        constexpr T&& operator*() && requires (!std::is_void_v<T>) { return std::move(value_); }

        constexpr const T* operator->() const requires (!std::is_void_v<T>) { return &value_; }
        constexpr T* operator->() requires (!std::is_void_v<T>) { return &value_; }

    private:
        bool has_value_;
        union {
            T value_;
            E error_;
        };
    };

    // Specialization for void
    template<typename E>
    class expected<void, E> {
    public:
        using value_type = void;
        using error_type = E;

        constexpr expected() : has_value_(true) {}

        constexpr expected(const unexpected<E>& e)
            : has_value_(false), error_(e.error()) {}

        constexpr expected(unexpected<E>&& e)
            : has_value_(false), error_(std::move(e.error())) {}

        ~expected() {
            if (!has_value_) {
                error_.~E();
            }
        }

        expected(const expected& other) : has_value_(other.has_value_) {
            if (!has_value_) {
                new (&error_) E(other.error_);
            }
        }

        expected(expected&& other) noexcept : has_value_(other.has_value_) {
            if (!has_value_) {
                new (&error_) E(std::move(other.error_));
            }
        }

        expected& operator=(const expected& other) {
            if (this != &other) {
                this->~expected();
                new (this) expected(other);
            }
            return *this;
        }

        expected& operator=(expected&& other) noexcept {
            if (this != &other) {
                this->~expected();
                new (this) expected(std::move(other));
            }
            return *this;
        }

        constexpr bool has_value() const noexcept { return has_value_; }
        constexpr explicit operator bool() const noexcept { return has_value_; }

        constexpr const E& error() const & { return error_; }
        constexpr E& error() & { return error_; }
        constexpr const E&& error() const && { return std::move(error_); }
        constexpr E&& error() && { return std::move(error_); }

    private:
        bool has_value_;
        union {
            char dummy_;
            E error_;
        };
    };
}
#endif
