#pragma once

#include <concepts>
#include <cstdint>
#include <ostream>
#include <string>
#include <string_view>
#include <vector>

namespace instantngp::core {

// Extremely small JSON writer for compact metadata.
// Supports writing objects with string/number/bool values and nested objects one level deep.

class JsonWriter {
public:
    explicit JsonWriter(std::ostream& os, bool pretty = true) : os_{os}, pretty_{pretty} {}

    void begin_object() { write_indent(); os_ << '{'; push(); newline(); }
    void end_object() { pop(); newline(); write_indent(); os_ << '}'; }

    void key(std::string_view k) { comma(); write_indent(); string(k); os_ << (pretty_ ? ": " : ":"); first_ = true; }

    void value(std::string_view v) { string(v); first_ = false; }
    void value(const char* v) { string(v); first_ = false; }
    void value(bool v) { os_ << (v ? "true" : "false"); first_ = false; }
    template <std::integral T>
    void value(T v) { os_ << v; first_ = false; }
    template <std::floating_point T>
    void value(T v) { os_ << v; first_ = false; }

    // Convenience for nested one-level object
    class Scope {
    public:
        explicit Scope(JsonWriter& w) : w_{w} { w_.begin_object(); }
        ~Scope() { w_.end_object(); }
    private:
        JsonWriter& w_;
    };

private:
    void string(std::string_view s) {
        os_ << '"';
        for (char c : s) {
            switch (c) {
                case '"': os_ << "\\\""; break;
                case '\\': os_ << "\\\\"; break;
                case '\n': os_ << "\\n"; break;
                case '\r': os_ << "\\r"; break;
                case '\t': os_ << "\\t"; break;
                default: os_ << c; break;
            }
        }
        os_ << '"';
    }

    void newline() { if (pretty_) os_ << '\n'; }
    void push() { ++indent_; first_ = true; }
    void pop() { if (indent_ > 0) --indent_; }
    void write_indent() { if (pretty_) for (std::uint32_t i = 0; i < indent_; ++i) os_ << "  "; }
    void comma() { if (!first_) { os_ << ','; newline(); } }

    std::ostream& os_;
    bool pretty_{true};
    std::uint32_t indent_{0};
    bool first_{true};
};

} // namespace instantngp::core

