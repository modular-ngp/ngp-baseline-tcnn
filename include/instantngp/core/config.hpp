#pragma once

#include <expected>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace instantngp::config {

struct ParseError {
    std::string message;
    std::size_t offset{};
};

class Value;

class Value {
public:
    enum class Type { Null, Boolean, Number, String, Array, Object };

    struct ObjectMember;

    using Array = std::vector<Value>;

    using Object = std::vector<ObjectMember>;

    Value();
    explicit Value(std::nullptr_t);
    explicit Value(bool boolean);
    explicit Value(double number);
    explicit Value(std::string string);
    explicit Value(Array array);
    explicit Value(Object object);

    [[nodiscard]] Type type() const noexcept;
    [[nodiscard]] bool is_null() const noexcept;
    [[nodiscard]] bool is_boolean() const noexcept;
    [[nodiscard]] bool is_number() const noexcept;
    [[nodiscard]] bool is_string() const noexcept;
    [[nodiscard]] bool is_array() const noexcept;
    [[nodiscard]] bool is_object() const noexcept;

    [[nodiscard]] bool as_boolean() const;
    [[nodiscard]] double as_number() const;
    [[nodiscard]] const std::string& as_string() const;
    [[nodiscard]] const Array& as_array() const;
    [[nodiscard]] const Object& as_object() const;

    [[nodiscard]] const Value* find(std::string_view key) const;

private:
    Type type_{Type::Null};
    bool boolean_{false};
    double number_{0.0};
    std::string string_;
    Array array_;
    Object object_;
};

struct Value::ObjectMember {
    std::string key;
    Value value;
};

class Document {
public:
    Document();
    explicit Document(Value root);

    [[nodiscard]] const Value& root() const noexcept;

    static std::expected<Document, ParseError> from_string(std::string_view source);
    static std::expected<Document, ParseError> from_file(std::string_view path);

private:
    Value root_;
};

} // namespace instantngp::config
