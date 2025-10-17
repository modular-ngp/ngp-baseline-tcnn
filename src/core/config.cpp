#include "instantngp/core/config.hpp"

#include <charconv>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace instantngp::config {

namespace {

class Parser {
public:
    explicit Parser(std::string_view source) : source_{source} {}

    instantngp::expected<Value, ParseError> parse() {
        skip_whitespace();
        auto value = parse_value();
        if (!value.has_value()) {
            return value;
        }
        skip_whitespace();
        if (!is_end()) {
            return instantngp::unexpected(ParseError{.message = "Trailing characters after JSON document", .offset = cursor_});
        }
        return value;
    }

private:
    [[nodiscard]] bool is_end() const noexcept { return cursor_ >= source_.size(); }

    [[nodiscard]] char peek() const noexcept {
        return is_end() ? '\0' : source_[cursor_];
    }

    [[nodiscard]] char advance() noexcept {
        return is_end() ? '\0' : source_[cursor_++];
    }

    void skip_whitespace() noexcept {
        while (!is_end()) {
            const char c = peek();
            if (c == ' ' || c == '\n' || c == '\r' || c == '\t') {
                advance();
            } else {
                break;
            }
        }
    }

    instantngp::expected<Value, ParseError> parse_value() {
        if (is_end()) {
            return instantngp::unexpected(ParseError{.message = "Unexpected end of input", .offset = cursor_});
        }

        const char c = peek();
        switch (c) {
            case '{':
                return parse_object();
            case '[':
                return parse_array();
            case '"': {
                auto string_value = parse_string();
                if (!string_value.has_value()) {
                    return std::unexpected(string_value.error());
                }
                return Value{std::move(string_value.value())};
            }
            case 't':
                return parse_literal("true", Value{true});
            case 'f':
                return parse_literal("false", Value{false});
            case 'n':
                return parse_literal("null", Value{nullptr});
            default:
                if (c == '-' || std::isdigit(static_cast<unsigned char>(c)) != 0) {
                    return parse_number();
                }
                return std::unexpected(ParseError{.message = "Invalid JSON token", .offset = cursor_});
        }
    }

    instantngp::expected<Value, ParseError> parse_object() {
        Value::Object members{};
        advance(); // consume '{'
        skip_whitespace();
        if (peek() == '}') {
            advance();
            return Value{std::move(members)};
        }

        while (true) {
            skip_whitespace();
            if (peek() != '"') {
                return std::unexpected(ParseError{.message = "Expected string key in object", .offset = cursor_});
            }
            auto key_result = parse_string();
            if (!key_result.has_value()) {
                return std::unexpected(key_result.error());
            }
            std::string key = std::move(key_result.value());
            skip_whitespace();
            if (advance() != ':') {
                return std::unexpected(ParseError{.message = "Expected ':' after object key", .offset = cursor_});
            }
            skip_whitespace();
            auto value = parse_value();
            if (!value.has_value()) {
                return value;
            }
            members.push_back(Value::ObjectMember{std::move(key), std::move(value.value())});
            skip_whitespace();
            const char delimiter = advance();
            if (delimiter == '}') {
                break;
            }
            if (delimiter != ',') {
                return std::unexpected(ParseError{.message = "Expected ',' or '}' in object", .offset = cursor_});
            }
            skip_whitespace();
        }
        return Value{std::move(members)};
    }

    instantngp::expected<Value, ParseError> parse_array() {
        Value::Array elements{};
        advance(); // consume '['
        skip_whitespace();
        if (peek() == ']') {
            advance();
            return Value{std::move(elements)};
        }

        while (true) {
            skip_whitespace();
            auto value = parse_value();
            if (!value.has_value()) {
                return value;
            }
            elements.push_back(std::move(value.value()));
            skip_whitespace();
            const char delimiter = advance();
            if (delimiter == ']') {
                break;
            }
            if (delimiter != ',') {
                return std::unexpected(ParseError{.message = "Expected ',' or ']' in array", .offset = cursor_});
            }
            skip_whitespace();
        }
        return Value{std::move(elements)};
    }

    instantngp::expected<Value, ParseError> parse_number() {
        const std::size_t start = cursor_;
        if (peek() == '-') {
            advance();
        }
        if (peek() == '0') {
            advance();
        } else {
            if (std::isdigit(static_cast<unsigned char>(peek())) == 0) {
                return std::unexpected(ParseError{.message = "Invalid number literal", .offset = cursor_});
            }
            while (std::isdigit(static_cast<unsigned char>(peek())) != 0) {
                advance();
            }
        }
        if (peek() == '.') {
            advance();
            if (std::isdigit(static_cast<unsigned char>(peek())) == 0) {
                return std::unexpected(ParseError{.message = "Invalid fractional component", .offset = cursor_});
            }
            while (std::isdigit(static_cast<unsigned char>(peek())) != 0) {
                advance();
            }
        }
        if (peek() == 'e' || peek() == 'E') {
            advance();
            if (peek() == '+' || peek() == '-') {
                advance();
            }
            if (std::isdigit(static_cast<unsigned char>(peek())) == 0) {
                return std::unexpected(ParseError{.message = "Invalid exponent component", .offset = cursor_});
            }
            while (std::isdigit(static_cast<unsigned char>(peek())) != 0) {
                advance();
            }
        }

        const std::string_view token = source_.substr(start, cursor_ - start);
        double number{};
        const auto conversion = std::from_chars(token.data(), token.data() + token.size(), number);
        if (conversion.ec != std::errc{}) {
            return std::unexpected(ParseError{.message = "Failed to parse numeric literal", .offset = start});
        }
        return Value{number};
    }

    instantngp::expected<Value, ParseError> parse_literal(std::string_view literal, Value value) {
        for (const char expected : literal) {
            if (is_end() || advance() != expected) {
                return std::unexpected(ParseError{.message = "Invalid literal", .offset = cursor_});
            }
        }
        return value;
    }

    instantngp::expected<std::string, ParseError> parse_string() {
        std::string result;
        if (advance() != '"') {
            return std::unexpected(ParseError{.message = "Expected '\"' to start string", .offset = cursor_});
        }
        while (!is_end()) {
            const char c = advance();
            if (c == '"') {
                return result;
            }
            if (c == '\\') {
                const char escaped = advance();
                switch (escaped) {
                    case '"': result.push_back('"'); break;
                    case '\\': result.push_back('\\'); break;
                    case '/': result.push_back('/'); break;
                    case 'b': result.push_back('\b'); break;
                    case 'f': result.push_back('\f'); break;
                    case 'n': result.push_back('\n'); break;
                    case 'r': result.push_back('\r'); break;
                    case 't': result.push_back('\t'); break;
                    default:
                        return std::unexpected(ParseError{.message = "Unsupported escape sequence", .offset = cursor_});
                }
            } else {
                result.push_back(c);
            }
        }
        return std::unexpected(ParseError{.message = "Unterminated string literal", .offset = cursor_});
    }

    std::string_view source_;
    std::size_t cursor_{0};
};

} // namespace

Value::Value() = default;
Value::Value(std::nullptr_t) : type_{Type::Null} {}
Value::Value(bool boolean) : type_{Type::Boolean}, boolean_{boolean} {}
Value::Value(double number) : type_{Type::Number}, boolean_{false}, number_{number} {}
Value::Value(std::string string) : type_{Type::String}, boolean_{false}, number_{0.0}, string_{std::move(string)} {}
Value::Value(Array array) : type_{Type::Array}, boolean_{false}, number_{0.0}, array_{std::move(array)} {}
Value::Value(Object object) : type_{Type::Object}, boolean_{false}, number_{0.0}, object_{std::move(object)} {}

Value::Type Value::type() const noexcept { return type_; }
bool Value::is_null() const noexcept { return type_ == Type::Null; }
bool Value::is_boolean() const noexcept { return type_ == Type::Boolean; }
bool Value::is_number() const noexcept { return type_ == Type::Number; }
bool Value::is_string() const noexcept { return type_ == Type::String; }
bool Value::is_array() const noexcept { return type_ == Type::Array; }
bool Value::is_object() const noexcept { return type_ == Type::Object; }

bool Value::as_boolean() const {
    if (!is_boolean()) {
        throw std::logic_error("JSON value is not a boolean");
    }
    return boolean_;
}

double Value::as_number() const {
    if (!is_number()) {
        throw std::logic_error("JSON value is not a number");
    }
    return number_;
}

const std::string& Value::as_string() const {
    if (!is_string()) {
        throw std::logic_error("JSON value is not a string");
    }
    return string_;
}

const Value::Array& Value::as_array() const {
    if (!is_array()) {
        throw std::logic_error("JSON value is not an array");
    }
    return array_;
}

const Value::Object& Value::as_object() const {
    if (!is_object()) {
        throw std::logic_error("JSON value is not an object");
    }
    return object_;
}

const Value* Value::find(std::string_view key) const {
    if (!is_object()) {
        return nullptr;
    }
    for (const auto& member : object_) {
        if (member.key == key) {
            return &member.value;
        }
    }
    return nullptr;
}

Document::Document() = default;
Document::Document(Value root) : root_{std::move(root)} {}

const Value& Document::root() const noexcept {
    return root_;
}

instantngp::expected<Document, ParseError> Document::from_string(std::string_view source) {
    Parser parser{source};
    auto result = parser.parse();
    if (!result.has_value()) {
        return std::unexpected(result.error());
    }
    return Document{std::move(result.value())};
}

instantngp::expected<Document, ParseError> Document::from_file(std::string_view path) {
    namespace fs = std::filesystem;
    const fs::path file_path{path};
    std::ifstream file{file_path, std::ios::binary};
    if (!file) {
        return std::unexpected(ParseError{.message = "Unable to open configuration file", .offset = 0});
    }
    std::string content;
    file.seekg(0, std::ios::end);
    content.resize(static_cast<std::size_t>(file.tellg()));
    file.seekg(0, std::ios::beg);
    file.read(content.data(), static_cast<std::streamsize>(content.size()));
    if (!file) {
        return std::unexpected(ParseError{.message = "Failed to read configuration file", .offset = 0});
    }
    return from_string(content);
}

} // namespace instantngp::config
