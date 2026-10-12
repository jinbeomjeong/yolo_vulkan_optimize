#pragma once
// Strict parsing of numeric command-line option values (tools and pipeline/main.cpp only).
//
// Until 2026-10-11 the tools read numbers with std::atoi/atol/atof, which turn "abc" into 0 and "10x" into 10
// without a word — `yvo_pipeline --frames abc` silently meant "run the whole video".
//
// parse_number() accepts exactly the forms those calls accepted for well-formed input, so valid command lines keep
// their meaning bit for bit:
//   integers       optional leading whitespace, optional sign, decimal digits (base 10 — "0x10" is not hex, as before)
//   floating point everything strtod accepts (exponent, hex float), parsed as double and then converted, as atof did
// and rejects what atoi/atof silently mangled: an empty string, any trailing character (including whitespace),
// out-of-range values (ERANGE, or not representable in the destination type), inf/nan, and a '-' for unsigned types.
//
// parse_stof() is the float variant for call sites that used std::stof: it calls strtof directly (no double rounding
// through strtod) and, like std::stof, rejects ERANGE (overflow and underflow), so every value std::stof accepted
// without throwing parses to the same bits. It additionally rejects trailing characters and inf/nan.
//
// parse_size() parses a "WxH" size (e.g. --display-size 1920x1080) as two parse_number<int> fields split at the first
// 'x'. It accepts what sscanf("%dx%d") accepted for well-formed input (each field may carry leading whitespace and a
// sign) and rejects trailing characters ("1920x1080x5", "1920x1080 "), overflow, and a width or height <= 0.
//
// arg_number() is the option-value wrapper: on failure it prints
//   <prog>: invalid value for <opt>: '<text>'
// to stderr and exits with code 2, the usage-error code of yvo_pipeline and most tools. It is meant to be called
// while parsing arguments, before any device is opened (std::exit does not unwind the stack).
//
// Library code does not include this header — it is for tools only.
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <string_view>
#include <type_traits>

namespace yvo::tool {

// Strictly parse the whole of text into out. Returns false (out unchanged) if text is not a valid T.
template <class T>
bool parse_number(const char* text, T& out) {
    static_assert(std::is_arithmetic_v<T> && !std::is_same_v<T, bool>, "parse_number: numeric types only");
    if (text == nullptr || *text == '\0') return false;
    char* end = nullptr;
    errno = 0;
    if constexpr (std::is_floating_point_v<T>) {
        const double v = std::strtod(text, &end);
        if (end == text || *end != '\0' || errno == ERANGE || !std::isfinite(v)) return false;
        if (std::fabs(v) > static_cast<double>(std::numeric_limits<T>::max())) return false;
        out = static_cast<T>(v);
    } else if constexpr (std::is_signed_v<T>) {
        const long long v = std::strtoll(text, &end, 10);
        if (end == text || *end != '\0' || errno == ERANGE) return false;
        if (v < static_cast<long long>(std::numeric_limits<T>::min()) ||
            v > static_cast<long long>(std::numeric_limits<T>::max()))
            return false;
        out = static_cast<T>(v);
    } else {
        // strtoull accepts "-1" and wraps it to ULLONG_MAX: reject any minus sign after the leading whitespace.
        const char* p = text;
        while (*p == ' ' || (*p >= '\t' && *p <= '\r')) ++p;
        if (*p == '-') return false;
        const unsigned long long v = std::strtoull(text, &end, 10);
        if (end == text || *end != '\0' || errno == ERANGE) return false;
        if (v > static_cast<unsigned long long>(std::numeric_limits<T>::max())) return false;
        out = static_cast<T>(v);
    }
    return true;
}

// Strictly parse the whole of text into out with std::stof semantics (see above). Returns false (out unchanged) on failure.
inline bool parse_stof(const char* text, float& out) {
    if (text == nullptr || *text == '\0') return false;
    char* end = nullptr;
    errno = 0;
    const float v = std::strtof(text, &end);
    if (end == text || *end != '\0' || errno == ERANGE || !std::isfinite(v)) return false;
    out = v;
    return true;
}

// Strictly parse a "WxH" size with W > 0 and H > 0 (see above). Returns false (w, h unchanged) on failure.
inline bool parse_size(const char* text, int& w, int& h) {
    if (text == nullptr) return false;
    const char* x = std::strchr(text, 'x');
    if (x == nullptr) return false;
    const std::string first(text, x);
    int pw = 0, ph = 0;
    if (!parse_number(first.c_str(), pw) || !parse_number(x + 1, ph) || pw <= 0 || ph <= 0) return false;
    w = pw;
    h = ph;
    return true;
}

// Program name for messages: argv[0] without its directory.
inline std::string_view prog_name(const char* argv0) {
    const std::string_view p = argv0 ? argv0 : "";
    const std::size_t slash = p.rfind('/');
    return slash == std::string_view::npos ? p : p.substr(slash + 1);
}

// Prints "<prog>: invalid value for <opt>: '<text>'" to stderr and exits with code 2.
[[noreturn]] inline void exit_invalid_value(const char* argv0, std::string_view opt, const char* text) {
    const std::string_view prog = prog_name(argv0);
    std::fprintf(stderr, "%.*s: invalid value for %.*s: '%s'\n", static_cast<int>(prog.size()), prog.data(),
                 static_cast<int>(opt.size()), opt.data(), text ? text : "");
    std::exit(2);
}

// Value of option opt (e.g. "--frames", or an environment variable name). Exits with code 2 if text is not a valid T.
template <class T>
T arg_number(const char* argv0, std::string_view opt, const char* text) {
    T v{};
    if (!parse_number(text, v)) exit_invalid_value(argv0, opt, text);
    return v;
}

// arg_number<float>() with std::stof semantics (parse_stof). Exits with code 2 on failure.
inline float arg_stof(const char* argv0, std::string_view opt, const char* text) {
    float v = 0.0f;
    if (!parse_stof(text, v)) exit_invalid_value(argv0, opt, text);
    return v;
}

}  // namespace yvo::tool
