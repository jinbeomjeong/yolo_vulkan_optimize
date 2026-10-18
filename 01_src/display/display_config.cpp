#include "display/display_config.hpp"

namespace yvo::display {

const char* sync_name(DisplaySync s) { return s == DisplaySync::Latest ? "latest" : "matched"; }
const char* pace_name(DisplayPace p) {
    switch (p) {
    case DisplayPace::Source: return "source";
    case DisplayPace::Asap: return "asap";
    default: return "none";
    }
}
bool parse_color_range(const std::string& s, ColorRangeRequest* out) {
    if (s == "auto") *out = ColorRangeRequest::Auto;
    else if (s == "full") *out = ColorRangeRequest::Full;
    else if (s == "limited") *out = ColorRangeRequest::Limited;
    else return false;
    return true;
}
const char* color_range_name(ColorRangeRequest r) {
    switch (r) {
    case ColorRangeRequest::Full: return "full";
    case ColorRangeRequest::Limited: return "limited";
    default: return "auto";
    }
}
bool parse_display_sync(const std::string& s, DisplaySync* out) {
    if (s == "matched") *out = DisplaySync::Matched;
    else if (s == "latest") *out = DisplaySync::Latest;
    else return false;
    return true;
}
bool parse_display_pace(const std::string& s, DisplayPace* out) {
    if (s == "source") *out = DisplayPace::Source;
    else if (s == "asap") *out = DisplayPace::Asap;
    else if (s == "none" || s == "off") *out = DisplayPace::None;
    else return false;
    return true;
}

} // namespace yvo::display
