#pragma once
// Reading and comparison of the GPU postprocessing fixtures (04_data/raw_tensors/) — **the single implementation for tests and probes**.
//
// Used by: 05_tests/test_postproc.cpp (PostprocFixture.* gate),
//          01_src/tools/vulkan/postproc_probe.cpp (--probe fixture/sweep/gpuab/live).
//
// Before 2026-09-21 this code (JSON parser · fixture loader · G5 comparison · G6 canonical form, about 200 lines) existed separately in two files
// and **had already diverged**: the test-side JSON parser did not handle string escapes
// and lacked the stream name field and max_score_err, while conversely the probe side lacked the frames == 0 check.
// The union of the two lives here. **Gate judgment (what counts as a pass) is left to each caller.**
//
// Header-only. Its only dependencies are core/, ref/ and vulkan/yolo_postproc.hpp.
#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "ref/cpu_yolov8_decode.hpp"
#include "core/frame_types.hpp"
#include "vulkan/yolo_postproc.hpp"

namespace yvo::fixture {

using yvo::DetectionKeyed;
using yvo::vk::kYoloNumScales;
using yvo::vk::kYoloNumTensors;
using yvo::vk::YoloTensorLayout;
using yvo::vk::yolo_box_tensor;
using yvo::vk::yolo_cls_tensor;

// G5 coordinate tolerance [px] (plan §4.5). The test gate and the probe judgment use the same value.
inline constexpr float kCoordTolPx = 0.25f;

struct Json {
    enum class Type { Null, Bool, Num, Str, Arr, Obj } type = Type::Null;
    bool b = false;
    double num = 0.0;
    std::string str;
    std::vector<Json> arr;
    std::vector<std::pair<std::string, Json>> obj;

    const Json* find(const std::string& k) const {
        for (const auto& kv : obj)
            if (kv.first == k) return &kv.second;
        return nullptr;
    }
    const Json& at(const std::string& k) const {
        const Json* p = find(k);
        if (!p) throw std::runtime_error("fixtures.json: missing key: " + k);
        return *p;
    }
    double n(const std::string& k) const { return at(k).num; }
    int i(const std::string& k) const { return static_cast<int>(at(k).num); }
    float f(const std::string& k) const { return static_cast<float>(at(k).num); }
    const std::string& s(const std::string& k) const { return at(k).str; }
};

class JsonParser {
public:
    explicit JsonParser(std::string t) : t_(std::move(t)) {}
    Json parse() {
        Json v = value();
        ws();
        return v;
    }

private:
    void ws() {
        while (p_ < t_.size() && std::isspace(static_cast<unsigned char>(t_[p_]))) ++p_;
    }
    [[noreturn]] void fail(const char* what) const {
        throw std::runtime_error(std::string("fixtures.json parse failed(") + what + ") at " + std::to_string(p_));
    }
    void expect(char c) {
        ws();
        if (p_ >= t_.size() || t_[p_] != c) fail("expect");
        ++p_;
    }
    Json value() {
        ws();
        if (p_ >= t_.size()) fail("eof");
        const char c = t_[p_];
        if (c == '{') return object();
        if (c == '[') return array();
        if (c == '"') {
            Json j;
            j.type = Json::Type::Str;
            j.str = string();
            return j;
        }
        if (t_.compare(p_, 4, "true") == 0) {
            p_ += 4;
            Json j;
            j.type = Json::Type::Bool;
            j.b = true;
            return j;
        }
        if (t_.compare(p_, 5, "false") == 0) {
            p_ += 5;
            Json j;
            j.type = Json::Type::Bool;
            return j;
        }
        if (t_.compare(p_, 4, "null") == 0) {
            p_ += 4;
            return Json{};
        }
        return number();
    }
    Json object() {
        expect('{');
        Json j;
        j.type = Json::Type::Obj;
        ws();
        if (p_ < t_.size() && t_[p_] == '}') {
            ++p_;
            return j;
        }
        for (;;) {
            ws();
            std::string k = string();
            expect(':');
            j.obj.emplace_back(std::move(k), value());
            ws();
            if (p_ < t_.size() && t_[p_] == ',') {
                ++p_;
                continue;
            }
            expect('}');
            return j;
        }
    }
    Json array() {
        expect('[');
        Json j;
        j.type = Json::Type::Arr;
        ws();
        if (p_ < t_.size() && t_[p_] == ']') {
            ++p_;
            return j;
        }
        for (;;) {
            j.arr.push_back(value());
            ws();
            if (p_ < t_.size() && t_[p_] == ',') {
                ++p_;
                continue;
            }
            expect(']');
            return j;
        }
    }
    std::string string() {
        expect('"');
        std::string out;
        while (p_ < t_.size() && t_[p_] != '"') {
            if (t_[p_] == '\\' && p_ + 1 < t_.size()) {
                ++p_;
                const char e = t_[p_++];
                switch (e) {
                    case 'n': out.push_back('\n'); break;
                    case 't': out.push_back('\t'); break;
                    case 'r': out.push_back('\r'); break;
                    default: out.push_back(e); break;
                }
                continue;
            }
            out.push_back(t_[p_++]);
        }
        expect('"');
        return out;
    }
    Json number() {
        const std::size_t start = p_;
        while (p_ < t_.size() && (std::isdigit(static_cast<unsigned char>(t_[p_])) || t_[p_] == '-' ||
                                  t_[p_] == '+' || t_[p_] == '.' || t_[p_] == 'e' || t_[p_] == 'E'))
            ++p_;
        if (start == p_) fail("number");
        Json j;
        j.type = Json::Type::Num;
        j.num = std::strtod(t_.substr(start, p_ - start).c_str(), nullptr);
        return j;
    }

    std::string t_;
    std::size_t p_ = 0;
};

// ─────────────────────────────────────────────────────────────────── fixtures

struct FixtureStream {
    std::string name;   // full HailoRT stream name
    std::string file;   // file suffix (frame%03d_<file>.bin)
    int scale_id = 0;
    bool is_cls = false;
    int grid_h = 0, grid_w = 0, w_padded = 0, features = 0, stride = 0;
    std::size_t bytes = 0;
    float qp_zp = 0.0f, qp_scale = 1.0f;
};

struct FixtureSet {
    std::string dir;
    yvo::LetterboxParams lb;
    int src_w = 0, src_h = 0;
    int frames = 0;
    int num_classes = 80;
    int reg_max = yvo::kDflRegMax;
    std::vector<FixtureStream> streams;   // 6 of them
};

inline FixtureSet load_fixtures(const std::string& dir) {
    const std::string path = dir + "/fixtures.json";
    std::ifstream in(path);
    if (!in) throw std::runtime_error("could not read fixture metadata: " + path + " (run yvo_hailo_raw_probe first)");
    std::stringstream ss;
    ss << in.rdbuf();
    const Json root = JsonParser(ss.str()).parse();

    FixtureSet fs;
    fs.dir = dir;
    fs.frames = root.i("frames");
    fs.src_w = root.i("src_width");
    fs.src_h = root.i("src_height");
    fs.num_classes = root.i("num_classes");
    fs.reg_max = root.i("reg_max");
    const Json& lb = root.at("letterbox");
    fs.lb.dst_w = lb.i("dst_w");
    fs.lb.dst_h = lb.i("dst_h");
    fs.lb.scale = lb.f("scale");
    fs.lb.pad_x = lb.i("pad_x");
    fs.lb.pad_y = lb.i("pad_y");
    fs.lb.pad_value = static_cast<std::uint8_t>(lb.i("pad_value"));

    for (const Json& s : root.at("streams").arr) {
        FixtureStream t;
        t.name = s.s("name");
        t.file = s.s("file");
        t.scale_id = s.i("scale_id");
        t.is_cls = s.s("role") == "cls";
        t.grid_h = s.i("grid_h");
        t.grid_w = s.i("grid_w");
        t.w_padded = s.i("w_padded");
        t.features = s.i("features");
        t.stride = s.i("stride");
        t.bytes = static_cast<std::size_t>(s.n("bytes"));
        t.qp_zp = s.f("qp_zp");
        t.qp_scale = s.f("qp_scale");
        fs.streams.push_back(std::move(t));
    }
    if (fs.streams.size() != kYoloNumTensors) throw std::runtime_error("fixtures.json streams count is not 6");
    if (fs.frames <= 0) throw std::runtime_error("fixtures.json frames is 0");
    return fs;
}

inline YoloTensorLayout layout_from_fixtures(const FixtureSet& fs) {
    YoloTensorLayout L;
    L.num_classes = fs.num_classes;
    L.reg_max = fs.reg_max;
    bool seen_cls[kYoloNumScales]{}, seen_box[kYoloNumScales]{};
    for (const FixtureStream& t : fs.streams) {
        yvo::vk::YoloScaleSpec& s = L.s[t.scale_id];
        s.grid_h = t.grid_h;
        s.grid_w = t.grid_w;
        s.w_padded = t.w_padded;
        s.stride = t.stride;
        if (t.is_cls) {
            s.cls_bytes = t.bytes;
            s.cls_scale = t.qp_scale;
            s.cls_zp = t.qp_zp;
            seen_cls[t.scale_id] = true;
        } else {
            s.box_bytes = t.bytes;
            s.box_scale = t.qp_scale;
            s.box_zp = t.qp_zp;
            seen_box[t.scale_id] = true;
        }
    }
    for (int i = 0; i < kYoloNumScales; ++i)
        if (!seen_cls[i] || !seen_box[i]) throw std::runtime_error("scale " + std::to_string(i) + " has no pair");
    return L;
}

// Tensor indices 0..5 ↔ streams in fixtures.json. See yolo_postproc.hpp for the index convention.
inline std::vector<const FixtureStream*> tensor_order(const FixtureSet& fs) {
    std::vector<const FixtureStream*> v(kYoloNumTensors, nullptr);
    for (const FixtureStream& t : fs.streams)
        v[static_cast<std::size_t>(t.is_cls ? yolo_cls_tensor(t.scale_id) : yolo_box_tensor(t.scale_id))] = &t;
    for (const auto* p : v)
        if (!p) throw std::runtime_error("tensor index mapping is empty");
    return v;
}

using FrameTensors = std::array<std::vector<std::uint8_t>, kYoloNumTensors>;

inline FrameTensors load_frame(const FixtureSet& fs, int frame) {
    const auto order = tensor_order(fs);
    FrameTensors out;
    for (int i = 0; i < kYoloNumTensors; ++i) {
        char stem[32];
        std::snprintf(stem, sizeof(stem), "frame%03d_", frame);
        const std::string path = fs.dir + "/" + stem + order[static_cast<std::size_t>(i)]->file + ".bin";
        std::ifstream in(path, std::ios::binary | std::ios::ate);
        if (!in) throw std::runtime_error("could not read fixture file: " + path);
        const auto n = static_cast<std::size_t>(in.tellg());
        if (n != order[static_cast<std::size_t>(i)]->bytes)
            throw std::runtime_error("fixture size differs from metadata: " + path);
        out[static_cast<std::size_t>(i)].resize(n);
        in.seekg(0);
        in.read(reinterpret_cast<char*>(out[static_cast<std::size_t>(i)].data()), static_cast<std::streamsize>(n));
    }
    return out;
}

// Builds the CPU reference input from 6 host pointers (= tensor-index version of make_raw_set)
inline yvo::RawOutputSet make_raw_set(const YoloTensorLayout& L, const std::uint8_t* const ptr[kYoloNumTensors]) {
    yvo::RawOutputSet r;
    r.num_classes = L.num_classes;
    r.reg_max = L.reg_max;
    for (int i = 0; i < kYoloNumScales; ++i) {
        yvo::RawScale& s = r.s[i];
        s.cls = ptr[yolo_cls_tensor(i)];
        s.box = ptr[yolo_box_tensor(i)];
        s.grid_h = L.s[i].grid_h;
        s.grid_w = L.s[i].grid_w;   // P5 uses only the valid columns (20). Columns 20–23 are padding (Appendix A.6)
        s.w_padded = L.s[i].w_padded;
        s.stride = L.s[i].stride;
        s.cls_scale = L.s[i].cls_scale;
        s.cls_zp = L.s[i].cls_zp;
        s.box_scale = L.s[i].box_scale;
        s.box_zp = L.s[i].box_zp;
    }
    return r;
}

// Byte sum of the **valid region** the shader runs over (h < grid_h, c < features, w < grid_w). Same range as kChecksum
// (P5 padding columns grid_w..w_padded-1 are excluded). Compare with `YoloPostproc::checksums()`.
inline std::uint32_t cpu_valid_sum(const std::uint8_t* p, int grid_h, int features, int grid_w, int w_padded) {
    std::uint32_t sum = 0;
    for (int h = 0; h < grid_h; ++h)
        for (int c = 0; c < features; ++c) {
            const std::uint8_t* row = p + yvo::nhcw_index(h, c, 0, features, w_padded);
            for (int w = 0; w < grid_w; ++w) sum += row[w];
        }
    return sum;
}

// ─────────────────────────────────────────────────────────────────── comparison (G5 / G6)

struct Compare {
    std::size_t cpu = 0, gpu = 0;
    std::size_t matched = 0, only_cpu = 0, only_gpu = 0;
    std::size_t score_bit_mismatch = 0;
    float max_coord_err = 0.0f;
    float max_score_err = 0.0f;
    bool ok() const {
        return cpu == gpu && only_cpu == 0 && only_gpu == 0 && score_bit_mismatch == 0 &&
               max_coord_err < kCoordTolPx;
    }
};

inline std::uint32_t bits_of(float v) {
    std::uint32_t u = 0;
    std::memcpy(&u, &v, 4);
    return u;
}

inline Compare compare_candidates(const std::vector<DetectionKeyed>& cpu, const std::vector<DetectionKeyed>& gpu) {
    Compare c;
    c.cpu = cpu.size();
    c.gpu = gpu.size();
    std::map<std::uint32_t, const DetectionKeyed*> g;
    for (const auto& d : gpu) g[d.anchor_key] = &d;
    for (const auto& a : cpu) {
        auto it = g.find(a.anchor_key);
        if (it == g.end()) {
            ++c.only_cpu;
            continue;
        }
        const DetectionKeyed& b = *it->second;
        ++c.matched;
        if (bits_of(a.score) != bits_of(b.score)) ++c.score_bit_mismatch;
        c.max_score_err = std::max(c.max_score_err, std::fabs(a.score - b.score));
        c.max_coord_err = std::max(c.max_coord_err, std::fabs(a.x1 - b.x1));
        c.max_coord_err = std::max(c.max_coord_err, std::fabs(a.y1 - b.y1));
        c.max_coord_err = std::max(c.max_coord_err, std::fabs(a.x2 - b.x2));
        c.max_coord_err = std::max(c.max_coord_err, std::fabs(a.y2 - b.y2));
        g.erase(it);
    }
    c.only_gpu = g.size();
    return c;
}

// G6: sort by the total order `(score descending, anchor_key ascending)` into a form that can be compared byte by byte.
inline std::string canonical(const std::vector<DetectionKeyed>& v) {
    std::vector<DetectionKeyed> s = v;
    std::sort(s.begin(), s.end(), [](const DetectionKeyed& a, const DetectionKeyed& b) {
        if (a.score != b.score) return a.score > b.score;
        return a.anchor_key < b.anchor_key;
    });
    std::string out;
    out.reserve(s.size() * 64);
    char buf[128];
    for (const auto& d : s) {
        std::snprintf(buf, sizeof(buf), "%08x %08x %08x %08x %08x %08x %d\n", d.anchor_key, bits_of(d.score),
                      bits_of(d.x1), bits_of(d.y1), bits_of(d.x2), bits_of(d.y2), d.class_id);
        out += buf;
    }
    return out;
}

}  // namespace yvo::fixture
