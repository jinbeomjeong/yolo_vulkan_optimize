// hvs_psnr: reports PSNR, per-channel mean signed error, and max error between two PPM (P6, 8-bit) images.
//   - mean signed error is used to tell color range (limited/full) errors from matrix (601/709) errors
//     (a range error lifts darks and crushes highlights, changing luma variance → compare std dev in --stats).
// usage: hvs_psnr ref.ppm test.ppm [--roi x y w h]
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

struct Img {
    int w = 0, h = 0;
    std::vector<std::uint8_t> px;
};

bool read_ppm(const char* path, Img& img) {
    FILE* f = std::fopen(path, "rb");
    if (!f) return false;
    char magic[3] = {};
    int maxv = 0;
    auto skip_ws_comments = [&]() {
        int c;
        while ((c = std::fgetc(f)) != EOF) {
            if (c == '#') {
                while ((c = std::fgetc(f)) != EOF && c != '\n') {}
            } else if (c != ' ' && c != '\n' && c != '\r' && c != '\t') {
                std::ungetc(c, f);
                return;
            }
        }
    };
    bool ok = std::fscanf(f, "%2s", magic) == 1 && std::string(magic) == "P6";
    if (ok) {
        skip_ws_comments();
        ok = std::fscanf(f, "%d", &img.w) == 1;
        skip_ws_comments();
        ok = ok && std::fscanf(f, "%d", &img.h) == 1;
        skip_ws_comments();
        ok = ok && std::fscanf(f, "%d", &maxv) == 1 && maxv == 255 && img.w > 0 && img.h > 0;
        std::fgetc(f); // 1 whitespace byte after the header
    }
    if (ok) {
        img.px.resize(static_cast<std::size_t>(img.w) * static_cast<std::size_t>(img.h) * 3);
        ok = std::fread(img.px.data(), 1, img.px.size(), f) == img.px.size();
    }
    std::fclose(f);
    return ok;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s ref.ppm test.ppm [--roi x y w h]\n", argv[0]);
        return 2;
    }
    Img a, b;
    if (!read_ppm(argv[1], a) || !read_ppm(argv[2], b)) {
        std::fprintf(stderr, "PPM read failed\n");
        return 1;
    }
    if (a.w != b.w || a.h != b.h) {
        std::fprintf(stderr, "size mismatch %dx%d vs %dx%d\n", a.w, a.h, b.w, b.h);
        return 1;
    }
    int rx = 0, ry = 0, rw = a.w, rh = a.h;
    for (int i = 3; i < argc; ++i) {
        if (std::string(argv[i]) == "--roi" && i + 4 < argc) {
            rx = std::atoi(argv[i + 1]);
            ry = std::atoi(argv[i + 2]);
            rw = std::atoi(argv[i + 3]);
            rh = std::atoi(argv[i + 4]);
            i += 4;
        }
    }
    double se[3] = {}, sd[3] = {}, sa[3] = {}, sb[3] = {}, sa2[3] = {}, sb2[3] = {};
    int maxd = 0;
    std::size_t n = 0;
    for (int y = ry; y < ry + rh && y < a.h; ++y) {
        for (int x = rx; x < rx + rw && x < a.w; ++x) {
            const std::size_t o = (static_cast<std::size_t>(y) * static_cast<std::size_t>(a.w) + static_cast<std::size_t>(x)) * 3;
            for (int c = 0; c < 3; ++c) {
                const int va = a.px[o + static_cast<std::size_t>(c)], vb = b.px[o + static_cast<std::size_t>(c)];
                const int d = vb - va;
                se[c] += static_cast<double>(d) * d;
                sd[c] += d;
                sa[c] += va;
                sb[c] += vb;
                sa2[c] += static_cast<double>(va) * va;
                sb2[c] += static_cast<double>(vb) * vb;
                if (std::abs(d) > maxd) maxd = std::abs(d);
            }
            ++n;
        }
    }
    if (n == 0) return 1;
    const double nn = static_cast<double>(n);
    const double mse = (se[0] + se[1] + se[2]) / (3 * nn);
    const double psnr = mse > 0 ? 10 * std::log10(255.0 * 255.0 / mse) : 99.0;
    std::printf("roi %d,%d %dx%d  pixels %zu\n", rx, ry, rw, rh, n);
    std::printf("PSNR %.2f dB  (MSE %.3f)  max|d| %d\n", psnr, mse, maxd);
    const char* nm = "RGB";
    for (int c = 0; c < 3; ++c) {
        const double ma = sa[c] / nn, mb = sb[c] / nn;
        const double stda = std::sqrt(std::max(0.0, sa2[c] / nn - ma * ma));
        const double stdb = std::sqrt(std::max(0.0, sb2[c] / nn - mb * mb));
        const double m = se[c] / nn;
        std::printf("  %c: PSNR %.2f dB  mean(test-ref) %+.3f  mean ref %.2f test %.2f  std ref %.2f test %.2f\n", nm[c],
                    m > 0 ? 10 * std::log10(255.0 * 255.0 / m) : 99.0, sd[c] / nn, ma, mb, stda, stdb);
    }
    return 0;
}
