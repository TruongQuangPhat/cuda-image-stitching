// Authors: TRƯƠNG QUANG PHÁT (23120318), S1-S2.
// FAST/Harris detection and oriented BRIEF descriptors.
// References: Rublee et al., ORB (2011); Rosten and Drummond, FAST (2006).

#include "orb.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace stitching::orb {
namespace {

constexpr std::array<int, 16> kCircleX = {
    0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1
};
constexpr std::array<int, 16> kCircleY = {
    -3, -3, -2, -1, 0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3
};
constexpr std::array<int, 7> kHarrisWeights = {1, 6, 15, 20, 15, 6, 1};

std::size_t imageSize(int width, int height) {
    if (width <= 0 || height <= 0) {
        throw std::invalid_argument("Image dimensions must be positive");
    }
    const auto w = static_cast<std::size_t>(width);
    const auto h = static_cast<std::size_t>(height);
    if (w > std::numeric_limits<std::size_t>::max() / h) {
        throw std::overflow_error("Image dimensions overflow size_t");
    }
    return w * h;
}

void validateImage(const GrayImage& image) {
    if (image.pixels.size() != imageSize(image.width, image.height)) {
        throw std::invalid_argument("GrayImage must contain width*height pixels");
    }
}

void validateConfig(const Config& c) {
    if (c.max_features < 0 || c.n_levels <= 0 ||
        !(c.scale_factor > 1.0f) || !std::isfinite(c.scale_factor) ||
        c.fast_threshold < 0 || c.fast_threshold > 254 ||
        c.edge_threshold < 4 || c.patch_size < 3 ||
        c.patch_size % 2 == 0 ||
        c.edge_threshold < (c.patch_size / 2) + 1 ||
        !(c.harris_k > 0.0f && c.harris_k < 0.25f)) {
        throw std::invalid_argument("Invalid ORB detection configuration");
    }
}

void validateDescriptionConfig(const Config& c) {
    if (c.patch_size != 31) {
        throw std::invalid_argument("This rBRIEF implementation requires patch_size=31");
    }
}

inline std::uint8_t pixel(const GrayImage& image, int x, int y) {
    return image.pixels[static_cast<std::size_t>(y) * image.width + x];
}

inline int clampTo(std::int64_t value, int high) {
    return static_cast<int>(std::max(std::int64_t{0},
        std::min(value, static_cast<std::int64_t>(high))));
}

// A small binomial Gaussian filter reduces aliasing before downsampling.
GrayImage blur5x5(const GrayImage& src) {
    constexpr std::array<int, 5> kWeights = {1, 4, 6, 4, 1};
    GrayImage dst{src.width, src.height, {}};
    dst.pixels.resize(imageSize(src.width, src.height));
    std::vector<std::uint16_t> horizontal(dst.pixels.size());

    for (int y = 0; y < src.height; ++y) {
        for (int x = 0; x < src.width; ++x) {
            int sum = 0;
            for (int dx = -2; dx <= 2; ++dx) {
                const int sx = clampTo(static_cast<std::int64_t>(x) + dx, src.width - 1);
                sum += kWeights[dx + 2] * pixel(src, sx, y);
            }
            horizontal[static_cast<std::size_t>(y) * src.width + x] =
                static_cast<std::uint16_t>(sum); // Max = 16*255.
        }
    }

    for (int y = 0; y < src.height; ++y) {
        for (int x = 0; x < src.width; ++x) {
            int sum = 0;
            for (int dy = -2; dy <= 2; ++dy) {
                const int sy = clampTo(static_cast<std::int64_t>(y) + dy, src.height - 1);
                sum += kWeights[dy + 2] * horizontal[
                    static_cast<std::size_t>(sy) * src.width + x];
            }
            dst.pixels[static_cast<std::size_t>(y) * src.width + x] =
                static_cast<std::uint8_t>((sum + 128) / 256);
        }
    }
    return dst;
}

// Resize with pixel-center mapping and bilinear interpolation.
GrayImage resizeBilinear(const GrayImage& src, int new_width, int new_height) {
    GrayImage dst{new_width, new_height, {}};
    dst.pixels.resize(imageSize(new_width, new_height));
    const float scale_x = static_cast<float>(src.width) / new_width;
    const float scale_y = static_cast<float>(src.height) / new_height;

    for (int y = 0; y < new_height; ++y) {
        const float sy = std::max(0.0f, (y + 0.5f) * scale_y - 0.5f);
        const int y0 = static_cast<int>(std::min(static_cast<double>(sy),
                                                 static_cast<double>(src.height - 1)));
        const int y1 = std::min(y0 + 1, src.height - 1);
        const float fy = sy - y0;
        for (int x = 0; x < new_width; ++x) {
            const float sx = std::max(0.0f, (x + 0.5f) * scale_x - 0.5f);
            const int x0 = static_cast<int>(std::min(static_cast<double>(sx),
                                                     static_cast<double>(src.width - 1)));
            const int x1 = std::min(x0 + 1, src.width - 1);
            const float fx = sx - x0;
            const float a = pixel(src, x0, y0) * (1.0f - fx) + pixel(src, x1, y0) * fx;
            const float b = pixel(src, x0, y1) * (1.0f - fx) + pixel(src, x1, y1) * fx;
            dst.pixels[static_cast<std::size_t>(y) * new_width + x] =
                static_cast<std::uint8_t>(std::lround(a * (1.0f - fy) + b * fy));
        }
    }
    return dst;
}

std::vector<PyramidLevel> buildPyramid(const GrayImage& image, const Config& c) {
    std::vector<PyramidLevel> pyramid;
    pyramid.reserve(static_cast<std::size_t>(c.n_levels));
    pyramid.push_back({image, 1.0f});

    for (int level = 1; level < c.n_levels; ++level) {
        const float scale = std::pow(c.scale_factor, static_cast<float>(level));
        if (!std::isfinite(scale)) break;
        const int width = static_cast<int>(std::lround(image.width / scale));
        const int height = static_cast<int>(std::lround(image.height / scale));
        // Equivalent to dimension < 2*edge_threshold+1, without multiplication.
        if (c.edge_threshold > (width - 1) / 2 ||
            c.edge_threshold > (height - 1) / 2) break;

        // Construct from the preceding level for predictable filtering cost.
        GrayImage filtered = blur5x5(pyramid.back().image);
        pyramid.push_back({resizeBilinear(filtered, width, height), scale});
    }
    return pyramid;
}

// True iff a 16-point FAST ring contains 9 consecutive bright or dark pixels.
bool containsNineConsecutive(std::uint16_t mask) {
    const std::uint32_t doubled = static_cast<std::uint32_t>(mask) |
                                  (static_cast<std::uint32_t>(mask) << 16);
    for (int start = 0; start < 16; ++start) {
        if (((doubled >> start) & 0x1FFu) == 0x1FFu) return true;
    }
    return false;
}

// Return FAST corner strength (0 = not a FAST-9 corner above threshold).
int fastScore(const GrayImage& image, int x, int y, int threshold) {
    const int center = pixel(image, x, y);
    std::array<int, 16> differences{};
    std::uint16_t brighter = 0;
    std::uint16_t darker = 0;

    for (int i = 0; i < 16; ++i) {
        const int d = static_cast<int>(pixel(image, x + kCircleX[i],
                                              y + kCircleY[i])) - center;
        differences[i] = d;
        if (d > threshold) brighter |= static_cast<std::uint16_t>(1u << i);
        if (d < -threshold) darker |= static_cast<std::uint16_t>(1u << i);
    }
    if (!containsNineConsecutive(brighter) && !containsNineConsecutive(darker)) {
        return 0;
    }

    int best = 0;
    for (int start = 0; start < 16; ++start) {
        int min_brighter = 255;
        int min_darker = 255;
        for (int j = 0; j < 9; ++j) {
            const int d = differences[(start + j) & 15];
            min_brighter = std::min(min_brighter, d);
            min_darker = std::min(min_darker, -d);
        }
        best = std::max(best, std::max(min_brighter, min_darker));
    }
    return best;
}

// 7x7 binomial-weighted structure tensor, using normalized central gradients.
float harrisScore(const GrayImage& image, int x, int y, float k) {
    double sxx = 0.0;
    double syy = 0.0;
    double sxy = 0.0;
    for (int dy = -3; dy <= 3; ++dy) {
        for (int dx = -3; dx <= 3; ++dx) {
            const int px = x + dx;
            const int py = y + dy;
            const double gx = (static_cast<int>(pixel(image, px + 1, py)) -
                               static_cast<int>(pixel(image, px - 1, py))) / 510.0;
            const double gy = (static_cast<int>(pixel(image, px, py + 1)) -
                               static_cast<int>(pixel(image, px, py - 1))) / 510.0;
            const double weight = kHarrisWeights[dx + 3] *
                                  kHarrisWeights[dy + 3] / 4096.0;
            sxx += weight * gx * gx;
            syy += weight * gy * gy;
            sxy += weight * gx * gy;
        }
    }
    const double trace = sxx + syy;
    return static_cast<float>(sxx * syy - sxy * sxy - k * trace * trace);
}

// Only the highest FAST score survives in its 3x3 neighborhood.
// Equal-score ties are broken deterministically by pixel scan order.
std::vector<Keypoint> detectLevel(const PyramidLevel& level, int octave,
                                  const Config& c) {
    const auto& image = level.image;
    const int margin = std::max(c.edge_threshold, 4);
    if (margin > (image.width - 1) / 2 ||
        margin > (image.height - 1) / 2) return {};
    std::vector<std::uint8_t> scores(imageSize(image.width, image.height), 0);

    for (int y = margin; y < image.height - margin; ++y) {
        for (int x = margin; x < image.width - margin; ++x) {
            scores[static_cast<std::size_t>(y) * image.width + x] =
                static_cast<std::uint8_t>(fastScore(image, x, y, c.fast_threshold));
        }
    }

    std::vector<Keypoint> candidates;
    for (int y = margin; y < image.height - margin; ++y) {
        for (int x = margin; x < image.width - margin; ++x) {
            const int score = scores[static_cast<std::size_t>(y) * image.width + x];
            if (score == 0) continue;
            bool is_max = true;
            for (int dy = -1; dy <= 1 && is_max; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    if (dx == 0 && dy == 0) continue;
                    const int other = scores[
                        static_cast<std::size_t>(y + dy) * image.width + x + dx];
                    if (other > score ||
                        (other == score && (dy < 0 || (dy == 0 && dx < 0)))) {
                        is_max = false;
                        break;
                    }
                }
            }
            if (!is_max) continue;
            candidates.push_back({x * level.scale, y * level.scale,
                                  c.patch_size * level.scale,
                                  harrisScore(image, x, y, c.harris_k),
                                  0.0f, octave});
        }
    }
    std::sort(candidates.begin(), candidates.end(),
              [](const Keypoint& a, const Keypoint& b) {
                  if (a.response != b.response) return a.response > b.response;
                  if (a.y != b.y) return a.y < b.y;
                  return a.x < b.x;
              });
    return candidates;
}

std::vector<int> quotas(int total, int n_levels, float factor) {
    std::vector<int> counts(n_levels, 0);
    std::vector<double> fractional(n_levels, 0.0);
    double sum = 0.0;
    for (int i = 0; i < n_levels; ++i) {
        fractional[i] = std::pow(static_cast<double>(factor), -i);
        sum += fractional[i];
    }
    int used = 0;
    for (int i = 0; i < n_levels; ++i) {
        const double exact = total * fractional[i] / sum;
        counts[i] = static_cast<int>(std::floor(exact));
        fractional[i] = exact - counts[i];
        used += counts[i];
    }
    std::vector<int> indices(n_levels);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&](int a, int b) {
        if (fractional[a] != fractional[b]) return fractional[a] > fractional[b];
        return a < b;
    });
    for (int i = 0; i < total - used; ++i) ++counts[indices[i]];
    return counts;
}

} // namespace

GrayImage toGrayscale(const std::uint8_t* data,
                      int width, int height, int channels,
                      std::size_t row_stride_bytes) {
    if (!data) throw std::invalid_argument("Input buffer must not be null");
    if (channels != 1 && channels != 3 && channels != 4) {
        throw std::invalid_argument("Expected grayscale, RGB or RGBA input");
    }
    const std::size_t n_pixels = imageSize(width, height);
    const std::size_t w = static_cast<std::size_t>(width);
    if (w > std::numeric_limits<std::size_t>::max() /
                static_cast<std::size_t>(channels)) {
        throw std::overflow_error("Image row size overflow");
    }
    const std::size_t min_stride = w * channels;
    if (row_stride_bytes == 0) row_stride_bytes = min_stride;
    if (row_stride_bytes < min_stride) {
        throw std::invalid_argument("Row stride smaller than active row size");
    }

    // The caller must provide at least (height-1)*stride + min_stride readable
    // bytes. A raw pointer cannot establish the actual buffer capacity.
    // Bound the full span by PTRDIFF_MAX; unused trailing padding is excluded.
    const auto span_limit = static_cast<std::size_t>(
        std::numeric_limits<std::ptrdiff_t>::max());
    const auto preceding_rows = static_cast<std::size_t>(height - 1);
    if (min_stride > span_limit ||
        (preceding_rows != 0 && row_stride_bytes >
            (span_limit - min_stride) / preceding_rows)) {
        throw std::overflow_error("Input image span exceeds ptrdiff_t range");
    }

    GrayImage output{width, height, std::vector<std::uint8_t>(n_pixels)};
    for (int y = 0; y < height; ++y) {
        const auto* row = data + static_cast<std::size_t>(y) * row_stride_bytes;
        for (int x = 0; x < width; ++x) {
            const std::size_t col = static_cast<std::size_t>(x) * channels;
            if (channels == 1) {
                output.pixels[static_cast<std::size_t>(y) * width + x] = row[col];
            } else {
                // Fixed-point approximation of ITU-R BT.601 luma.
                const int r = row[col], g = row[col + 1], b = row[col + 2];
                output.pixels[static_cast<std::size_t>(y) * width + x] =
                    static_cast<std::uint8_t>((77 * r + 150 * g + 29 * b + 128) >> 8);
            }
        }
    }
    return output;
}

DetectionResult detect(const GrayImage& grayscale, const Config& config) {
    validateImage(grayscale);
    validateConfig(config);
    DetectionResult output;
    output.pyramid = buildPyramid(grayscale, config);
    if (config.max_features == 0) return output;

    const int num_levels = static_cast<int>(output.pyramid.size());
    const auto level_quotas = quotas(config.max_features, num_levels,
                                     config.scale_factor);
    std::vector<std::vector<Keypoint>> candidates(num_levels);
    std::vector<std::size_t> taken(num_levels, 0);
    for (int level = 0; level < num_levels; ++level) {
        candidates[level] = detectLevel(output.pyramid[level], level, config);
        taken[level] = std::min(candidates[level].size(),
                                static_cast<std::size_t>(level_quotas[level]));
    }

    // Redistribute unused quota if a pyramid level did not have enough corners.
    std::vector<Keypoint> extras;
    std::size_t allocated = 0;
    for (int level = 0; level < num_levels; ++level) {
        allocated += taken[level];
        extras.insert(extras.end(), candidates[level].begin() +
                      static_cast<std::ptrdiff_t>(taken[level]), candidates[level].end());
    }
    std::sort(extras.begin(), extras.end(),
              [](const Keypoint& a, const Keypoint& b) {
                  if (a.response != b.response) return a.response > b.response;
                  if (a.octave != b.octave) return a.octave < b.octave;
                  if (a.y != b.y) return a.y < b.y;
                  return a.x < b.x;
              });
    const std::size_t deficit = static_cast<std::size_t>(config.max_features) - allocated;
    extras.resize(std::min(deficit, extras.size()));

    output.keypoints.reserve(allocated + extras.size());
    for (int level = 0; level < num_levels; ++level) {
        output.keypoints.insert(output.keypoints.end(), candidates[level].begin(),
                                candidates[level].begin() +
                                    static_cast<std::ptrdiff_t>(taken[level]));
    }
    output.keypoints.insert(output.keypoints.end(), extras.begin(), extras.end());
    std::sort(output.keypoints.begin(), output.keypoints.end(),
              [](const Keypoint& a, const Keypoint& b) {
                  if (a.octave != b.octave) return a.octave < b.octave;
                  if (a.response != b.response) return a.response > b.response;
                  if (a.y != b.y) return a.y < b.y;
                  return a.x < b.x;
              });
    return output;
}


namespace {

constexpr int kPatchRadius = 15;
constexpr int kDescriptorBits = 256;

struct PatternPoint {
    int x;
    int y;
};

struct PatternPair {
    PatternPoint a;
    PatternPoint b;
};

// Fixed, reproducible sample pattern. The reference ORB implementation uses
// its own learned pattern; this independent rBRIEF pattern is NOT bitwise
// compatible with OpenCV descriptors, but is shared by every image in this V0.
std::uint32_t nextRandom(std::uint32_t& state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

PatternPoint randomPoint(std::uint32_t& state) {
    while (true) {
        const int x = static_cast<int>(nextRandom(state) % 31u) - 15;
        const int y = static_cast<int>(nextRandom(state) % 31u) - 15;
        if (x * x + y * y <= kPatchRadius * kPatchRadius) {
            return {x, y};
        }
    }
}

const std::array<PatternPair, kDescriptorBits>& briefPattern() {
    static const auto pattern = [] {
        std::array<PatternPair, kDescriptorBits> pairs{};
        std::uint32_t state = 0xA5C31E27u;
        for (int i = 0; i < kDescriptorBits; ++i) {
            PatternPoint a, b;
            do {
                a = randomPoint(state);
                b = randomPoint(state);
            } while ((a.x - b.x) * (a.x - b.x) +
                     (a.y - b.y) * (a.y - b.y) < 25);
            pairs[static_cast<std::size_t>(i)] = {a, b};
        }
        return pairs;
    }();
    return pattern;
}

// m10 = sum(dx * intensity), m01 = sum(dy * intensity).
// The denominator m00 cancels in atan2(m01/m00, m10/m00).
float computeOrientation(const GrayImage& img, int cx, int cy) {
    std::int64_t m10 = 0;
    std::int64_t m01 = 0;
    for (int dy = -kPatchRadius; dy <= kPatchRadius; ++dy) {
        const int limit = static_cast<int>(std::floor(
            std::sqrt(static_cast<double>(kPatchRadius * kPatchRadius - dy * dy))));
        for (int dx = -limit; dx <= limit; ++dx) {
            const int value = pixel(img, cx + dx, cy + dy);
            m10 += static_cast<std::int64_t>(dx) * value;
            m01 += static_cast<std::int64_t>(dy) * value;
        }
    }
    // A truly symmetric patch has no preferred direction.
    if (m10 == 0 && m01 == 0) return 0.0f;
    return std::atan2(static_cast<float>(m01), static_cast<float>(m10));
}

Descriptor computeDescriptor(const GrayImage& blurred, int cx, int cy,
                             float angle_rad) {
    Descriptor descriptor;
    const float cs = std::cos(angle_rad);
    const float sn = std::sin(angle_rad);
    const auto& pattern = briefPattern();

    for (int i = 0; i < kDescriptorBits; ++i) {
        const auto& pair = pattern[static_cast<std::size_t>(i)];
        const int ax = static_cast<int>(std::lround(cs * pair.a.x - sn * pair.a.y));
        const int ay = static_cast<int>(std::lround(sn * pair.a.x + cs * pair.a.y));
        const int bx = static_cast<int>(std::lround(cs * pair.b.x - sn * pair.b.y));
        const int by = static_cast<int>(std::lround(sn * pair.b.x + cs * pair.b.y));
        const auto ia = pixel(blurred, cx + ax, cy + ay);
        const auto ib = pixel(blurred, cx + bx, cy + by);
        if (ia < ib) {
            descriptor.bytes[static_cast<std::size_t>(i / 8)] |=
                static_cast<std::uint8_t>(1u << (i % 8));
        }
    }
    return descriptor;
}

// Reject malformed external inputs, rather than risk going outside the patch.
std::pair<int, int> keypointInLevel(const Keypoint& kp,
                                    const PyramidLevel& level) {
    validateImage(level.image);
    if (!(level.scale > 0.0f) || !std::isfinite(level.scale) ||
        !std::isfinite(kp.x) || !std::isfinite(kp.y)) {
        throw std::invalid_argument("Invalid keypoint or pyramid scale");
    }
    const double x = static_cast<double>(kp.x) / level.scale;
    const double y = static_cast<double>(kp.y) / level.scale;
    if (x < kPatchRadius + 1 || y < kPatchRadius + 1 ||
        x >= level.image.width - kPatchRadius - 1 ||
        y >= level.image.height - kPatchRadius - 1) {
        throw std::out_of_range("Keypoint is too close to the image boundary");
    }
    const int cx = static_cast<int>(std::lround(x));
    const int cy = static_cast<int>(std::lround(y));
    return {cx, cy};
}

} // namespace

DescriptionResult describe(const DetectionResult& detection,
                           const Config& config) {
    validateDescriptionConfig(config);
    if (detection.pyramid.empty()) {
        throw std::invalid_argument("DetectionResult must contain a pyramid");
    }
    DescriptionResult output;
    output.keypoints = detection.keypoints;
    output.descriptors.reserve(output.keypoints.size());
    if (output.keypoints.empty()) return output;

    // One smoothed image per occupied octave; reuse it across descriptors.
    std::vector<GrayImage> blurred(detection.pyramid.size());
    std::vector<bool> built(detection.pyramid.size(), false);

    for (auto& kp : output.keypoints) {
        if (kp.octave < 0 ||
            static_cast<std::size_t>(kp.octave) >= detection.pyramid.size()) {
            throw std::out_of_range("Keypoint has invalid octave");
        }
        const std::size_t octave = static_cast<std::size_t>(kp.octave);
        const auto& level = detection.pyramid[octave];
        const auto [cx, cy] = keypointInLevel(kp, level);
        if (!built[octave]) {
            blurred[octave] = blur5x5(level.image);
            built[octave] = true;
        }
        kp.angle_rad = computeOrientation(level.image, cx, cy);
        output.descriptors.push_back(computeDescriptor(
            blurred[octave], cx, cy, kp.angle_rad));
    }
    return output;
}

DescriptionResult extract(const GrayImage& grayscale, const Config& config) {
    validateDescriptionConfig(config);
    auto detection = detect(grayscale, config);
    return describe(detection, config);
}

} // namespace stitching::orb
