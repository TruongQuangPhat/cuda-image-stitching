// Stage 1 (ORB keypoint detection): original educational implementation.
// Algorithm references: Rublee et al., ORB, ICCV 2011; Rosten et al., FAST.

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

inline std::uint8_t pixel(const GrayImage& image, int x, int y) {
    return image.pixels[static_cast<std::size_t>(y) * image.width + x];
}

inline int clampTo(int value, int high) {
    return std::max(0, std::min(value, high));
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
                sum += kWeights[dx + 2] * pixel(src, clampTo(x + dx, src.width - 1), y);
            }
            horizontal[static_cast<std::size_t>(y) * src.width + x] =
                static_cast<std::uint16_t>(sum); // Max = 16*255.
        }
    }

    for (int y = 0; y < src.height; ++y) {
        for (int x = 0; x < src.width; ++x) {
            int sum = 0;
            for (int dy = -2; dy <= 2; ++dy) {
                sum += kWeights[dy + 2] * horizontal[
                    static_cast<std::size_t>(clampTo(y + dy, src.height - 1)) * src.width + x];
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
        const int y0 = std::min(static_cast<int>(sy), src.height - 1);
        const int y1 = std::min(y0 + 1, src.height - 1);
        const float fy = sy - y0;
        for (int x = 0; x < new_width; ++x) {
            const float sx = std::max(0.0f, (x + 0.5f) * scale_x - 0.5f);
            const int x0 = std::min(static_cast<int>(sx), src.width - 1);
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
        if (width < 2 * c.edge_threshold + 1 ||
            height < 2 * c.edge_threshold + 1) break;

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
    if (image.width <= 2 * margin || image.height <= 2 * margin) return {};
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

} // namespace stitching::orb
