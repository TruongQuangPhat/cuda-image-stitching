// Authors: HUỲNH TẤN PHƯỚC (23120334), S6.
// Feather blending and optional minimum-error seam with exposure compensation.
// Reference: Szeliski, Computer Vision: Algorithms and Applications (2022), 2nd ed.

#include "blending.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace stitching::blending {
using warping::RgbImage;
using warping::WarpResult;
namespace {

bool validImage(const RgbImage& image, std::size_t& count) {
    if (image.width <= 0 || image.height <= 0) return false;
    const auto width = static_cast<std::size_t>(image.width);
    const auto height = static_cast<std::size_t>(image.height);
    const auto limit = std::numeric_limits<std::size_t>::max();
    if (width > limit / height) return false;
    count = width * height;
    return count <= limit / 3 && image.pixels.size() == count * 3;
}

std::vector<float> featherWeights(const std::vector<std::uint8_t>& mask,
                                  int width, int height) {
    constexpr float diagonal = 1.41421356237f;
    std::vector<float> distance(mask.size(), 0.0f);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const auto i = static_cast<std::size_t>(y) * width + x;
            if (!mask[i]) continue;
            float d = (x == 0 || y == 0 || x == width-1 || y == height-1)
                ? 1.0f : std::numeric_limits<float>::infinity();
            if (x > 0) d = std::min(d, distance[i-1] + 1.0f);
            if (y > 0) {
                d = std::min(d, distance[i-width] + 1.0f);
                if (x > 0) d = std::min(d, distance[i-width-1] + diagonal);
                if (x+1 < width) d = std::min(d, distance[i-width+1] + diagonal);
            }
            distance[i] = d;
        }
    }
    for (int y = height-1; y >= 0; --y) {
        for (int x = width-1; x >= 0; --x) {
            const auto i = static_cast<std::size_t>(y) * width + x;
            if (!mask[i]) continue;
            float d = distance[i];
            if (x+1 < width) d = std::min(d, distance[i+1] + 1.0f);
            if (y+1 < height) {
                d = std::min(d, distance[i+width] + 1.0f);
                if (x > 0) d = std::min(d, distance[i+width-1] + diagonal);
                if (x+1 < width) d = std::min(d, distance[i+width+1] + diagonal);
            }
            distance[i] = d;
        }
    }
    return distance;
}

struct Seam {
    std::vector<int> path;
    bool horizontal = false;
    bool a_first = true;
    int along_start = 0;
};

double exposed(std::uint8_t value, double gain) {
    // Gain in intensity-odds space preserves both black and white without clipping.
    return gain == 1.0 ? value : 255.0*gain*value/(255.0+(gain-1.0)*value);
}

Seam minimumErrorSeam(const WarpResult& warped, const std::array<double,3>& gains) {
    const int width = warped.warped_a.width, height = warped.warped_a.height;
    int x0 = width, y0 = height, x1 = -1, y1 = -1;
    double ax = 0, ay = 0, bx = 0, by = 0, na = 0, nb = 0;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const auto i = static_cast<std::size_t>(y)*width+x;
            if (warped.mask_a[i]) { ax += x; ay += y; ++na; }
            if (warped.mask_b[i]) { bx += x; by += y; ++nb; }
            if (warped.mask_a[i] && warped.mask_b[i]) {
                x0 = std::min(x0,x); x1 = std::max(x1,x);
                y0 = std::min(y0,y); y1 = std::max(y1,y);
            }
        }
    }
    Seam seam;
    if (x1 < x0) return seam;
    // A nonempty overlap guarantees both mask populations are positive.
    seam.horizontal = std::abs(ay/na-by/nb) > std::abs(ax/na-bx/nb);
    seam.a_first = seam.horizontal ? ay/na <= by/nb : ax/na <= bx/nb;
    seam.along_start = seam.horizontal ? x0 : y0;
    const int cross_start = seam.horizontal ? y0 : x0;
    const int span = seam.horizontal ? y1-y0+1 : x1-x0+1;
    const int length = seam.horizontal ? x1-x0+1 : y1-y0+1;
    const auto index = [&](int along, int cross) {
        const int x = seam.horizontal ? along+x0 : cross+x0;
        const int y = seam.horizontal ? cross+y0 : along+y0;
        return static_cast<std::size_t>(y)*width+x;
    };
    const float infinity = std::numeric_limits<float>::infinity();
    std::vector<float> previous(static_cast<std::size_t>(span), infinity), current(previous);
    std::vector<std::int8_t> parent(static_cast<std::size_t>(length)*span, 0);
    for (int along = 0; along < length; ++along) {
        std::fill(current.begin(), current.end(), infinity);
        for (int cross = 0; cross < span; ++cross) {
            const auto i = index(along,cross);
            if (!warped.mask_a[i] || !warped.mask_b[i]) continue;
            float cost = 1.0f;
            for (std::size_t c = 0; c < 3; ++c)
                cost += static_cast<float>(std::abs(warped.warped_a.pixels[i*3+c] -
                    exposed(warped.warped_b.pixels[i*3+c],gains[c])) / 3.0);
            int best = cross;
            if (along > 0) {
                if (cross > 0 && previous[cross-1] < previous[best]) best = cross-1;
                if (cross+1 < span && previous[cross+1] < previous[best]) best = cross+1;
                cost += previous[best];
            }
            current[cross] = cost;
            parent[static_cast<std::size_t>(along)*span+cross] =
                static_cast<std::int8_t>(best-cross);
        }
        previous.swap(current);
    }
    int cross = static_cast<int>(std::min_element(previous.begin(),previous.end())-previous.begin());
    if (!std::isfinite(previous[cross])) return seam;
    seam.path.resize(static_cast<std::size_t>(length));
    for (int along = length-1; along >= 0; --along) {
        seam.path[along] = cross+cross_start;
        cross += parent[static_cast<std::size_t>(along)*span+cross];
    }
    return seam;
}

}  // namespace

BlendResult blend(const WarpResult& warped, const BlendConfig& config) {
    BlendResult result;
    if ((config.mode != BlendMode::Feather && config.mode != BlendMode::MinimumErrorSeam) ||
        !std::isfinite(config.seam_feather_radius) || config.seam_feather_radius < 0) {
        result.error_message = "Invalid blend mode or seam feather radius";
        return result;
    }
    if (!warped.valid) {
        result.error_message = "WarpResult.valid is false";
        return result;
    }
    std::size_t count = 0, count_b = 0;
    if (!validImage(warped.warped_a, count) || !validImage(warped.warped_b, count_b) ||
        warped.warped_a.width != warped.warped_b.width ||
        warped.warped_a.height != warped.warped_b.height ||
        warped.mask_a.size() != count || warped.mask_b.size() != count) {
        result.error_message = "Warped RGB images and masks must share canvas dimensions";
        return result;
    }
    const int width = warped.warped_a.width, height = warped.warped_a.height;
    if (config.compensate_exposure) {
        std::array<double,3> sum_a{}, sum_b{}, samples{};
        for (std::size_t i = 0; i < count; ++i) {
            if (!warped.mask_a[i] || !warped.mask_b[i]) continue;
            for (std::size_t c = 0; c < 3; ++c) {
                const int a = warped.warped_a.pixels[i*3+c], b = warped.warped_b.pixels[i*3+c];
                if (a > 8 && a < 247 && b > 8 && b < 247) {
                    sum_a[c] += a; sum_b[c] += b; ++samples[c];
                }
            }
        }
        for (std::size_t c = 0; c < 3; ++c)
            if (sum_b[c] > 0) result.exposure_gains[c] = std::clamp(
                sum_a[c]*(255*samples[c]-sum_b[c]) /
                (sum_b[c]*(255*samples[c]-sum_a[c])),0.5,2.0);
    }
    Seam seam;
    if (config.mode == BlendMode::MinimumErrorSeam) {
        seam = minimumErrorSeam(warped,result.exposure_gains);
        result.seam_applied = !seam.path.empty();
        if (!result.seam_applied)
            result.info_message = "No continuous overlap seam; used mask-distance feather blending";
    }
    const auto weights_a = result.seam_applied ? std::vector<float>{} :
        featherWeights(warped.mask_a, width, height);
    const auto weights_b = result.seam_applied ? std::vector<float>{} :
        featherWeights(warped.mask_b, width, height);
    result.panorama = {width, height, std::vector<std::uint8_t>(count*3, 0)};
    for (std::size_t i = 0; i < count; ++i) {
        const bool a = warped.mask_a[i] != 0, b = warped.mask_b[i] != 0;
        if (a && b) {
            double alpha = 0.0;
            if (result.seam_applied) {
                const int x = static_cast<int>(i % static_cast<std::size_t>(width));
                const int y = static_cast<int>(i / static_cast<std::size_t>(width));
                const int along = (seam.horizontal ? x : y)-seam.along_start;
                const int cross = seam.horizontal ? y : x;
                // Every overlap pixel lies in the bounding box spanned by path.
                const double side = (cross-seam.path[along]) * (seam.a_first ? 1.0 : -1.0);
                alpha = config.seam_feather_radius == 0 ? (side <= 0 ? 1.0 : 0.0) :
                    std::clamp(0.5-side/(2*config.seam_feather_radius),0.0,1.0);
            } else {
                alpha = weights_a[i] / (static_cast<double>(weights_a[i])+weights_b[i]);
            }
            for (std::size_t channel = 0; channel < 3; ++channel) {
                const auto offset = i*3 + channel;
                result.panorama.pixels[offset] = static_cast<std::uint8_t>(std::clamp(
                    std::lround(alpha*warped.warped_a.pixels[offset] + (1-alpha)*
                        exposed(warped.warped_b.pixels[offset],result.exposure_gains[channel])),
                    0L, 255L));
            }
        } else if (a || b) {
            const auto& source = a ? warped.warped_a : warped.warped_b;
            for (std::size_t channel = 0; channel < 3; ++channel)
                result.panorama.pixels[i*3+channel] = static_cast<std::uint8_t>(std::clamp(
                    std::lround(exposed(source.pixels[i*3+channel],a ? 1.0 : result.exposure_gains[channel])),0L,255L));
        }
    }
    result.valid = true;
    return result;
}

}  // namespace stitching::blending
