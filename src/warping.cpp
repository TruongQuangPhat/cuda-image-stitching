// Authors: HUỲNH TẤN PHƯỚC (23120334), S5.
// Canvas construction and inverse warping with bilinear interpolation.
// Reference: Szeliski, Computer Vision: Algorithms and Applications (2022), 2nd ed.

#include "warping.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace stitching::warping {
namespace {

using Mat3 = std::array<double, 9>;
constexpr double kRelativeEpsilon = 64.0 * std::numeric_limits<double>::epsilon();

bool validImage(const RgbImage& image, std::size_t& count) {
    if (image.width <= 0 || image.height <= 0) return false;
    const auto width = static_cast<std::size_t>(image.width);
    const auto height = static_cast<std::size_t>(image.height);
    const auto limit = std::numeric_limits<std::size_t>::max();
    if (width > limit / height) return false;
    count = width * height;
    return count <= limit / 3 && image.pixels.size() == count * 3;
}

bool normalize(const Mat3& input, Mat3& output) {
    double scale = 0.0;
    for (double value : input) {
        if (!std::isfinite(value)) return false;
        scale = std::max(scale, std::abs(value));
    }
    if (scale == 0.0) return false;
    for (std::size_t i = 0; i < input.size(); ++i) output[i] = input[i] / scale;
    return true;
}

bool inverse(const Mat3& h, Mat3& output) {
    const Mat3 adjugate = {
        h[4]*h[8] - h[5]*h[7], h[2]*h[7] - h[1]*h[8], h[1]*h[5] - h[2]*h[4],
        h[5]*h[6] - h[3]*h[8], h[0]*h[8] - h[2]*h[6], h[2]*h[3] - h[0]*h[5],
        h[3]*h[7] - h[4]*h[6], h[1]*h[6] - h[0]*h[7], h[0]*h[4] - h[1]*h[3]
    };
    const double t0 = h[0] * adjugate[0];
    const double t1 = h[1] * adjugate[3];
    const double t2 = h[2] * adjugate[6];
    const double determinant = t0 + t1 + t2;
    const double scale = std::abs(t0) + std::abs(t1) + std::abs(t2);
    if (scale == 0.0 || std::abs(determinant) <= kRelativeEpsilon * scale) return false;
    // Homogeneous projection ignores a common scale, so division by det is unnecessary.
    return normalize(adjugate, output);
}

bool project(const Mat3& h, double x, double y, double& px, double& py) {
    const double denominator = h[6]*x + h[7]*y + h[8];
    const double scale = std::abs(h[6]*x) + std::abs(h[7]*y) + std::abs(h[8]);
    if (std::abs(denominator) <= kRelativeEpsilon * scale) return false;
    px = (h[0]*x + h[1]*y + h[2]) / denominator;
    py = (h[3]*x + h[4]*y + h[5]) / denominator;
    return std::isfinite(px) && std::isfinite(py);
}

bool sampleBilinear(const RgbImage& image, double x, double y, std::uint8_t* output) {
    // Accept round-off at an edge, without extending the image by a full pixel.
    const double tolerance = kRelativeEpsilon * std::max(image.width, image.height);
    if (x < -tolerance || y < -tolerance ||
        x > image.width - 1.0 + tolerance || y > image.height - 1.0 + tolerance) return false;
    x = std::clamp(x, 0.0, image.width - 1.0);
    y = std::clamp(y, 0.0, image.height - 1.0);
    const int x0 = static_cast<int>(std::floor(x));
    const int y0 = static_cast<int>(std::floor(y));
    const int x1 = std::min(x0 + 1, image.width - 1);
    const int y1 = std::min(y0 + 1, image.height - 1);
    const double fx = x - x0;
    const double fy = y - y0;
    const auto index = [&image](int sx, int sy) {
        return (static_cast<std::size_t>(sy) * image.width + sx) * 3;
    };
    for (std::size_t channel = 0; channel < 3; ++channel) {
        const double top = (1.0-fx)*image.pixels[index(x0,y0)+channel] +
                           fx*image.pixels[index(x1,y0)+channel];
        const double bottom = (1.0-fx)*image.pixels[index(x0,y1)+channel] +
                              fx*image.pixels[index(x1,y1)+channel];
        output[channel] = static_cast<std::uint8_t>(std::clamp(
            std::lround((1.0-fy)*top + fy*bottom), 0L, 255L));
    }
    return true;
}

}  // namespace

WarpResult warp(const RgbImage& image_a, const RgbImage& image_b,
                const homography::HomographyResult& homography, const WarpConfig& config) {
    WarpResult result;
    std::size_t count_a = 0, count_b = 0;
    if (!validImage(image_a, count_a) || !validImage(image_b, count_b)) {
        result.error_message = "Input images must contain width*height*3 RGB bytes";
        return result;
    }
    if (!homography.valid) {
        result.error_message = "HomographyResult.valid is false";
        return result;
    }
    if (config.max_canvas_pixels == 0) {
        result.error_message = "max_canvas_pixels must be positive";
        return result;
    }
    Mat3 h{}, inverse_h{};
    if (!normalize(homography.H, h) || !inverse(h, inverse_h)) {
        result.error_message = "Homography is non-finite, singular or numerically degenerate";
        return result;
    }

    double min_x = 0.0, min_y = 0.0;
    double max_x = image_a.width - 1.0, max_y = image_a.height - 1.0;
    const std::array<std::array<double, 2>, 4> corners = {{{0,0},
        {image_b.width-1.0,0}, {image_b.width-1.0,image_b.height-1.0},
        {0,image_b.height-1.0}}};
    const bool positive_denominator = h[8] > 0.0;
    for (const auto& corner : corners) {
        double x = 0.0, y = 0.0;
        const double denominator = h[6]*corner[0] + h[7]*corner[1] + h[8];
        // A denominator is affine: opposite signs at corners imply a horizon inside B.
        if (!project(h, corner[0], corner[1], x, y) ||
            (denominator > 0.0) != positive_denominator) {
            result.error_message = "Projective horizon intersects image B";
            return result;
        }
        min_x = std::min(min_x, x);
        min_y = std::min(min_y, y);
        max_x = std::max(max_x, x);
        max_y = std::max(max_y, y);
    }
    const double left = std::floor(min_x), top = std::floor(min_y);
    const double canvas_width = std::ceil(max_x) - left + 1.0;
    const double canvas_height = std::ceil(max_y) - top + 1.0;
    const double int_limit = std::numeric_limits<int>::max();
    if (!std::isfinite(canvas_width) || !std::isfinite(canvas_height) ||
        canvas_width < 1.0 || canvas_height < 1.0 ||
        canvas_width > int_limit || canvas_height > int_limit) {
        result.error_message = "Canvas dimensions are not representable";
        return result;
    }
    const int width = static_cast<int>(canvas_width);
    const int height = static_cast<int>(canvas_height);
    const auto w = static_cast<std::size_t>(width);
    const auto height_size = static_cast<std::size_t>(height);
    if (w > config.max_canvas_pixels / height_size ||
        w > std::numeric_limits<std::size_t>::max() / height_size / 3) {
        result.error_message = "Canvas exceeds max_canvas_pixels or RGB buffer size limit";
        return result;
    }
    const auto count = w * height_size;
    result.warped_a = {width, height, std::vector<std::uint8_t>(count*3, 0)};
    result.warped_b = {width, height, std::vector<std::uint8_t>(count*3, 0)};
    result.mask_a.assign(count, 0);
    result.mask_b.assign(count, 0);
    result.canvas_transform = {1,0,-left, 0,1,-top, 0,0,1};
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const auto i = static_cast<std::size_t>(y) * width + x;
            const double ax = x + left, ay = y + top;
            if (ax >= 0 && ay >= 0 && ax < image_a.width && ay < image_a.height) {
                const auto source = (static_cast<std::size_t>(ay) * image_a.width +
                                     static_cast<std::size_t>(ax)) * 3;
                std::copy_n(image_a.pixels.data()+source, 3, result.warped_a.pixels.data()+i*3);
                result.mask_a[i] = 1;
            }
            double bx = 0.0, by = 0.0;
            if (project(inverse_h, ax, ay, bx, by) &&
                sampleBilinear(image_b, bx, by, result.warped_b.pixels.data()+i*3)) {
                result.mask_b[i] = 1;
            }
        }
    }
    result.valid = true;
    return result;
}

}  // namespace stitching::warping
