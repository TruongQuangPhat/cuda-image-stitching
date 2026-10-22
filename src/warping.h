// Authors: HUỲNH TẤN PHƯỚC (23120334), S5.
// RGB images, masks, canvas geometry and warping interfaces.
// Reference: Szeliski, Computer Vision: Algorithms and Applications (2022), 2nd ed.

#pragma once

#include "homography.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace stitching::warping {

// Owned, tightly packed, interleaved RGB8 (not BGR). Pixel centers use integer
// coordinates [0,width-1] x [0,height-1], matching the ORB coordinate system.
struct RgbImage {
    int width = 0;
    int height = 0;
    std::vector<std::uint8_t> pixels;
};

struct WarpConfig {
    // Bounds memory use when an unstable H produces an enormous canvas.
    // Warped RGB images and masks use eight bytes per canvas pixel in total.
    std::size_t max_canvas_pixels = 16u * 1024u * 1024u;
};

struct WarpResult {
    RgbImage warped_a;
    RgbImage warped_b;
    // One byte per canvas pixel: 1 = valid source sample, 0 = background.
    // Color values, including genuine black pixels, never determine validity.
    std::vector<std::uint8_t> mask_a;
    std::vector<std::uint8_t> mask_b;
    // Row-major T mapping A coordinates to canvas coordinates. B uses T * H_B_to_A.
    std::array<double, 9> canvas_transform{};
    bool valid = false;
    std::string error_message;
};

// S5: union of projected corner bounds, canvas translation, inverse mapping and
// bilinear RGB sampling. A is copied using T; B is sampled using (T*H)^-1.
// Rejects invalid images/H/config, singular H, a projective horizon crossing B,
// and canvases exceeding the configured limit. Logical failures return valid=false.
WarpResult warp(const RgbImage& image_a,
                const RgbImage& image_b,
                const homography::HomographyResult& homography,
                const WarpConfig& config = WarpConfig{});

}  // namespace stitching::warping
