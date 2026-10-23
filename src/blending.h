// Authors: HUỲNH TẤN PHƯỚC (23120334), S6.
// Blending configuration and results.
// Reference: Szeliski, Computer Vision: Algorithms and Applications (2022), 2nd ed.

#pragma once

#include "warping.h"

namespace stitching::blending {

struct BlendResult {
    warping::RgbImage panorama;
    bool valid = false;
    std::string error_message;
    bool seam_applied = false;
    std::string info_message;
    std::array<double,3> exposure_gains{1,1,1};
};

enum class BlendMode { Feather, MinimumErrorSeam };
struct BlendConfig {
    BlendMode mode = BlendMode::Feather;
    // Zero selects a single source on each side of the seam, avoiding averaging
    // displaced objects. A positive value feathers within this many pixels.
    double seam_feather_radius = 0.0;
    // Optional bounded per-channel intensity-odds gain from unsaturated overlap
    // means, using f(v)=255*g*v/(255+(g-1)*v) to preserve black/white without clipping.
    // Applies to B in both exclusive and overlap regions; masks/background stay unchanged.
    bool compensate_exposure = false;
};

// S6: feather weights are 8-neighbor chamfer distances (1, sqrt(2)) to the
// nearest invalid pixel or canvas exterior. This approximates Euclidean distance
// in two sequential passes. Exclusive regions are copied exactly, overlap is
// normalized by the sum of weights, and background is black. Keeps canvas size.
// Accepts any nonzero mask value as valid; validates all buffer dimensions.
// MinimumErrorSeam optionally finds a low-color-difference path through overlap
// by dynamic programming, oriented along the overlap's long direction relative
// to source displacement. Disconnected overlap falls back to feather with an
// info_message. This reduces double exposure, but cannot correct parallax geometry.
BlendResult blend(const warping::WarpResult& warped, const BlendConfig& config = BlendConfig{});

}  // namespace stitching::blending
