// Authors: HUỲNH TẤN PHƯỚC (23120334), S6.
// Blending configuration and results.
// Reference: Szeliski, Computer Vision: Algorithms and Applications (2022), 2nd ed.

#pragma once

#include "warping.h"

namespace stitching::blending {

struct BlendResult {
    warping::RgbImage panorama;
    // Successful compositing, including disjoint/empty masks; not a geometry
    // or seam-quality certificate. On logical failure, panorama stays empty.
    bool valid = false;
    std::string error_message;
    // True only when the requested DP seam has a complete path. A successful
    // fallback has valid=true, seam_applied=false, and a nonempty info_message.
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
    // Channels with no qualifying overlap samples keep gain 1; one sample is
    // sufficient. Seam fallback retains this setting, not the default config.
    bool compensate_exposure = false;
};

// S6: feather weights are 8-neighbor chamfer distances (1, sqrt(2)) to the
// nearest invalid pixel or canvas exterior. This approximates Euclidean distance
// in two sequential passes. Exclusive A is copied exactly; exclusive B is copied
// or exposure-transformed. Feather overlap is normalized by the sum of weights,
// and background is black. Keeps canvas size.
// Accepts any nonzero mask value as valid; validates all buffer dimensions.
// MinimumErrorSeam optionally finds a low-color-difference path through overlap
// by dynamic programming, oriented along the overlap's long direction relative
// to source displacement. A path must span the overlap bounding box along the
// chosen direction. Disconnected overlap may still contain such a path; neither
// full overlap connectivity nor visual seam quality is tested. Only absence of
// a complete path causes feather fallback with an info_message. This reduces
// double exposure, but cannot correct parallax geometry. Logical failures return
// valid=false; allocation exceptions may propagate.
BlendResult blend(const warping::WarpResult& warped, const BlendConfig& config = BlendConfig{});

}  // namespace stitching::blending
