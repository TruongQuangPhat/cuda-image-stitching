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
    // Zero selects one source per side; positive values feather within this radius in pixels.
    double seam_feather_radius = 0.0;
    // Optional bounded per-channel intensity-odds gain from unsaturated overlap
    // means, using f(v)=255*g*v/(255+(g-1)*v) to preserve black/white without clipping.
    // Applies to B in both exclusive and overlap regions; masks/background stay unchanged.
    // Channels with no qualifying overlap samples keep gain 1; one sample is
    // sufficient. Seam fallback retains this setting, not the default config.
    bool compensate_exposure = false;
};

// S6: two-pass 8-neighbor chamfer weights (1, sqrt(2)) approximate distance to
// invalid pixels or canvas exterior. Overlap weights are normalized; exclusive
// A is copied, exclusive B is copied or exposure-transformed, background is black.
// Keeps canvas size; any nonzero mask value is valid. Buffer dimensions are checked.
// MinimumErrorSeam uses color-difference DP; mask-centroid displacement selects
// its direction. A path must span the overlap bounding box in that direction.
// Disconnected overlap can contain a complete path; connectivity and seam quality
// are not certified. No complete path triggers feather fallback with info_message.
// Seam cannot correct parallax. Logical failures return valid=false; allocation
// exceptions may propagate.
BlendResult blend(const warping::WarpResult& warped, const BlendConfig& config = BlendConfig{});

}  // namespace stitching::blending
