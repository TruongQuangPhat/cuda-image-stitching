// Authors: TRƯƠNG QUANG PHÁT (23120318), S1-S2.
// ORB-style detection and description interfaces.
// References: Rublee et al., ORB (2011); Rosten and Drummond, FAST (2006).

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace stitching::orb {

struct GrayImage {
    int width = 0;
    int height = 0;
    std::vector<std::uint8_t> pixels;
};

struct Keypoint {
    float x = 0.0f;
    float y = 0.0f;
    float size = 0.0f;       // Nominal patch diameter in original-image pixels.
    float response = 0.0f;   // Harris score for ranking.
    float angle_rad = 0.0f;  // Stage 2 estimates angle in [-pi, pi].
    int octave = 0;         // Pyramid level at which FAST detected the point.
};

struct Config {
    int max_features = 2000;  // Across all pyramid levels, per input image.
    int n_levels = 8;
    float scale_factor = 1.2f;
    int fast_threshold = 20;  // Strict brightness difference (0..254).
    int edge_threshold = 31;  // Exclusion border at every pyramid level.
    int patch_size = 31;      // Detection size metadata/border validation; S2 requires 31.
    float harris_k = 0.04f;
};

struct PyramidLevel {
    GrayImage image;
    float scale = 1.0f;
};

struct DetectionResult {
    std::vector<PyramidLevel> pyramid;
    std::vector<Keypoint> keypoints;
};

// A 256-bit binary rBRIEF descriptor (32 bytes).
// Bit i is stored in bytes[i / 8], least-significant bit first.
struct Descriptor {
    std::array<std::uint8_t, 32> bytes{};
};

struct DescriptionResult {
    std::vector<Keypoint> keypoints;
    std::vector<Descriptor> descriptors;
};

// Convert grayscale, RGB or RGBA input to owned grayscale; stride 0 means packed.
// Caller supplies (height-1)*stride + width*channels readable bytes, fitting ptrdiff_t.
// The raw pointer's allocation size cannot be checked here.
GrayImage toGrayscale(const std::uint8_t* data,
                      int width, int height, int channels,
                      std::size_t row_stride_bytes = 0);

// S1: pyramid -> FAST-9 -> 3x3 NMS -> Harris ranking -> per-level quotas.
// Validates the image/config; returns owned buffers and original-image coordinates.
// Empty keypoints are valid. Detection permits odd patch sizes; S2 requires 31.
DetectionResult detect(const GrayImage& grayscale,
                       const Config& config = Config{});

// S2: intensity-centroid orientation, then steered BRIEF on blurred pyramid levels.
// Returns aligned keypoints/descriptors without modifying detection; empty points are valid.
// Only config.patch_size is used (must be 31). Requires a nonempty pyramid and valid
// octave, scale, coordinates and patch bounds; malformed inputs throw.
DescriptionResult describe(const DetectionResult& detection,
                           const Config& config = Config{});

// Combined S1/S2 with the same Config; checks descriptor compatibility first.
// Call detect() and describe() separately for per-stage timing.
DescriptionResult extract(const GrayImage& grayscale,
                          const Config& config = Config{});

}
