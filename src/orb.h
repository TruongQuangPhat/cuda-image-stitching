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

// Convert RGB, RGBA or 1-channel interleaved 8-bit input into owned grayscale.
// Channels: 1, 3, or 4. The 3/4-channel layout is RGB/RGBA, not BGR/BGRA.
// Pass row_stride_bytes = 0 for a tightly packed source buffer.
// Caller owns data and must provide (height-1)*effective_stride + width*channels
// readable bytes. The full span must fit ptrdiff_t; unused final-row padding is
// excluded. Actual allocation capacity cannot be validated from a raw pointer.
GrayImage toGrayscale(const std::uint8_t* data,
                      int width, int height, int channels,
                      std::size_t row_stride_bytes = 0);

// Stage 1 only: pyramid -> FAST-9 -> 3x3 NMS -> Harris ranking -> quotas.
// Validates the image and every Config field. Detection-only use permits odd
// patch sizes other than 31; use 31 when chaining with this rBRIEF descriptor.
// Keypoint coordinates are in original-image space; output owns its buffers.
// An empty keypoint list is valid (including max_features == 0).
DetectionResult detect(const GrayImage& grayscale,
                       const Config& config = Config{});

// Stage 2: calculate intensity-centroid orientations on original pyramid
// levels, then build 256-bit steered BRIEF descriptors on blurred levels.
// Keypoints and descriptors are aligned by index. Does not modify detection.
// Only config.patch_size is used and must equal 31; detection-only fields are
// ignored. Requires a nonempty pyramid and valid octave, scale, coordinates and
// patch bounds for every keypoint. Empty keypoints yield aligned empty vectors.
// Malformed inputs/configurations throw; detection is left unchanged.
DescriptionResult describe(const DetectionResult& detection,
                           const Config& config = Config{});

// Call detect() and describe() separately when profiling each stage.
// Uses the same Config for both stages. Checks description compatibility before
// detection; then applies detect()'s image/config validation.
DescriptionResult extract(const GrayImage& grayscale,
                          const Config& config = Config{});

} 
