#pragma once

// Stage 1 (ORB keypoint detection): original educational implementation.
// Algorithm references: Rublee et al., ORB, ICCV 2011; Rosten et al., FAST.

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
    float angle_rad = 0.0f;  // Placeholder for Stage 2 orientation.
    int octave = 0;         // Pyramid level at which FAST detected the point.
};

struct Config {
    int max_features = 2000; 
    int n_levels = 8;
    float scale_factor = 1.2f;
    int fast_threshold = 20; 
    int edge_threshold = 31;
    int patch_size = 31;
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

// Convert RGB, RGBA or 1-channel interleaved 8-bit input into owned grayscale.
GrayImage toGrayscale(const std::uint8_t* data,
                      int width, int height, int channels,
                      std::size_t row_stride_bytes = 0);


DetectionResult detect(const GrayImage& grayscale,
                       const Config& config = Config{});

}
