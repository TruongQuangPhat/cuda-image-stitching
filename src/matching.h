// Feature Matching (Brute-Force Hamming + 2-NN Ratio Test + Cross-Check):
// educational implementation for CSC14120 – Parallel Programming, V0 sequential
// baseline.
//
// Authors: Truong Quang Phat (23120318)
// AI-assisted development: implemented with Antigravity (Google DeepMind), reviewed
// and corrected by the author per course academic integrity requirements.
//
// Algorithm references:
//   - Lowe, D.G. (2004). Distinctive image features from scale-invariant keypoints.
//     International Journal of Computer Vision, 60(2), 91-110.

#pragma once

#include "orb.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace stitching::matching {

// Configuration for brute-force 2-NN matching with Lowe's ratio test.
struct MatchingConfig {
    // Lowe's ratio test threshold: accept match only when d1 < ratio_threshold * d2.
    // Must satisfy 0 < ratio_threshold < 1.
    // Values outside this range cause match() to return valid=false.
    float ratio_threshold = 0.8f;

    // Cross-check (mutual nearest neighbor filter).
    // When true, a forward match A[i]->B[j] is kept only if B[j]'s nearest
    // neighbor in A is also A[i]. Applied AFTER the ratio test.
    // Default false (V0 baseline).
    bool cross_check = false;
};

// One accepted correspondence between ORB keypoints in image A and image B.
// index_a and index_b are valid indices into the DescriptionResult::keypoints
// (and ::descriptors) vectors passed to match().
struct Match {
    std::size_t index_a   = 0;  // Index into image-A DescriptionResult.
    std::size_t index_b   = 0;  // Index into image-B DescriptionResult.
    int         distance  = 0;  // Hamming distance of the winning pair (d1).
    int         distance2 = 0;  // Hamming distance of the runner-up (d2 >= d1).
};

struct MatchResult {
    std::vector<Match> matches;   // Matches that passed ratio test (and cross-check).
    bool   valid = true;          // False when inputs or config are invalid.
    std::string error_message;    // Non-empty when valid == false.
};

// Compute the Hamming distance between two 256-bit ORB descriptors.
// Result is always in [0, 256].
int hammingDistance(const orb::Descriptor& a, const orb::Descriptor& b);

// Run brute-force 2-NN matching followed by Lowe's ratio test.
//
// Preconditions (both checked at runtime):
//   - desc_a.keypoints.size() == desc_a.descriptors.size()
//   - desc_b.keypoints.size() == desc_b.descriptors.size()
//   - 0 < config.ratio_threshold < 1
//
// On invalid config or mismatched keypoint/descriptor sizes, returns
// MatchResult{.valid=false} with a descriptive error_message.
// Does NOT throw exceptions.
MatchResult match(const orb::DescriptionResult& desc_a,
                  const orb::DescriptionResult& desc_b,
                  const MatchingConfig& config = MatchingConfig{});

}  // namespace stitching::matching
