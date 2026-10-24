// Authors: TRƯƠNG QUANG PHÁT (23120318); CHÂU HUỲNH PHÚC (23120329, S3).
// Hamming matching interfaces.
// Reference: Lowe (2004), Distinctive image features from scale-invariant keypoints.

#pragma once

#include "orb.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace stitching::matching {

// Configuration for brute-force 2-NN matching with Lowe's ratio test.
struct MatchingConfig {
    // Accept d1 < ratio_threshold * d2; require 0 < ratio_threshold < 1.
    float ratio_threshold = 0.8f;

    // After the ratio test, keep A[i]->B[j] only if B[j]'s nearest neighbor is A[i].
    bool cross_check = false;
};

// One accepted correspondence between ORB keypoints in image A and image B.
// index_a and index_b are valid indices into the DescriptionResult::keypoints
// (and ::descriptors) vectors passed to match().
struct Match {
    std::size_t index_a   = 0;
    std::size_t index_b   = 0;
    int         distance  = 0;  // Hamming distance of the winning pair (d1).
    int         distance2 = 0;  // Hamming distance of the runner-up (d2 >= d1).
};

struct MatchResult {
    std::vector<Match> matches;   // Matches that passed ratio test (and cross-check).
    bool   valid = true;          // False when inputs or config are invalid.
    // valid=true with no matches is a normal result; it does not certify that
    // there are enough correspondences for Homography estimation.
    std::string error_message;    // Non-empty when valid == false.
};

// Hamming distance between 256-bit descriptors, in [0, 256].
int hammingDistance(const orb::Descriptor& a, const orb::Descriptor& b);

// Run brute-force 2-NN matching followed by Lowe's ratio test.
//
// Runtime preconditions:
//   - desc_a.keypoints.size() == desc_a.descriptors.size()
//   - desc_b.keypoints.size() == desc_b.descriptors.size()
//   - 0 < config.ratio_threshold < 1
//
// Invalid config or misaligned inputs return valid=false with error_message.
// Allocation exceptions may propagate.
MatchResult match(const orb::DescriptionResult& desc_a,
                  const orb::DescriptionResult& desc_b,
                  const MatchingConfig& config = MatchingConfig{});

}  // namespace stitching::matching
