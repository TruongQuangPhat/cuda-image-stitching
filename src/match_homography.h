// Stage 3 (Feature Matching) and Stage 4 (Homography Estimation): original
// educational implementation for CSC14120 – Parallel Programming, V0 sequential
// baseline.
//
// Authors: Truong Quang Phat (23120318)
// Algorithm references:
//   - Lowe, D.G. (2004). Distinctive image features from scale-invariant keypoints.
//     International Journal of Computer Vision, 60(2), 91–110.
//   - Hartley, R. & Zisserman, A. (2004). Multiple View Geometry in Computer Vision
//     (2nd ed.). Cambridge University Press. [Chapters 4, 11]
//   - Fischler, M.A. & Bolles, R.C. (1981). Random Sample Consensus.
//     Communications of the ACM, 24(6), 381–395.

#pragma once

#include "orb.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// ============================================================
// S3 – Feature Matching (Brute-Force Hamming + 2-NN Ratio Test)
// ============================================================

namespace stitching::matching {

struct MatchingConfig {
    float ratio_threshold = 0.8f;  // Lowe's ratio test threshold d1 < t * d2.
    bool  cross_check     = false; // Not default; disabled per team decision.
    // Cross-check: accept match only if A→B and B→A agree (not used in V0).
};

// One accepted correspondence between ORB keypoint/descriptor in image A and B.
// Both index_a and index_b are indices into the DescriptionResult::keypoints
// (and DescriptionResult::descriptors) vectors passed to match().
struct Match {
    std::size_t index_a    = 0;   // Index into image-A DescriptionResult.
    std::size_t index_b    = 0;   // Index into image-B DescriptionResult.
    int         distance   = 0;   // Hamming distance of the winning pair (d1).
    int         distance2  = 0;   // Hamming distance of the runner-up (d2 ≥ d1).
};

struct MatchResult {
    std::vector<Match> matches;   // Only matches that passed ratio test.
    bool   valid = true;          // False when inputs are fundamentally invalid.
    std::string error_message;    // Non-empty when valid == false.
};

// Compute the Hamming distance between two 256-bit descriptors.
// Result is in [0, 256].
int hammingDistance(const orb::Descriptor& a, const orb::Descriptor& b);

// Run brute-force 2-NN matching followed by Lowe's ratio test.
// Descriptors and keypoints in each DescriptionResult must be aligned:
//   desc_a.keypoints.size() == desc_a.descriptors.size() (and same for B).
// Throws std::invalid_argument on mismatched keypoint/descriptor counts.
MatchResult match(const orb::DescriptionResult& desc_a,
                  const orb::DescriptionResult& desc_b,
                  const MatchingConfig& config = MatchingConfig{});

}  // namespace stitching::matching

// ============================================================
// S4 – Homography Estimation (Hartley Normalized DLT + RANSAC)
// ============================================================

namespace stitching::homography {

struct HomographyConfig {
    int    ransac_iterations    = 2000;
    double reprojection_threshold = 3.0; // Pixels in the input image coordinate system.
    int    min_inliers          = 4;     // Reject model with fewer inliers.
    std::uint32_t random_seed   = 0xC5C14120u; // Reproducible by default.
};

// Failure reasons, exposed for diagnostics.
enum class FailureReason {
    None,                // No failure.
    TooFewMatches,       // Fewer than 4 input matches.
    AllHypothesesFailed, // No valid DLT solution in any RANSAC iteration.
    RefinementFailed,    // Final DLT refinement on inliers failed.
    TooFewInliers,       // Best model had fewer than min_inliers inliers.
};

std::string failureReasonToString(FailureReason r);

// The estimated Homography H_B_to_A such that:
//   x_A ~ H_B_to_A * x_B
// Stored row-major: H[row*3 + col].
// When valid == false, H contains zeros and must not be used.
struct HomographyResult {
    std::array<double, 9> H{};           // 3×3 row-major Homography.
    bool           valid          = false;
    FailureReason  failure_reason = FailureReason::None;
    std::string    error_message;

    std::vector<bool> inlier_mask;        // One entry per input match.
    int               num_inliers   = 0;
    double            inlier_ratio  = 0.0;
    double            mean_reprojection_error     = 0.0; // Over inliers only.
    double            max_reprojection_error      = 0.0; // Over inliers only.
};

// Estimate H_B_to_A from the matched correspondences.
// matches:      output of stitching::matching::match().
// desc_a, desc_b: same DescriptionResult objects passed to match().
//   Keypoint coordinates are used; they are in original-image pixel space.
// config:       algorithm parameters.
HomographyResult estimate(
    const std::vector<matching::Match>& matches,
    const orb::DescriptionResult&       desc_a,
    const orb::DescriptionResult&       desc_b,
    const HomographyConfig&             config = HomographyConfig{});

// Convenience: project a 2-D point using H (homogeneous coordinates).
// Returns {x, y} in Euclidean space.
// Throws std::domain_error if the homogeneous denominator is near zero.
struct Point2d { double x, y; };
Point2d project(const std::array<double, 9>& H, double x, double y);

// Euclidean reprojection error: ||project(H, p_b) - p_a||.
// Returns a large sentinel value if projection is degenerate.
double reprojectionError(const std::array<double, 9>& H,
                         double xb, double yb,
                         double xa, double ya);

}  // namespace stitching::homography
