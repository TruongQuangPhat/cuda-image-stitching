// Stage 3 (Feature Matching) and Stage 4 (Homography Estimation): original
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
//   - Hartley, R. & Zisserman, A. (2004). Multiple View Geometry in Computer Vision
//     (2nd ed.). Cambridge University Press. [Chapters 4, 11]
//   - Fischler, M.A. & Bolles, R.C. (1981). Random Sample Consensus.
//     Communications of the ACM, 24(6), 381-395.
//   - Golub, G.H. & Van Loan, C.F. (2013). Matrix Computations (4th ed.),
//     Johns Hopkins University Press. [Section 8.4 - Jacobi Methods]

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

// ============================================================
// S4 – Homography Estimation (Hartley Normalized DLT + RANSAC)
// ============================================================

namespace stitching::homography {

// Configuration for RANSAC-based Homography estimation.
// All fields are validated at the start of estimate(); invalid config causes
// estimate() to return valid=false with failure_reason=InvalidConfig.
struct HomographyConfig {
    // Number of RANSAC iterations. Must be > 0.
    int ransac_iterations = 2000;

    // Inlier reprojection threshold in pixels (original image coordinate space).
    // Must be finite and > 0.
    double reprojection_threshold = 3.0;

    // Minimum inliers required to accept a model. Must be >= 4.
    // The DLT requires at least 4 non-degenerate correspondences.
    int min_inliers = 4;

    // Minimum inlier ratio required to accept a model: num_inliers / matches.size().
    // Range: [0.0, 1.0]. Default 0.0 disables the ratio gate (accepts any model
    // with num_inliers >= min_inliers). For production stitching, setting this to
    // e.g. 0.15 - 0.20 rejects accidental 4-point consensus in noisy match sets.
    double min_inlier_ratio = 0.0;

    // RNG seed for deterministic RANSAC. Same seed -> same result.
    std::uint32_t random_seed = 0xC5C14120u;
};

// Failure diagnostics.
enum class FailureReason {
    None,                // Success.
    InvalidConfig,       // HomographyConfig failed validation.
    InvalidInput,        // Input matches or keypoints are invalid/out-of-range.
    TooFewMatches,       // Fewer than 4 input matches.
    AllHypothesesFailed, // No valid DLT solution found in RANSAC.
    TooFewInliers,       // Best model has fewer than min_inliers or inlier_ratio < min_inlier_ratio.
    RefinementFailed,    // DLT refinement failed and fallback also failed.
};

std::string failureReasonToString(FailureReason r);

// Estimated H_B_to_A (maps points in image B to image A):
//   x_A ~ H_B_to_A * x_B
//
// H is a 3x3 matrix stored row-major: H[row*3 + col].
// When valid == false, H is all-zeros and must NOT be used.
struct HomographyResult {
    // The estimated Homography.
    std::array<double, 9> H{};

    bool           valid          = false;
    FailureReason  failure_reason = FailureReason::None;

    // Non-empty: either an error description (valid=false) or an info message
    // (valid=true, e.g. when refinement fell back to RANSAC best H).
    std::string    error_message;

    // Inlier mask: one entry per input match (inlier_mask.size()==matches.size()).
    // Computed with the RETURNED H (always consistent with H above).
    std::vector<bool> inlier_mask;
    int               num_inliers  = 0;   // count(inlier_mask == true)
    double            inlier_ratio = 0.0; // num_inliers / matches.size()

    // Reprojection error statistics over inliers only, using the returned H.
    double mean_reprojection_error = 0.0;
    double max_reprojection_error  = 0.0;

    // True if inlier-refinement DLT succeeded AND the refined model was accepted
    // as the final model (improving inliers or reprojection error).
    // False if refinement failed or was rejected in favor of the RANSAC best hypothesis.
    bool refinement_succeeded = false;
};

// Estimate H_B_to_A from matched correspondences.
//
//   matches:        output of stitching::matching::match()
//   desc_a, desc_b: same DescriptionResult objects passed to match()
//   config:         algorithm parameters (validated on entry)
//
// All errors are communicated through HomographyResult.valid / .failure_reason.
// Does NOT throw exceptions.
HomographyResult estimate(
    const std::vector<matching::Match>& matches,
    const orb::DescriptionResult&       desc_a,
    const orb::DescriptionResult&       desc_b,
    const HomographyConfig&             config = HomographyConfig{});

// Project a 2-D point through H (performs homogeneous division).
// Invariant to projective scaling: project(c * H, x, y) == project(H, x, y) for any c != 0.
// Returns {x, y} in Euclidean space.
// Throws std::domain_error if:
//   - H is all-zeros or degenerate,
//   - H contains non-finite values,
//   - input coordinates are non-finite,
//   - homogeneous denominator is near zero relative to coordinate scale,
//   - projected coordinates are non-finite.
struct Point2d { double x, y; };
Point2d project(const std::array<double, 9>& H, double x, double y);

// Euclidean reprojection error: ||project(H, p_b) - p_a||.
// Consistent with project(): scale-invariant to c * H for any c != 0.
// Returns std::numeric_limits<double>::max() on degeneracy or non-finite result.
// Never throws.
double reprojectionError(const std::array<double, 9>& H,
                         double xb, double yb,
                         double xa, double ya);

}  // namespace stitching::homography
