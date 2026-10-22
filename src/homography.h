// Authors: TRƯƠNG QUANG PHÁT (23120318); CHÂU HUỲNH PHÚC (23120329, S4).
// Homography estimation and projection interfaces.
// References: Hartley and Zisserman, Multiple View Geometry (2004);
// Fischler and Bolles, Random Sample Consensus (1981);
// Golub and Van Loan, Matrix Computations (2013), Jacobi methods.

#pragma once

#include "matching.h"
#include "orb.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

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
