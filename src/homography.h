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

// estimate() validates all fields; invalid config returns valid=false / InvalidConfig.
struct HomographyConfig {
    // Number of RANSAC iterations. Must be > 0.
    int ransac_iterations = 2000;

    // Inlier reprojection threshold in pixels (original image coordinate space).
    // Must be finite and > 0.
    double reprojection_threshold = 3.0;

    // At least 4 inliers; DLT requires four non-degenerate correspondences.
    int min_inliers = 4;

    // Required num_inliers / matches.size(), in [0, 1]. Zero disables the ratio
    // gate; min_inliers still applies and may admit a four-point consensus.
    double min_inlier_ratio = 0.0;

    // RNG seed for deterministic RANSAC with the same inputs and implementation.
    std::uint32_t random_seed = 0xC5C14120u;
};

enum class FailureReason {
    None,                // Success.
    InvalidConfig,       // HomographyConfig failed validation.
    InvalidInput,        // Input matches or keypoints are invalid/out-of-range.
    TooFewMatches,       // Fewer than 4 input matches.
    AllHypothesesFailed, // No valid DLT solution found in RANSAC.
    TooFewInliers,       // Best model has fewer than min_inliers or inlier_ratio < min_inlier_ratio.
    RefinementFailed,    // Reserved for compatibility; current estimate() retains accepted RANSAC H.
};

std::string failureReasonToString(FailureReason r);

// Row-major H_B_to_A: x_A ~ H_B_to_A * x_B, indexed by row*3+col.
// estimate() success: valid=true, failure_reason=None, H/mask/count/ratio/errors agree.
// estimate() failure: valid=false, failure_reason!=None, H stays zero; mask/count/ratio
// may describe a rejected hypothesis. Never use H when valid=false.
// A default result is also invalid, although its failure_reason starts at None.
// Gate on valid, not mask population or message emptiness. Stage 5 additionally
// checks invertibility, the projective horizon and canvas bounds.
struct HomographyResult {
    std::array<double, 9> H{};

    bool           valid          = false;
    FailureReason  failure_reason = FailureReason::None;

    // Errors when invalid, or optional '(info)' diagnostics on success (e.g. fallback).
    // The shared field is retained for compatibility; nonempty does not imply failure.
    std::string    error_message;

    // One entry per input match. On success, inliers use the returned H.
    // TooFewInliers preserves mask/count/ratio of the rejected best hypothesis;
    // other estimate() failures return an all-false mask.
    std::vector<bool> inlier_mask;
    int               num_inliers  = 0;   // count(inlier_mask == true)
    double            inlier_ratio = 0.0; // num_inliers / matches.size()

    // On success, reprojection statistics over returned-H inliers only.
    // On failure these remain zero (unavailable, not a zero-error estimate).
    double mean_reprojection_error = 0.0;
    double max_reprojection_error  = 0.0;

    // Refinement is attempted after RANSAC passes confidence. True if accepted:
    // more inliers, or equal count within error tolerance. Failed/rejected
    // refinement retains the accepted RANSAC H without invalidating the result.
    bool refinement_succeeded = false;
};

// Estimate H_B_to_A using matches and the same DescriptionResult objects passed to match().
// Logical failures are communicated through valid / failure_reason. Allocation
// exceptions may propagate. valid means the configured consensus policy passed;
// it does not certify image-wide quality, coverage or a bounded warp canvas.
HomographyResult estimate(
    const std::vector<matching::Match>& matches,
    const orb::DescriptionResult&       desc_a,
    const orb::DescriptionResult&       desc_b,
    const HomographyConfig&             config = HomographyConfig{});

// Project a 2-D point through H (performs homogeneous division).
// Returns Euclidean coordinates at this point; does not check matrix
// invertibility. Even a singular matrix may project an individual point.
// Scaling H preserves the result within floating-point precision when both
// representations remain finite, max(abs(H)) >= 1e-300, and projection is valid.
// Throws std::domain_error if:
//   - max(abs(H)) < 1e-300 (including the zero matrix),
//   - H contains non-finite values,
//   - input coordinates are non-finite,
//   - homogeneous denominator is near zero relative to coordinate scale,
//   - projected coordinates are non-finite.
struct Point2d { double x, y; };
Point2d project(const std::array<double, 9>& H, double x, double y);

// Euclidean reprojection error: ||project(H, p_b) - p_a||.
// Uses the same projection validity/scaling limits as project().
// Returns std::numeric_limits<double>::max() on invalid projection/non-finite input or error.
// Never throws.
double reprojectionError(const std::array<double, 9>& H,
                         double xb, double yb,
                         double xa, double ya);

}  // namespace stitching::homography
