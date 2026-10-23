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

    // RNG seed for deterministic RANSAC with the same inputs and implementation.
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
    RefinementFailed,    // Reserved for compatibility; current estimate() retains accepted RANSAC H.
};

std::string failureReasonToString(FailureReason r);

// Estimated H_B_to_A (maps points in image B to image A):
//   x_A ~ H_B_to_A * x_B
//
// H is a 3x3 matrix stored row-major: H[row*3 + col].
// When valid == false, H is all-zeros and must NOT be used.
// Inspect valid first: a default-constructed result is not an estimate even
// though failure_reason starts at None. estimate() sets None only on success.
// For a result returned by estimate():
//   success: valid=true, failure_reason=None, H/mask/count/ratio/errors agree.
//   failure: valid=false, failure_reason!=None, H remains zero. Only the
//            diagnostic mask/count/ratio may describe a rejected hypothesis.
// Consumers must gate on valid, never on mask population or message emptiness.
// Stage 5 additionally checks invertibility, the projective horizon and canvas.
struct HomographyResult {
    // The estimated Homography.
    std::array<double, 9> H{};

    bool           valid          = false;
    FailureReason  failure_reason = FailureReason::None;

    // May be non-empty: either an error description (valid=false) or an info message
    // (valid=true, e.g. when refinement fell back to RANSAC best H).
    // Kept as one field for compatibility; an '(info)' message is not a failure.
    std::string    error_message;

    // Inlier mask: one entry per input match (inlier_mask.size()==matches.size()).
    // On success, computed with the RETURNED H. On TooFewInliers failure, this
    // and the count/ratio describe the rejected best hypothesis; they do NOT
    // validate the zero H. Other estimate() failures return an all-false mask.
    std::vector<bool> inlier_mask;
    int               num_inliers  = 0;   // count(inlier_mask == true)
    double            inlier_ratio = 0.0; // num_inliers / matches.size()

    // On success, reprojection statistics over returned-H inliers only.
    // On failure these remain zero (unavailable, not a zero-error estimate).
    double mean_reprojection_error = 0.0;
    double max_reprojection_error  = 0.0;

    // True if inlier-refinement DLT succeeded AND the refined model was accepted
    // as the final model (more inliers, or same count with error within tolerance).
    // False if refinement failed or was rejected in favor of the RANSAC best hypothesis.
    // A false value does not invalidate an accepted fallback H. Refinement is
    // attempted only after RANSAC passes confidence; failure retains that H.
    bool refinement_succeeded = false;
};

// Estimate H_B_to_A from matched correspondences.
//
//   matches:        output of stitching::matching::match()
//   desc_a, desc_b: same DescriptionResult objects passed to match()
//   config:         algorithm parameters (validated on entry)
//
// Logical failures are communicated through valid / failure_reason. Allocation
// exceptions may propagate. valid means the configured consensus policy passed;
// it does not certify image-wide quality, coverage or a bounded warp canvas.
HomographyResult estimate(
    const std::vector<matching::Match>& matches,
    const orb::DescriptionResult&       desc_a,
    const orb::DescriptionResult&       desc_b,
    const HomographyConfig&             config = HomographyConfig{});

// Project a 2-D point through H (performs homogeneous division).
// Returns Euclidean coordinates at this point; does NOT check matrix
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
