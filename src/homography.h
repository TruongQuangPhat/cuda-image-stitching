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

// Row-major H_B_to_A: x_A ~ H * x_B. Use H only when valid is true.
// Success returns consistent H, inliers and errors. Failure leaves H zero;
// mask/count/ratio may still describe a rejected hypothesis.
// S5 separately checks invertibility, the projective horizon and canvas bounds.
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

// Estimate B->A from the same descriptor sets used by match().
// Check valid/failure_reason for consensus failure; allocation may throw.
// Accepted consensus does not guarantee image-wide quality or a bounded canvas.
HomographyResult estimate(
    const std::vector<matching::Match>& matches,
    const orb::DescriptionResult&       desc_a,
    const orb::DescriptionResult&       desc_b,
    const HomographyConfig&             config = HomographyConfig{});

// Project a point with homogeneous division; this does not check H invertibility.
// Scale-invariant within floating-point precision for finite H with max(abs(H)) >= 1e-300.
// Throws domain_error for non-finite inputs/output, a smaller H, or a denominator
// near zero relative to coordinate scale.
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
