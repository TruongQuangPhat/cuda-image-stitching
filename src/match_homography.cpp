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

#include "match_homography.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>

// ============================================================
//  Internal utilities
// ============================================================

namespace {

// ---- Hamming popcount on individual uint8 blocks --------------------------
// Uses the compiler builtin when available (GCC/Clang with -mpopcnt or
// the software fallback path). No SIMD, no manual bit tricks.
inline int popcount8(std::uint8_t v) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_popcount(static_cast<unsigned>(v));
#else
    // Kernighan's bit-counting algorithm – portable fallback.
    int count = 0;
    for (; v; v &= static_cast<std::uint8_t>(v - 1u)) ++count;
    return count;
#endif
}

// ---- 3×3 matrix helpers (row-major, indices [row*3+col]) ------------------
using Mat3 = std::array<double, 9>;

inline double det3(const Mat3& m) {
    return m[0] * (m[4] * m[8] - m[5] * m[7])
         - m[1] * (m[3] * m[8] - m[5] * m[6])
         + m[2] * (m[3] * m[7] - m[4] * m[6]);
}

// Returns false if the matrix is not invertible (det near zero).
bool invert3(const Mat3& m, Mat3& inv) {
    const double d = det3(m);
    constexpr double kEps = 1e-12;
    if (std::abs(d) < kEps) return false;
    const double id = 1.0 / d;
    inv[0] =  (m[4] * m[8] - m[5] * m[7]) * id;
    inv[1] = -(m[1] * m[8] - m[2] * m[7]) * id;
    inv[2] =  (m[1] * m[5] - m[2] * m[4]) * id;
    inv[3] = -(m[3] * m[8] - m[5] * m[6]) * id;
    inv[4] =  (m[0] * m[8] - m[2] * m[6]) * id;
    inv[5] = -(m[0] * m[5] - m[2] * m[3]) * id;
    inv[6] =  (m[3] * m[7] - m[4] * m[6]) * id;
    inv[7] = -(m[0] * m[7] - m[1] * m[6]) * id;
    inv[8] =  (m[0] * m[4] - m[1] * m[3]) * id;
    return true;
}

Mat3 matmul3(const Mat3& A, const Mat3& B) {
    Mat3 C{};
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            for (int k = 0; k < 3; ++k)
                C[r * 3 + c] += A[r * 3 + k] * B[k * 3 + c];
    return C;
}

// ---- Hartley normalization ------------------------------------------------
// Given N 2-D points, compute the similarity transform T such that
//   T * [x_i; y_i; 1]  has centroid at (0,0) and mean distance sqrt(2).
// Returns false if points are degenerate (fewer than 1, or all coincident).
struct NormResult {
    Mat3 T;
    std::vector<double> xs;  // Normalized x coordinates.
    std::vector<double> ys;  // Normalized y coordinates.
};

bool hartleyNormalize(const std::vector<double>& x,
                      const std::vector<double>& y,
                      NormResult& out) {
    const std::size_t n = x.size();
    if (n == 0) return false;

    // Centroid.
    double cx = 0.0, cy = 0.0;
    for (std::size_t i = 0; i < n; ++i) { cx += x[i]; cy += y[i]; }
    cx /= static_cast<double>(n);
    cy /= static_cast<double>(n);

    // Mean distance from centroid.
    double mean_dist = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double dx = x[i] - cx;
        const double dy = y[i] - cy;
        mean_dist += std::sqrt(dx * dx + dy * dy);
    }
    mean_dist /= static_cast<double>(n);

    // Avoid division by zero when all points are coincident.
    constexpr double kEps = 1e-9;
    if (mean_dist < kEps) return false;

    const double s = std::sqrt(2.0) / mean_dist;

    // T = [[s, 0, -s*cx], [0, s, -s*cy], [0, 0, 1]]
    out.T = {};
    out.T[0] = s;   out.T[1] = 0.0; out.T[2] = -s * cx;
    out.T[3] = 0.0; out.T[4] = s;   out.T[5] = -s * cy;
    out.T[6] = 0.0; out.T[7] = 0.0; out.T[8] =  1.0;

    out.xs.resize(n);
    out.ys.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        out.xs[i] = s * (x[i] - cx);
        out.ys[i] = s * (y[i] - cy);
    }
    return true;
}

// ---- Minimal SVD for a 9×n or n×9 thin problem via Jacobi ---------------
// We need the null-space of a (2n)×9 system. We implement a compact
// Golub-Reinsch SVD for exactly the 9×9 normal-equation approach.
//
// Approach: build A^T A (9×9) and find the eigenvector of the smallest
// eigenvalue via Jacobi iterations. This is sufficient for n ≥ 4 and is
// numerically equivalent to the null-space of A for well-conditioned data.

// 9×9 symmetric matrix as a flat array [i*9+j].
using SymMat9 = std::array<double, 81>;

// Accumulate A^T * A from a (2n × 9) matrix stored row by row.
// rows is a vector of rows, each of length 9.
SymMat9 buildATA(const std::vector<std::array<double, 9>>& rows) {
    SymMat9 M{};
    for (const auto& row : rows)
        for (int i = 0; i < 9; ++i)
            for (int j = i; j < 9; ++j) {
                M[i * 9 + j] += row[i] * row[j];
                M[j * 9 + i] = M[i * 9 + j];
            }
    return M;
}

// Jacobi eigendecomposition of a real symmetric 9×9 matrix.
// Returns the eigenvectors as columns of V (9×9 row-major) and eigenvalues.
// Convergence criterion: max off-diagonal < 1e-14, up to 1000 sweeps.
void jacobiSym9(SymMat9 A, std::array<double, 9>& eigenvalues,
                std::array<double, 81>& V) {
    // Initialize V to identity.
    V.fill(0.0);
    for (int i = 0; i < 9; ++i) V[i * 9 + i] = 1.0;

    constexpr int kMaxSweeps = 1000;
    constexpr double kTol = 1e-14;

    for (int sweep = 0; sweep < kMaxSweeps; ++sweep) {
        // Find off-diagonal element of maximum absolute value.
        double max_off = 0.0;
        for (int p = 0; p < 9; ++p)
            for (int q = p + 1; q < 9; ++q)
                max_off = std::max(max_off, std::abs(A[p * 9 + q]));
        if (max_off < kTol) break;

        // One sweep: rotate all off-diagonal pairs.
        for (int p = 0; p < 8; ++p) {
            for (int q = p + 1; q < 9; ++q) {
                const double apq = A[p * 9 + q];
                if (std::abs(apq) < kTol * 0.001) continue;
                const double tau = (A[q * 9 + q] - A[p * 9 + p]) / (2.0 * apq);
                const double t = (tau >= 0.0)
                    ? 1.0 / (tau + std::sqrt(1.0 + tau * tau))
                    : 1.0 / (tau - std::sqrt(1.0 + tau * tau));
                const double c = 1.0 / std::sqrt(1.0 + t * t);
                const double s = t * c;
                // Update A.
                A[p * 9 + p] -= t * apq;
                A[q * 9 + q] += t * apq;
                A[p * 9 + q] = 0.0;
                A[q * 9 + p] = 0.0;
                for (int r = 0; r < 9; ++r) {
                    if (r == p || r == q) continue;
                    const double arp = A[r * 9 + p];
                    const double arq = A[r * 9 + q];
                    A[r * 9 + p] = c * arp - s * arq;
                    A[p * 9 + r] = A[r * 9 + p];
                    A[r * 9 + q] = s * arp + c * arq;
                    A[q * 9 + r] = A[r * 9 + q];
                }
                // Update V (accumulate rotation).
                for (int r = 0; r < 9; ++r) {
                    const double vrp = V[r * 9 + p];
                    const double vrq = V[r * 9 + q];
                    V[r * 9 + p] = c * vrp - s * vrq;
                    V[r * 9 + q] = s * vrp + c * vrq;
                }
            }
        }
    }
    for (int i = 0; i < 9; ++i) eigenvalues[i] = A[i * 9 + i];
}

// ---- Normalized DLT for Homography estimation ----------------------------
// Given N≥4 correspondences (xb[i],yb[i]) → (xa[i],ya[i]),
// estimates H_B_to_A such that xa_i ~ H * xb_i.
// Returns true on success; H is written in row-major order.
// The function normalizes coordinates (Hartley) before DLT to improve
// numerical conditioning.
bool normalizedDLT(const std::vector<double>& xa,
                   const std::vector<double>& ya,
                   const std::vector<double>& xb,
                   const std::vector<double>& yb,
                   Mat3& H) {
    const std::size_t n = xa.size();
    if (n < 4 || xb.size() != n || ya.size() != n || yb.size() != n)
        return false;

    // Check all inputs are finite.
    for (std::size_t i = 0; i < n; ++i) {
        if (!std::isfinite(xa[i]) || !std::isfinite(ya[i]) ||
            !std::isfinite(xb[i]) || !std::isfinite(yb[i]))
            return false;
    }

    // Hartley normalization.
    NormResult Na, Nb;
    if (!hartleyNormalize(xa, ya, Na)) return false;
    if (!hartleyNormalize(xb, yb, Nb)) return false;

    // Build 2n×9 matrix A.
    // Each correspondence contributes two rows of the form from Hartley p.91.
    // Row for x constraint: [0 0 0 -xb -yb -1 ya*xb ya*yb ya]
    // Row for y constraint: [xb yb 1 0 0 0 -xa*xb -xa*yb -xa]
    std::vector<std::array<double, 9>> rows;
    rows.reserve(2 * n);
    for (std::size_t i = 0; i < n; ++i) {
        const double Xa = Na.xs[i], Ya = Na.ys[i];
        const double Xb = Nb.xs[i], Yb = Nb.ys[i];
        rows.push_back({0.0, 0.0, 0.0, -Xb, -Yb, -1.0, Ya * Xb, Ya * Yb, Ya});
        rows.push_back({Xb, Yb, 1.0, 0.0, 0.0, 0.0, -Xa * Xb, -Xa * Yb, -Xa});
    }

    // Build A^T * A and find eigenvector for smallest eigenvalue.
    const SymMat9 ATA = buildATA(rows);
    std::array<double, 9> eigenvalues{};
    std::array<double, 81> V{};
    jacobiSym9(ATA, eigenvalues, V);

    // Find the index of the smallest eigenvalue.
    int min_idx = 0;
    for (int i = 1; i < 9; ++i)
        if (eigenvalues[i] < eigenvalues[min_idx]) min_idx = i;

    // The corresponding eigenvector (column of V).
    Mat3 Hn{};
    for (int i = 0; i < 9; ++i) Hn[i] = V[i * 9 + min_idx];

    // Check Hn is finite.
    for (double v : Hn)
        if (!std::isfinite(v)) return false;

    // Denormalize: H = T_a^{-1} * Hn * T_b
    Mat3 T_a_inv{};
    if (!invert3(Na.T, T_a_inv)) return false;

    H = matmul3(T_a_inv, matmul3(Hn, Nb.T));

    // Normalize so that H[8] == 1.0 when non-zero (scale ambiguity).
    constexpr double kEps = 1e-12;
    if (std::abs(H[8]) > kEps) {
        const double s = 1.0 / H[8];
        for (double& v : H) v *= s;
    } else {
        // Find the largest magnitude coefficient and normalize by it.
        double max_abs = 0.0;
        for (double v : H) max_abs = std::max(max_abs, std::abs(v));
        if (max_abs < kEps) return false;
        for (double& v : H) v /= max_abs;
    }

    // Final sanity: all entries finite.
    for (double v : H)
        if (!std::isfinite(v)) return false;

    return true;
}

// ---- Degeneracy check for a 4-point sample --------------------------------
// Returns true if the four points (xa, ya) or (xb, yb) are nearly collinear
// or contain repeated points, which would make the DLT system singular.
bool isSampleDegenerate(const std::vector<double>& xa,
                        const std::vector<double>& ya,
                        const std::vector<double>& xb,
                        const std::vector<double>& yb,
                        std::size_t i0, std::size_t i1,
                        std::size_t i2, std::size_t i3) {
    // Check for repeated points in image A or B.
    constexpr double kMinDist = 1e-3;
    std::array<std::size_t, 4> idx = {i0, i1, i2, i3};
    for (int p = 0; p < 4; ++p) {
        for (int q = p + 1; q < 4; ++q) {
            const double dxa = xa[idx[p]] - xa[idx[q]];
            const double dya = ya[idx[p]] - ya[idx[q]];
            const double dxb = xb[idx[p]] - xb[idx[q]];
            const double dyb = yb[idx[p]] - yb[idx[q]];
            if (dxa * dxa + dya * dya < kMinDist * kMinDist) return true;
            if (dxb * dxb + dyb * dyb < kMinDist * kMinDist) return true;
        }
    }

    // Check collinearity using the area of the convex hull.
    // For 4 points, the minimal area triangle must be non-degenerate.
    // We check all 4 triangles formed by choosing 3 of the 4 points.
    constexpr double kMinArea = 1e-3;
    auto crossZ = [](double ax, double ay, double bx, double by,
                     double cx, double cy) {
        return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
    };
    for (int skip = 0; skip < 4; ++skip) {
        std::array<std::size_t, 3> tri;
        int k = 0;
        for (int p = 0; p < 4; ++p) if (p != skip) tri[k++] = idx[p];
        // A
        if (std::abs(crossZ(xa[tri[0]], ya[tri[0]],
                            xa[tri[1]], ya[tri[1]],
                            xa[tri[2]], ya[tri[2]])) < kMinArea) return true;
        // B
        if (std::abs(crossZ(xb[tri[0]], yb[tri[0]],
                            xb[tri[1]], yb[tri[1]],
                            xb[tri[2]], yb[tri[2]])) < kMinArea) return true;
    }
    return false;
}

// Project point (x, y) in B through H_B_to_A and return reprojection error.
// Returns a large sentinel on degeneracy.
double computeReprojectionError(const Mat3& H,
                                double xb, double yb,
                                double xa, double ya) {
    const double w  = H[6] * xb + H[7] * yb + H[8];
    constexpr double kEps = 1e-9;
    if (std::abs(w) < kEps) return std::numeric_limits<double>::max();
    const double px = (H[0] * xb + H[1] * yb + H[2]) / w;
    const double py = (H[3] * xb + H[4] * yb + H[5]) / w;
    const double dx = px - xa;
    const double dy = py - ya;
    return std::sqrt(dx * dx + dy * dy);
}

}  // anonymous namespace

// ============================================================
//  S3 Implementation
// ============================================================

namespace stitching::matching {

int hammingDistance(const orb::Descriptor& a, const orb::Descriptor& b) {
    int dist = 0;
    for (std::size_t i = 0; i < 32; ++i)
        dist += popcount8(static_cast<std::uint8_t>(a.bytes[i] ^ b.bytes[i]));
    return dist;
}

MatchResult match(const orb::DescriptionResult& desc_a,
                  const orb::DescriptionResult& desc_b,
                  const MatchingConfig& config) {
    // Validate that keypoints and descriptors are aligned.
    if (desc_a.keypoints.size() != desc_a.descriptors.size()) {
        return {
            {}, false,
            "Image A: keypoints.size() != descriptors.size()"
        };
    }
    if (desc_b.keypoints.size() != desc_b.descriptors.size()) {
        return {
            {}, false,
            "Image B: keypoints.size() != descriptors.size()"
        };
    }

    const std::size_t na = desc_a.descriptors.size();
    const std::size_t nb = desc_b.descriptors.size();

    MatchResult result;

    // Need at least 2 descriptors in B for a meaningful ratio test.
    if (na == 0 || nb < 2) {
        result.valid = true;  // Not an error – just no matches possible.
        return result;
    }

    result.matches.reserve(na);

    for (std::size_t i = 0; i < na; ++i) {
        int d1 = std::numeric_limits<int>::max(); // Smallest distance.
        int d2 = std::numeric_limits<int>::max(); // Second smallest.
        std::size_t j1 = 0;                       // Index of best match in B.

        for (std::size_t j = 0; j < nb; ++j) {
            const int d = hammingDistance(desc_a.descriptors[i],
                                          desc_b.descriptors[j]);
            if (d < d1) {
                d2 = d1; // Old best becomes runner-up.
                d1 = d;
                j1 = j;
            } else if (d < d2) {
                d2 = d;
            }
        }

        // Ratio test: d1 < ratio_threshold * d2.
        // Guard against d2 == 0 (ambiguous match).
        if (d2 == 0) continue;
        if (static_cast<float>(d1) < config.ratio_threshold *
                                     static_cast<float>(d2)) {
            result.matches.push_back({i, j1, d1, d2});
        }
    }

    return result;
}

}  // namespace stitching::matching

// ============================================================
//  S4 Implementation
// ============================================================

namespace stitching::homography {

std::string failureReasonToString(FailureReason r) {
    switch (r) {
        case FailureReason::None:                return "None";
        case FailureReason::TooFewMatches:       return "TooFewMatches";
        case FailureReason::AllHypothesesFailed: return "AllHypothesesFailed";
        case FailureReason::RefinementFailed:    return "RefinementFailed";
        case FailureReason::TooFewInliers:       return "TooFewInliers";
    }
    return "Unknown";
}

Point2d project(const std::array<double, 9>& H, double x, double y) {
    const double w = H[6] * x + H[7] * y + H[8];
    constexpr double kEps = 1e-9;
    if (std::abs(w) < kEps)
        throw std::domain_error("Homogeneous denominator near zero in project()");
    return {(H[0] * x + H[1] * y + H[2]) / w,
            (H[3] * x + H[4] * y + H[5]) / w};
}

double reprojectionError(const std::array<double, 9>& H,
                         double xb, double yb,
                         double xa, double ya) {
    return computeReprojectionError(H, xb, yb, xa, ya);
}

HomographyResult estimate(
    const std::vector<matching::Match>& matches,
    const orb::DescriptionResult&       desc_a,
    const orb::DescriptionResult&       desc_b,
    const HomographyConfig&             config) {

    HomographyResult result;

    // Minimum 4 correspondences for a valid Homography.
    if (matches.size() < 4) {
        result.valid          = false;
        result.failure_reason = FailureReason::TooFewMatches;
        result.error_message  = "Need at least 4 matches, got " +
                                std::to_string(matches.size());
        result.inlier_mask.assign(matches.size(), false);
        return result;
    }

    const std::size_t N = matches.size();

    // Extract point arrays from matches.
    std::vector<double> xa(N), ya(N), xb(N), yb(N);
    for (std::size_t k = 0; k < N; ++k) {
        const auto& m = matches[k];
        if (m.index_a >= desc_a.keypoints.size() ||
            m.index_b >= desc_b.keypoints.size()) {
            result.valid = false;
            result.failure_reason = FailureReason::TooFewMatches;
            result.error_message  = "Match index out of range";
            result.inlier_mask.assign(N, false);
            return result;
        }
        xa[k] = desc_a.keypoints[m.index_a].x;
        ya[k] = desc_a.keypoints[m.index_a].y;
        xb[k] = desc_b.keypoints[m.index_b].x;
        yb[k] = desc_b.keypoints[m.index_b].y;
    }

    // RANSAC.
    std::mt19937 rng(config.random_seed);
    std::uniform_int_distribution<std::size_t> dist(0, N - 1);

    const double thresh_sq = config.reprojection_threshold *
                             config.reprojection_threshold;

    Mat3 best_H{};
    int  best_inlier_count = -1;
    double best_inlier_error_sum = std::numeric_limits<double>::max();
    std::vector<bool> best_mask(N, false);

    bool any_hypothesis_succeeded = false;

    for (int iter = 0; iter < config.ransac_iterations; ++iter) {
        // Sample 4 distinct indices.
        std::array<std::size_t, 4> sample{};
        for (int s = 0; s < 4; ) {
            sample[s] = dist(rng);
            bool unique = true;
            for (int t = 0; t < s; ++t)
                if (sample[t] == sample[s]) { unique = false; break; }
            if (unique) ++s;
        }

        // Degeneracy check.
        if (isSampleDegenerate(xa, ya, xb, yb,
                               sample[0], sample[1], sample[2], sample[3]))
            continue;

        // Build 4-point subsets.
        std::vector<double> sxa(4), sya(4), sxb(4), syb(4);
        for (int s = 0; s < 4; ++s) {
            sxa[s] = xa[sample[s]]; sya[s] = ya[sample[s]];
            sxb[s] = xb[sample[s]]; syb[s] = yb[sample[s]];
        }

        Mat3 H_cand{};
        if (!normalizedDLT(sxa, sya, sxb, syb, H_cand)) continue;

        // Evaluate hypothesis on all matches.
        int inlier_count = 0;
        double inlier_error_sum = 0.0;
        std::vector<bool> mask(N, false);

        for (std::size_t k = 0; k < N; ++k) {
            const double err = computeReprojectionError(H_cand,
                                                       xb[k], yb[k],
                                                       xa[k], ya[k]);
            if (std::isfinite(err) && err * err <= thresh_sq) {
                // Use squared error for comparison consistency
                mask[k] = true;
                ++inlier_count;
                inlier_error_sum += err;
            }
        }

        // Update best model.
        const bool better_count = (inlier_count > best_inlier_count);
        const bool same_count_less_error =
            (inlier_count == best_inlier_count &&
             inlier_error_sum < best_inlier_error_sum);

        if (better_count || same_count_less_error) {
            best_H             = H_cand;
            best_inlier_count  = inlier_count;
            best_inlier_error_sum = inlier_error_sum;
            best_mask          = mask;
            any_hypothesis_succeeded = true;
        }
    }

    if (!any_hypothesis_succeeded) {
        result.valid          = false;
        result.failure_reason = FailureReason::AllHypothesesFailed;
        result.error_message  = "No valid DLT hypothesis could be computed in "
                                + std::to_string(config.ransac_iterations) +
                                " RANSAC iterations";
        result.inlier_mask.assign(N, false);
        return result;
    }

    if (best_inlier_count < config.min_inliers) {
        result.valid          = false;
        result.failure_reason = FailureReason::TooFewInliers;
        result.error_message  = "Best model has only " +
                                std::to_string(best_inlier_count) +
                                " inliers (required: " +
                                std::to_string(config.min_inliers) + ")";
        result.inlier_mask    = best_mask;
        result.num_inliers    = best_inlier_count;
        result.inlier_ratio   = static_cast<double>(best_inlier_count) / N;
        return result;
    }

    // Refinement: re-estimate H from all inliers.
    std::vector<double> ixa, iya, ixb, iyb;
    ixa.reserve(static_cast<std::size_t>(best_inlier_count));
    iya.reserve(static_cast<std::size_t>(best_inlier_count));
    ixb.reserve(static_cast<std::size_t>(best_inlier_count));
    iyb.reserve(static_cast<std::size_t>(best_inlier_count));

    for (std::size_t k = 0; k < N; ++k) {
        if (best_mask[k]) {
            ixa.push_back(xa[k]);
            iya.push_back(ya[k]);
            ixb.push_back(xb[k]);
            iyb.push_back(yb[k]);
        }
    }

    Mat3 H_refined{};
    if (!normalizedDLT(ixa, iya, ixb, iyb, H_refined)) {
        // Refinement failed; fall back to RANSAC best.
        H_refined = best_H;
    }

    // Recompute inlier mask and error statistics with refined H.
    std::vector<bool> final_mask(N, false);
    int final_inlier_count = 0;
    double sum_err = 0.0, max_err = 0.0;

    for (std::size_t k = 0; k < N; ++k) {
        const double err = computeReprojectionError(H_refined,
                                                    xb[k], yb[k],
                                                    xa[k], ya[k]);
        if (std::isfinite(err) && err * err <= thresh_sq) {
            final_mask[k] = true;
            ++final_inlier_count;
            sum_err += err;
            max_err  = std::max(max_err, err);
        }
    }

    if (final_inlier_count < config.min_inliers) {
        result.valid          = false;
        result.failure_reason = FailureReason::RefinementFailed;
        result.error_message  = "Refinement left only " +
                                std::to_string(final_inlier_count) + " inliers";
        result.inlier_mask    = final_mask;
        result.num_inliers    = final_inlier_count;
        result.inlier_ratio   = static_cast<double>(final_inlier_count) / N;
        return result;
    }

    result.valid          = true;
    result.failure_reason = FailureReason::None;
    result.H              = H_refined;
    result.inlier_mask    = final_mask;
    result.num_inliers    = final_inlier_count;
    result.inlier_ratio   = static_cast<double>(final_inlier_count) /
                            static_cast<double>(N);
    result.mean_reprojection_error = (final_inlier_count > 0)
        ? sum_err / final_inlier_count : 0.0;
    result.max_reprojection_error  = max_err;
    return result;
}

}  // namespace stitching::homography
