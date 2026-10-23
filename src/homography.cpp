// Authors: TRƯƠNG QUANG PHÁT (23120318); CHÂU HUỲNH PHÚC (23120329, S4).
// Normalized DLT, Jacobi eigensolver, RANSAC and inlier refinement.
// References: Hartley and Zisserman, Multiple View Geometry (2004);
// Fischler and Bolles, Random Sample Consensus (1981);
// Golub and Van Loan, Matrix Computations (2013), Jacobi methods.

#include "homography.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// ---- 3x3 matrix helpers (row-major, indices [row*3+col]) ------------------
using Mat3 = std::array<double, 9>;

inline double det3(const Mat3& m) {
    return m[0] * (m[4] * m[8] - m[5] * m[7])
         - m[1] * (m[3] * m[8] - m[5] * m[6])
         + m[2] * (m[3] * m[7] - m[4] * m[6]);
}

// Use cancellation relative to determinant terms, not an absolute determinant
// cutoff: Hartley transforms can have a tiny but well-conditioned scale.
bool invert3(const Mat3& m, Mat3& inv) {
    for (double value : m)
        if (!std::isfinite(value)) return false;
    const double d = det3(m);
    const double scale = std::abs(m[0] * (m[4] * m[8] - m[5] * m[7])) +
                         std::abs(m[1] * (m[3] * m[8] - m[5] * m[6])) +
                         std::abs(m[2] * (m[3] * m[7] - m[4] * m[6]));
    constexpr double kRelativeEpsilon = 64.0 * std::numeric_limits<double>::epsilon();
    if (!std::isfinite(d) || !std::isfinite(scale) || scale == 0.0 ||
        std::abs(d) <= kRelativeEpsilon * scale) return false;
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
    for (double value : inv)
        if (!std::isfinite(value)) return false;
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
// Returns false if points are degenerate (fewer than 1, or all coincident,
// or any coordinate is non-finite).
struct NormResult {
    Mat3 T;
    std::vector<double> xs;  // Normalized x coordinates.
    std::vector<double> ys;  // Normalized y coordinates.
};

bool hartleyNormalize(const std::vector<double>& x,
                      const std::vector<double>& y,
                      NormResult& out) {
    const std::size_t n = x.size();
    if (n == 0 || y.size() != n) return false;

    // Verify finiteness before computing statistics.
    for (std::size_t i = 0; i < n; ++i)
        if (!std::isfinite(x[i]) || !std::isfinite(y[i])) return false;

    // Centroid.
    double cx = 0.0, cy = 0.0;
    for (std::size_t i = 0; i < n; ++i) { cx += x[i]; cy += y[i]; }
    cx /= static_cast<double>(n);
    cy /= static_cast<double>(n);
    if (!std::isfinite(cx) || !std::isfinite(cy)) return false;

    // Mean distance from centroid.
    double mean_dist = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double dx = x[i] - cx;
        const double dy = y[i] - cy;
        mean_dist += std::hypot(dx, dy);  // std::hypot avoids overflow
    }
    mean_dist /= static_cast<double>(n);

    // Avoid division by zero when all points are coincident.
    constexpr double kEps = 1e-9;
    if (!std::isfinite(mean_dist) || mean_dist < kEps) return false;

    const double s = std::sqrt(2.0) / mean_dist;
    if (!std::isfinite(s) || s <= 0.0) return false;

    // T = [[s, 0, -s*cx], [0, s, -s*cy], [0, 0, 1]]
    out.T = {};
    out.T[0] = s;   out.T[1] = 0.0; out.T[2] = -s * cx;
    out.T[3] = 0.0; out.T[4] = s;   out.T[5] = -s * cy;
    out.T[6] = 0.0; out.T[7] = 0.0; out.T[8] =  1.0;
    for (double value : out.T)
        if (!std::isfinite(value)) return false;

    out.xs.resize(n);
    out.ys.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        out.xs[i] = s * (x[i] - cx);
        out.ys[i] = s * (y[i] - cy);
        if (!std::isfinite(out.xs[i]) || !std::isfinite(out.ys[i])) return false;
    }
    return true;
}

// ---- 9x9 symmetric Jacobi eigendecomposition ------------------------------
// We solve the DLT null-space by finding the eigenvector of A^T*A with the
// smallest eigenvalue, using the classical Jacobi method.
//
// Implementation: Golub & Van Loan, "Matrix Computations" (4th ed.), Sec. 8.4.
// Note: Using A^T*A squares the condition number. This is numerically adequate
// for typical image-resolution data after Hartley normalization (mean_dist=sqrt(2)).
// If conditioning is a concern, consider direct one-sided Jacobi SVD on A.
//
// Returns false when:
//   - Jacobi fails to converge (max off-diagonal still large after kMaxSweeps).
//   - Any eigenvalue or eigenvector entry is non-finite.

using SymMat9 = std::array<double, 81>;

// Build A^T * A from a (2n x 9) matrix stored row by row.
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

// Symmetric 9x9 Jacobi eigendecomposition.
// On return:
//   eigenvalues[i] = i-th eigenvalue (diagonal of A after diagonalization).
//   V[:,i] = i-th eigenvector (columns of V, V stored row-major).
// Returns false if convergence was not achieved or the result is non-finite.
bool jacobiSym9(SymMat9 A, std::array<double, 9>& eigenvalues,
                std::array<double, 81>& V) {
    // Initialize V to identity.
    V.fill(0.0);
    for (int i = 0; i < 9; ++i) V[i * 9 + i] = 1.0;

    constexpr int    kMaxSweeps = 1000;

    // Convergence tolerance: relative to the Frobenius norm of A.
    // Compute initial Frobenius norm for scaling.
    double frob_sq = 0.0;
    for (int i = 0; i < 9; ++i)
        for (int j = 0; j < 9; ++j)
            frob_sq += A[i * 9 + j] * A[i * 9 + j];
    const double kAbsTol = 1e-14 * std::sqrt(frob_sq) + 1e-30;

    for (int sweep = 0; sweep < kMaxSweeps; ++sweep) {
        // Find off-diagonal element of maximum absolute value.
        double max_off = 0.0;
        for (int p = 0; p < 9; ++p)
            for (int q = p + 1; q < 9; ++q)
                max_off = std::max(max_off, std::abs(A[p * 9 + q]));
        if (max_off < kAbsTol) {
            // Converged. Check output is finite.
            for (int i = 0; i < 9; ++i) {
                eigenvalues[i] = A[i * 9 + i];
                if (!std::isfinite(eigenvalues[i])) return false;
            }
            for (double v : V)
                if (!std::isfinite(v)) return false;
            return true;
        }

        // One sweep: rotate all off-diagonal pairs.
        for (int p = 0; p < 8; ++p) {
            for (int q = p + 1; q < 9; ++q) {
                const double apq = A[p * 9 + q];
                if (std::abs(apq) < kAbsTol * 0.001) continue;
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

    // Did not converge within kMaxSweeps sweeps.
    return false;
}

// ---- Normalized DLT for Homography estimation ----------------------------
// Given N>=4 correspondences (xb[i],yb[i]) -> (xa[i],ya[i]),
// estimates H_B_to_A such that xa_i ~ H * xb_i.
// Returns true on success; H is written in row-major order.
// Algorithm: Hartley (2004), Algorithm 4.2.
bool normalizedDLT(const std::vector<double>& xa,
                   const std::vector<double>& ya,
                   const std::vector<double>& xb,
                   const std::vector<double>& yb,
                   Mat3& H) {
    const std::size_t n = xa.size();
    if (n < 4 || xb.size() != n || ya.size() != n || yb.size() != n)
        return false;

    // Hartley normalization - also validates finiteness internally.
    NormResult Na, Nb;
    if (!hartleyNormalize(xa, ya, Na)) return false;
    if (!hartleyNormalize(xb, yb, Nb)) return false;

    // Build 2n x 9 DLT matrix A row-by-row.
    // Using normalized coordinates Xa,Ya (image A) and Xb,Yb (image B).
    // Each correspondence gives two rows (Hartley & Zisserman, p. 91):
    //   Row 1: [0, 0, 0, -Xb, -Yb, -1, Ya*Xb, Ya*Yb, Ya]
    //   Row 2: [Xb, Yb, 1, 0, 0, 0, -Xa*Xb, -Xa*Yb, -Xa]
    std::vector<std::array<double, 9>> rows;
    rows.reserve(2 * n);
    for (std::size_t i = 0; i < n; ++i) {
        const double Xa = Na.xs[i], Ya = Na.ys[i];
        const double Xb = Nb.xs[i], Yb = Nb.ys[i];
        rows.push_back({0.0, 0.0, 0.0, -Xb, -Yb, -1.0, Ya * Xb, Ya * Yb, Ya});
        rows.push_back({Xb, Yb, 1.0, 0.0, 0.0, 0.0, -Xa * Xb, -Xa * Yb, -Xa});
    }

    // Build A^T * A and find eigenvector of smallest eigenvalue.
    const SymMat9 ATA = buildATA(rows);
    std::array<double, 9> eigenvalues{};
    std::array<double, 81> V{};
    if (!jacobiSym9(ATA, eigenvalues, V)) return false;

    // Find index of smallest eigenvalue.
    // Small eigenvalues near 0 correspond to near-null vectors.
    // If two smallest eigenvalues are nearly equal, the null-space is
    // degenerate (rank deficiency >= 2), making H non-unique. Reject.
    int min_idx = 0;
    for (int i = 1; i < 9; ++i)
        if (eigenvalues[i] < eigenvalues[min_idx]) min_idx = i;

    // Rank deficiency check: ratio of two smallest eigenvalues relative to max eigenvalue.
    // For a non-degenerate 4+ point configuration, rank(A) >= 8, so nullspace dim == 1.
    // If both sorted_ev[0] and sorted_ev[1] are near zero relative to max_ev,
    // the nullspace is at least 2-dimensional (degenerate collinear points, etc.).
    std::array<double, 9> sorted_ev = eigenvalues;
    std::sort(sorted_ev.begin(), sorted_ev.end());
    const double max_ev = sorted_ev[8];
    if (!std::isfinite(max_ev) || max_ev < 1e-12) return false;

    constexpr double kRankRelTol = 1e-9;
    if (sorted_ev[0] < kRankRelTol * max_ev && sorted_ev[1] < kRankRelTol * max_ev) {
        return false;  // Rank deficiency: H is not uniquely determined.
    }

    // Residual of eigenpair: ||A^T A * v - lambda * v||_2
    double res_sq = 0.0;
    const double lambda = eigenvalues[min_idx];
    for (int i = 0; i < 9; ++i) {
        double Av_i = 0.0;
        for (int j = 0; j < 9; ++j) {
            Av_i += ATA[i * 9 + j] * V[j * 9 + min_idx];
        }
        const double r_i = Av_i - lambda * V[i * 9 + min_idx];
        res_sq += r_i * r_i;
    }
    if (std::sqrt(res_sq) > 1e-7 * max_ev + 1e-10) {
        return false;  // Eigenpair residual too large
    }

    // Extract the eigenvector corresponding to smallest eigenvalue.
    Mat3 Hn{};
    for (int i = 0; i < 9; ++i) Hn[i] = V[i * 9 + min_idx];

    // Eigenvector must be finite (already checked in jacobiSym9, but be safe).
    for (double v : Hn)
        if (!std::isfinite(v)) return false;

    // Denormalize: H = T_a^{-1} * Hn * T_b
    Mat3 T_a_inv{};
    if (!invert3(Na.T, T_a_inv)) return false;

    H = matmul3(T_a_inv, matmul3(Hn, Nb.T));

    // Scale to remove projective ambiguity.
    // Prefer H[8] != 0; otherwise use largest element.
    constexpr double kEps = 1e-12;
    if (std::abs(H[8]) > kEps) {
        const double s = 1.0 / H[8];
        for (double& v : H) v *= s;
    } else {
        double max_abs = 0.0;
        for (double v : H) max_abs = std::max(max_abs, std::abs(v));
        if (max_abs < kEps) return false;
        for (double& v : H) v /= max_abs;
    }

    // Final finiteness check.
    for (double v : H)
        if (!std::isfinite(v)) return false;

    return true;
}

// ---- Degeneracy check for a 4-point sample --------------------------------
// Returns true if the four points in A or B are nearly collinear or contain
// nearly coincident pairs.
// Thresholds are relative to coordinate scale to handle varying image sizes.
bool isSampleDegenerate(const std::vector<double>& xa,
                        const std::vector<double>& ya,
                        const std::vector<double>& xb,
                        const std::vector<double>& yb,
                        std::size_t i0, std::size_t i1,
                        std::size_t i2, std::size_t i3,
                        double scale_a, double scale_b) {
    // Minimum normalized distance: 1e-3 of the coordinate scale.
    const double kMinDist2_a = (1e-3 * scale_a) * (1e-3 * scale_a);
    const double kMinDist2_b = (1e-3 * scale_b) * (1e-3 * scale_b);
    // Minimum area for collinearity: 1e-3 * scale^2.
    const double kMinArea_a = 1e-3 * scale_a * scale_a;
    const double kMinArea_b = 1e-3 * scale_b * scale_b;

    std::array<std::size_t, 4> idx = {i0, i1, i2, i3};
    for (int p = 0; p < 4; ++p) {
        for (int q = p + 1; q < 4; ++q) {
            const double dxa = xa[idx[p]] - xa[idx[q]];
            const double dya = ya[idx[p]] - ya[idx[q]];
            const double dxb = xb[idx[p]] - xb[idx[q]];
            const double dyb = yb[idx[p]] - yb[idx[q]];
            if (dxa * dxa + dya * dya < kMinDist2_a) return true;
            if (dxb * dxb + dyb * dyb < kMinDist2_b) return true;
        }
    }

    // Collinearity check: all 4 triangles formed by choosing 3 of 4 points.
    auto crossZ = [](double ax, double ay, double bx, double by,
                     double cx, double cy) {
        return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
    };
    for (int skip = 0; skip < 4; ++skip) {
        std::array<std::size_t, 3> tri;
        int k = 0;
        for (int p = 0; p < 4; ++p) if (p != skip) tri[k++] = idx[p];
        if (std::abs(crossZ(xa[tri[0]], ya[tri[0]],
                            xa[tri[1]], ya[tri[1]],
                            xa[tri[2]], ya[tri[2]])) < kMinArea_a) return true;
        if (std::abs(crossZ(xb[tri[0]], yb[tri[0]],
                            xb[tri[1]], yb[tri[1]],
                            xb[tri[2]], yb[tri[2]])) < kMinArea_b) return true;
    }
    return false;
}

// Compute coordinate scale (RMS of point spreads) for degeneracy thresholding.
double computeCoordScale(const std::vector<double>& x,
                         const std::vector<double>& y) {
    if (x.empty()) return 1.0;
    double cx = 0.0, cy = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i) { cx += x[i]; cy += y[i]; }
    cx /= static_cast<double>(x.size());
    cy /= static_cast<double>(x.size());
    double rms = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        rms += (x[i]-cx)*(x[i]-cx) + (y[i]-cy)*(y[i]-cy);
    }
    rms = std::sqrt(rms / static_cast<double>(x.size()));
    return std::max(rms, 1.0);  // Minimum scale of 1 pixel.
}

// Internal result type for homogeneous projection, separating validity from value.
// This avoids any reliance on DBL_MAX arithmetic in RANSAC inlier classification.
struct ProjResult {
    double x  = 0.0;
    double y  = 0.0;
    bool valid = false;
};

// Project point (xb, yb) through H_B_to_A with explicit validity tracking.
// Returns {px, py, true} on success, {0,0,false} on degeneracy or overflow.
// Pointwise validity only, not matrix invertibility. Scaling is subject to the
// finite representation and normalization limits documented by project().
static ProjResult projectInternal(const Mat3& H, double xb, double yb) {
    if (!std::isfinite(xb) || !std::isfinite(yb)) return {};

    double max_h = 0.0;
    for (double v : H) {
        if (!std::isfinite(v)) return {};
        max_h = std::max(max_h, std::abs(v));
    }
    if (max_h < 1e-300) return {};

    const double inv_m = 1.0 / max_h;
    // Normalized homogeneous row: h6, h7, h8 are the last row divided by max_h.
    const double h6 = H[6] * inv_m;
    const double h7 = H[7] * inv_m;
    const double h8 = H[8] * inv_m;
    const double w  = h6 * xb + h7 * yb + h8;

    // Denominator tolerance: relative to the magnitude of its own terms.
    // This correctly handles both tiny H (scale invariance) and large input coords.
    // We compare |w| against a fraction of the largest term that contributed to it.
    const double denom_scale = std::max({std::abs(h6 * xb),
                                         std::abs(h7 * yb),
                                         std::abs(h8),
                                         1e-300});
    if (!std::isfinite(w) || std::abs(w) < 1e-12 * denom_scale) return {};

    const double px = ((H[0]*inv_m)*xb + (H[1]*inv_m)*yb + (H[2]*inv_m)) / w;
    const double py = ((H[3]*inv_m)*xb + (H[4]*inv_m)*yb + (H[5]*inv_m)) / w;
    if (!std::isfinite(px) || !std::isfinite(py)) return {};

    return {px, py, true};
}

// Project point (xb, yb) in B through H_B_to_A and return reprojection error.
// Returns std::numeric_limits<double>::max() on degeneracy or non-finite result.
// The returned sentinel is guaranteed to be > any finite threshold (use the
// explicit validity flag in projectInternal for inlier classification).
double computeReprojectionError(const Mat3& H,
                                double xb, double yb,
                                double xa, double ya) {
    if (!std::isfinite(xa) || !std::isfinite(ya))
        return std::numeric_limits<double>::max();
    const ProjResult pr = projectInternal(H, xb, yb);
    if (!pr.valid) return std::numeric_limits<double>::max();
    const double err = std::hypot(pr.x - xa, pr.y - ya);
    // If somehow err itself overflowed (extremely large coords), signal invalid.
    if (!std::isfinite(err)) return std::numeric_limits<double>::max();
    return err;
}

// Validate HomographyConfig. Returns false and sets msg if invalid.
bool validateHomographyConfig(const stitching::homography::HomographyConfig& cfg,
                              std::string& msg) {
    if (cfg.ransac_iterations <= 0) {
        msg = "ransac_iterations must be > 0";
        return false;
    }
    if (!std::isfinite(cfg.reprojection_threshold) ||
        cfg.reprojection_threshold <= 0.0) {
        msg = "reprojection_threshold must be finite and > 0";
        return false;
    }
    if (cfg.min_inliers < 4) {
        msg = "min_inliers must be >= 4 (DLT requires at least 4 correspondences)";
        return false;
    }
    if (!std::isfinite(cfg.min_inlier_ratio) ||
        cfg.min_inlier_ratio < 0.0 || cfg.min_inlier_ratio > 1.0) {
        msg = "min_inlier_ratio must be finite and in [0.0, 1.0]";
        return false;
    }
    return true;
}

}  // anonymous namespace

namespace stitching::homography {

std::string failureReasonToString(FailureReason r) {
    switch (r) {
        case FailureReason::None:                 return "None";
        case FailureReason::InvalidConfig:        return "InvalidConfig";
        case FailureReason::InvalidInput:         return "InvalidInput";
        case FailureReason::TooFewMatches:        return "TooFewMatches";
        case FailureReason::AllHypothesesFailed:  return "AllHypothesesFailed";
        case FailureReason::RefinementFailed:     return "RefinementFailed";
        case FailureReason::TooFewInliers:        return "TooFewInliers";
    }
    return "Unknown";
}

Point2d project(const std::array<double, 9>& H, double x, double y) {
    if (!std::isfinite(x) || !std::isfinite(y))
        throw std::domain_error("project(): Input coordinates are non-finite");

    // Use projectInternal (defined in anonymous namespace) which correctly handles
    // scale-invariance and denominator tolerance for large input coordinates.
    const ProjResult pr = projectInternal(H, x, y);

    // Propagate specific failure modes as domain_error.
    if (!std::isfinite(H[0]) || !std::isfinite(H[1]) || !std::isfinite(H[2]) ||
        !std::isfinite(H[3]) || !std::isfinite(H[4]) || !std::isfinite(H[5]) ||
        !std::isfinite(H[6]) || !std::isfinite(H[7]) || !std::isfinite(H[8]))
        throw std::domain_error("project(): Homography contains non-finite values");

    double max_h = 0.0;
    for (double v : H) max_h = std::max(max_h, std::abs(v));
    if (max_h < 1e-300)
        throw std::domain_error("project(): Homography is all zeros or degenerate");

    if (!pr.valid)
        throw std::domain_error("project(): Homogeneous denominator near zero or result non-finite");

    return {pr.x, pr.y};
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

    // --- Configuration validation ---
    {
        std::string cfg_err;
        if (!validateHomographyConfig(config, cfg_err)) {
            result.valid          = false;
            result.failure_reason = FailureReason::InvalidConfig;
            result.error_message  = "Invalid HomographyConfig: " + cfg_err;
            result.inlier_mask.assign(matches.size(), false);
            return result;
        }
    }

    // --- Match count check ---
    if (matches.size() < 4) {
        result.valid          = false;
        result.failure_reason = FailureReason::TooFewMatches;
        result.error_message  = "Need at least 4 matches, got " +
                                std::to_string(matches.size());
        result.inlier_mask.assign(matches.size(), false);
        return result;
    }

    const std::size_t N = matches.size();

    // --- Input correspondence validation and coordinate extraction ---
    std::vector<double> xa(N), ya(N), xb(N), yb(N);
    for (std::size_t k = 0; k < N; ++k) {
        const auto& m = matches[k];
        if (m.index_a >= desc_a.keypoints.size() ||
            m.index_b >= desc_b.keypoints.size()) {
            result.valid          = false;
            result.failure_reason = FailureReason::InvalidInput;
            result.error_message  = "Match[" + std::to_string(k) +
                                    "]: index out of range (index_a=" +
                                    std::to_string(m.index_a) +
                                    ", desc_a.size=" +
                                    std::to_string(desc_a.keypoints.size()) + ")";
            result.inlier_mask.assign(N, false);
            return result;
        }
        const float raw_xa = desc_a.keypoints[m.index_a].x;
        const float raw_ya = desc_a.keypoints[m.index_a].y;
        const float raw_xb = desc_b.keypoints[m.index_b].x;
        const float raw_yb = desc_b.keypoints[m.index_b].y;

        // Reject any correspondence with non-finite coordinates.
        if (!std::isfinite(raw_xa) || !std::isfinite(raw_ya) ||
            !std::isfinite(raw_xb) || !std::isfinite(raw_yb)) {
            result.valid          = false;
            result.failure_reason = FailureReason::InvalidInput;
            result.error_message  = "Match[" + std::to_string(k) +
                                    "]: non-finite keypoint coordinate";
            result.inlier_mask.assign(N, false);
            return result;
        }

        xa[k] = raw_xa; ya[k] = raw_ya;
        xb[k] = raw_xb; yb[k] = raw_yb;
    }

    // Compute coordinate scales for scale-adaptive degeneracy detection.
    const double scale_a = computeCoordScale(xa, ya);
    const double scale_b = computeCoordScale(xb, yb);

    // RANSAC.
    std::mt19937 rng(config.random_seed);
    std::uniform_int_distribution<std::size_t> dist(0, N - 1);

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

        // Scale-adaptive degeneracy check.
        if (isSampleDegenerate(xa, ya, xb, yb,
                               sample[0], sample[1], sample[2], sample[3],
                               scale_a, scale_b))
            continue;

        // Build 4-point subsets for DLT.
        std::vector<double> sxa(4), sya(4), sxb(4), syb(4);
        for (int s = 0; s < 4; ++s) {
            sxa[s] = xa[sample[s]]; sya[s] = ya[sample[s]];
            sxb[s] = xb[sample[s]]; syb[s] = yb[sample[s]];
        }

        Mat3 H_cand{};
        if (!normalizedDLT(sxa, sya, sxb, syb, H_cand)) continue;

        // Evaluate hypothesis on all matches.
        // IMPORTANT: Use projectInternal for inlier classification to avoid the
        // DBL_MAX overflow bug: DBL_MAX*DBL_MAX = Inf, and Inf <= Inf = true,
        // which would incorrectly count invalid projections as inliers when
        // the threshold is very large. Direct error comparison avoids this.
        int inlier_count = 0;
        double inlier_error_sum = 0.0;
        std::vector<bool> mask(N, false);

        for (std::size_t k = 0; k < N; ++k) {
            // projectInternal returns valid=false for degenerate/invalid projections.
            // Only finite, valid projections can be inliers – regardless of threshold.
            const ProjResult pr = projectInternal(H_cand, xb[k], yb[k]);
            if (!pr.valid) continue;  // invalid projection: never an inlier
            const double err = std::hypot(pr.x - xa[k], pr.y - ya[k]);
            if (!std::isfinite(err)) continue;  // overflow guard
            if (err <= config.reprojection_threshold) {
                mask[k] = true;
                ++inlier_count;
                inlier_error_sum += err;
            }
        }

        // Update best model (by inlier count, then by total error as tiebreaker).
        const bool better_count = (inlier_count > best_inlier_count);
        const bool same_count_less_error =
            (inlier_count == best_inlier_count &&
             inlier_error_sum < best_inlier_error_sum);

        if (better_count || same_count_less_error) {
            best_H                = H_cand;
            best_inlier_count     = inlier_count;
            best_inlier_error_sum = inlier_error_sum;
            best_mask             = mask;
            any_hypothesis_succeeded = true;
        }
    }

    if (!any_hypothesis_succeeded) {
        result.valid          = false;
        result.failure_reason = FailureReason::AllHypothesesFailed;
        result.error_message  = "No valid DLT hypothesis in " +
                                std::to_string(config.ransac_iterations) +
                                " RANSAC iterations";
        result.inlier_mask.assign(N, false);
        return result;
    }

    const double best_inlier_ratio = static_cast<double>(best_inlier_count) / static_cast<double>(N);
    if (best_inlier_count < config.min_inliers ||
        (config.min_inlier_ratio > 0.0 && best_inlier_ratio < config.min_inlier_ratio)) {
        result.valid          = false;
        result.failure_reason = FailureReason::TooFewInliers;
        result.error_message  = "Best model has " +
                                std::to_string(best_inlier_count) +
                                " inliers (" + std::to_string(best_inlier_ratio) +
                                " ratio), required >= " +
                                std::to_string(config.min_inliers) +
                                " inliers and >= " +
                                std::to_string(config.min_inlier_ratio) + " ratio";
        result.inlier_mask    = best_mask;
        // Diagnostics from the rejected best hypothesis, not from result.H.
        // H remains all-zeros and must not be used when valid is false.
        result.num_inliers    = best_inlier_count;
        result.inlier_ratio   = best_inlier_ratio;
        return result;
    }

    // Refinement: re-estimate H from all inliers using normalized DLT.
    std::vector<double> ixa, iya, ixb, iyb;
    ixa.reserve(static_cast<std::size_t>(best_inlier_count));
    iya.reserve(static_cast<std::size_t>(best_inlier_count));
    ixb.reserve(static_cast<std::size_t>(best_inlier_count));
    iyb.reserve(static_cast<std::size_t>(best_inlier_count));
    for (std::size_t k = 0; k < N; ++k) {
        if (best_mask[k]) {
            ixa.push_back(xa[k]); iya.push_back(ya[k]);
            ixb.push_back(xb[k]); iyb.push_back(yb[k]);
        }
    }

    Mat3 H_refined{};
    const bool solver_succeeded = normalizedDLT(ixa, iya, ixb, iyb, H_refined);

    std::vector<bool> refined_mask(N, false);
    int refined_inlier_count = 0;
    double refined_sum_err = 0.0, refined_max_err = 0.0;

    if (solver_succeeded) {
        for (std::size_t k = 0; k < N; ++k) {
            // Use projectInternal consistently with RANSAC evaluation loop.
            const ProjResult pr = projectInternal(H_refined, xb[k], yb[k]);
            if (!pr.valid) continue;
            const double err = std::hypot(pr.x - xa[k], pr.y - ya[k]);
            if (!std::isfinite(err)) continue;
            if (err <= config.reprojection_threshold) {
                refined_mask[k] = true;
                ++refined_inlier_count;
                refined_sum_err += err;
                refined_max_err  = std::max(refined_max_err, err);
            }
        }
    }

    // Acceptance policy:
    // Refined model must meet min_inliers and min_inlier_ratio.
    // Prefer more inliers, or the same count with error within the tolerance below.
    const double refined_ratio = static_cast<double>(refined_inlier_count) / static_cast<double>(N);
    const bool meets_confidence = (refined_inlier_count >= config.min_inliers) &&
                                  (config.min_inlier_ratio <= 0.0 || refined_ratio >= config.min_inlier_ratio);

    const bool ref_more_inliers = (refined_inlier_count > best_inlier_count);
    // Floating-point keypoint coordinates (float) have discretization precision ~1e-4 px.
    // When error sums are equal within this sub-pixel tolerance, the refined model
    // (estimated from all consensus inliers) is accepted as the deterministic tie-breaker.
    const double kSubpixelTol = 1e-4 * static_cast<double>(best_inlier_count);
    const bool ref_same_inliers_lower_err =
        (refined_inlier_count == best_inlier_count &&
         (refined_sum_err <= best_inlier_error_sum + kSubpixelTol));

    const bool accept_refined = solver_succeeded && meets_confidence &&
                                (ref_more_inliers || ref_same_inliers_lower_err);

    Mat3 final_H{};
    std::vector<bool> final_mask;
    int final_inlier_count = 0;
    double final_sum_err = 0.0, final_max_err = 0.0;

    if (accept_refined) {
        final_H            = H_refined;
        final_mask         = refined_mask;
        final_inlier_count = refined_inlier_count;
        final_sum_err      = refined_sum_err;
        final_max_err      = refined_max_err;
        result.refinement_succeeded = true;
    } else {
        // Fallback to the already accepted RANSAC hypothesis. Neither a solver
        // failure nor rejection of the refinement changes success into failure.
        final_H            = best_H;
        final_mask         = best_mask;
        final_inlier_count = best_inlier_count;
        final_sum_err      = best_inlier_error_sum;
        final_max_err      = 0.0;
        for (std::size_t k = 0; k < N; ++k) {
            if (best_mask[k]) {
                const ProjResult pr = projectInternal(best_H, xb[k], yb[k]);
                if (!pr.valid) continue;
                const double err = std::hypot(pr.x - xa[k], pr.y - ya[k]);
                if (std::isfinite(err)) final_max_err = std::max(final_max_err, err);
            }
        }
        result.refinement_succeeded = false;
        if (!solver_succeeded) {
            result.error_message = "(info) DLT refinement solver failed; retained RANSAC best H";
        } else {
            result.error_message = "(info) Refined model not accepted (inliers: " +
                                   std::to_string(refined_inlier_count) + " vs " +
                                   std::to_string(best_inlier_count) +
                                   ", err: " + std::to_string(refined_sum_err) +
                                   " vs " + std::to_string(best_inlier_error_sum) +
                                   "); retained RANSAC best H";
        }
    }

    result.valid          = true;
    result.failure_reason = FailureReason::None;
    result.H              = final_H;
    result.inlier_mask    = final_mask;
    result.num_inliers    = final_inlier_count;
    result.inlier_ratio   = static_cast<double>(final_inlier_count) / static_cast<double>(N);
    result.mean_reprojection_error = (final_inlier_count > 0)
        ? final_sum_err / final_inlier_count : 0.0;
    result.max_reprojection_error  = final_max_err;
    return result;
}

}  // namespace stitching::homography
