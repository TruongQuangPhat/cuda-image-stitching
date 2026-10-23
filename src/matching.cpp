// Authors: TRƯƠNG QUANG PHÁT (23120318); CHÂU HUỲNH PHÚC (23120329, S3).
// Brute-force Hamming matching, ratio test and optional cross-check.
// Reference: Lowe (2004), Distinctive image features from scale-invariant keypoints.

#include "matching.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace {

// ---- Hamming popcount on individual uint8 blocks --------------------------
// Uses the compiler builtin when available (GCC/Clang).
// No SIMD, no manual bit tricks.
inline int popcount8(std::uint8_t v) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_popcount(static_cast<unsigned>(v));
#else
    // Kernighan's bit-counting algorithm - portable fallback.
    int count = 0;
    for (; v; v &= static_cast<std::uint8_t>(v - 1u)) ++count;
    return count;
#endif
}

// Validate MatchingConfig. Returns false and sets msg if invalid.
bool validateMatchingConfig(const stitching::matching::MatchingConfig& cfg,
                            std::string& msg) {
    if (!std::isfinite(cfg.ratio_threshold)) {
        msg = "ratio_threshold is not finite (NaN or Inf)";
        return false;
    }
    if (cfg.ratio_threshold <= 0.0f) {
        msg = "ratio_threshold must be > 0";
        return false;
    }
    if (cfg.ratio_threshold >= 1.0f) {
        msg = "ratio_threshold must be < 1 for Lowe's ratio test to be meaningful";
        return false;
    }
    return true;
}

}  // anonymous namespace

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
    // Validate configuration.
    {
        std::string cfg_err;
        if (!validateMatchingConfig(config, cfg_err)) {
            return {{}, false, "Invalid MatchingConfig: " + cfg_err};
        }
    }

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
    result.valid = true;

    // Need at least 2 descriptors in B for a meaningful ratio test.
    if (na == 0 || nb < 2) {
        return result;  // valid=true, no matches
    }

    result.matches.reserve(na);

    // Forward matching: A -> B (find 2 nearest neighbors in B for each in A).
    // Stores best-match index in B for each descriptor in A.
    std::vector<std::size_t> fwd_best(na, 0);

    for (std::size_t i = 0; i < na; ++i) {
        int d1 = std::numeric_limits<int>::max();
        int d2 = std::numeric_limits<int>::max();
        std::size_t j1 = 0;

        for (std::size_t j = 0; j < nb; ++j) {
            const int d = hammingDistance(desc_a.descriptors[i],
                                          desc_b.descriptors[j]);
            if (d < d1) {
                d2 = d1;
                d1 = d;
                j1 = j;
            } else if (d < d2) {
                d2 = d;
            }
        }

        fwd_best[i] = j1;

        // Ratio test: d1 < ratio_threshold * d2.
        // Guard against d2 == 0 (ambiguous match - both nearest neighbors identical).
        if (d2 == 0) continue;
        if (static_cast<float>(d1) < config.ratio_threshold *
                                     static_cast<float>(d2)) {
            result.matches.push_back({i, j1, d1, d2});
        }
    }

    // Cross-check: keep only mutual nearest neighbors.
    // For each accepted forward match A[i] -> B[j], verify that the nearest
    // neighbor of B[j] in A is also A[i].
    if (config.cross_check && !result.matches.empty()) {
        // Build reverse best-match: for each descriptor in B, find nearest in A.
        std::vector<std::size_t> rev_best(nb, 0);
        for (std::size_t j = 0; j < nb; ++j) {
            int best_d = std::numeric_limits<int>::max();
            std::size_t best_i = 0;
            for (std::size_t i = 0; i < na; ++i) {
                const int d = hammingDistance(desc_b.descriptors[j],
                                              desc_a.descriptors[i]);
                if (d < best_d) { best_d = d; best_i = i; }
            }
            rev_best[j] = best_i;
        }

        // Filter: keep only matches where forward and reverse agree.
        std::vector<Match> filtered;
        filtered.reserve(result.matches.size());
        for (const auto& m : result.matches) {
            if (rev_best[m.index_b] == m.index_a)
                filtered.push_back(m);
        }
        result.matches = std::move(filtered);
    }

    return result;
}

}  // namespace stitching::matching
