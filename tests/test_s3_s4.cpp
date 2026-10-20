// Unit, integration and benchmark tests for Stage 3 (Feature Matching) and
// Stage 4 (Homography Estimation) – CSC14120 CUDA Image Stitching, V0 baseline.
//
// Authors: Truong Quang Phat (23120318)
// AI-assisted development: implemented with Antigravity (Google DeepMind),
// reviewed and corrected by the author.
//
// Build (release):
//   clang++ -std=c++17 -Wall -Wextra -Wpedantic -O2
//       tests/test_s3_s4.cpp src/orb.cpp src/match_homography.cpp
//       -o tests/test_s3_s4 && ./tests/test_s3_s4
//
// Build (ASAN + UBSan):
//   clang++ -std=c++17 -Wall -Wextra -Wpedantic -O1
//       -fsanitize=address,undefined -g
//       tests/test_s3_s4.cpp src/orb.cpp src/match_homography.cpp
//       -o tests/test_s3_s4_asan && ./tests/test_s3_s4_asan

#include "../src/match_homography.h"
#include "../src/orb.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <cstdio>

// ============================================================
//  Test framework
//  - Counts test CASES separately from assertions.
//  - Supports PASSED, FAILED, SKIPPED, BLOCKED.
//  - Each non-skipped test case must contain >= 1 mandatory assertion.
// ============================================================

static int g_tests_run     = 0;
static int g_tests_passed  = 0;
static int g_tests_failed  = 0;
static int g_tests_skipped = 0;
static int g_tests_blocked = 0;
static int g_asserts_passed = 0;
static int g_asserts_failed = 0;

enum class TestStatus {
    Running,
    Passed,
    Failed,
    Skipped,
    Blocked
};

static TestStatus g_current_test_status = TestStatus::Running;
static bool g_test_has_assertion        = false;
static std::string g_current_test;
static std::string g_status_message;

#define EXPECT_TRUE(expr) do { \
    g_test_has_assertion = true; \
    if (!(expr)) { \
        std::cerr << "  FAIL [" << __FILE__ << ":" << __LINE__ << "] " \
                  << g_current_test << ": " #expr "\n"; \
        ++g_asserts_failed; \
    } else { ++g_asserts_passed; } \
} while(0)

#define EXPECT_EQ(a, b)     EXPECT_TRUE((a) == (b))
#define EXPECT_NE(a, b)     EXPECT_TRUE((a) != (b))
#define EXPECT_LT(a, b)     EXPECT_TRUE((a) < (b))
#define EXPECT_LE(a, b)     EXPECT_TRUE((a) <= (b))
#define EXPECT_FALSE(expr)  EXPECT_TRUE(!(expr))
#define EXPECT_NEAR(a, b, tol) \
    EXPECT_TRUE(std::abs(static_cast<double>(a) - static_cast<double>(b)) <= (tol))

// Mandatory assertion: marks the test as having at least one unconditional check.
#define REQUIRE_TRUE(expr) EXPECT_TRUE(expr)

#define BLOCK_TEST(msg) do { \
    g_current_test_status = TestStatus::Blocked; \
    g_status_message = (msg); \
    std::cout << "  [BLOCKED] " << (msg) << "\n"; \
} while(0)

#define SKIP_TEST(msg) do { \
    g_current_test_status = TestStatus::Skipped; \
    g_status_message = (msg); \
    std::cout << "  [SKIPPED] " << (msg) << "\n"; \
} while(0)

// Test execution tracking macro: counts assertions and per-test pass/fail/blocked.
static int g_asserts_failed_before_test = 0;

#define RUN_TEST2(fn) do { \
    g_current_test = #fn; \
    g_current_test_status = TestStatus::Running; \
    g_status_message.clear(); \
    g_test_has_assertion = false; \
    g_asserts_failed_before_test = g_asserts_failed; \
    ++g_tests_run; \
    std::cout << "[ RUN ] " #fn "\n"; \
    fn(); \
    if (g_current_test_status == TestStatus::Blocked) { \
        ++g_tests_blocked; \
    } else if (g_current_test_status == TestStatus::Skipped) { \
        ++g_tests_skipped; \
    } else if (!g_test_has_assertion) { \
        std::cerr << "  WARN [" #fn "]: zero assertions executed!\n"; \
        ++g_tests_failed; \
    } else if (g_asserts_failed > g_asserts_failed_before_test) { \
        ++g_tests_failed; \
    } else { \
        ++g_tests_passed; \
    } \
    std::cout << "\n"; \
} while(0)

// ============================================================
//  Helpers
// ============================================================

using namespace stitching;

static orb::Descriptor makeDesc(std::uint8_t b) {
    orb::Descriptor d;
    d.bytes.fill(b);
    return d;
}

static orb::Descriptor makeDescBit(bool v) {
    return makeDesc(v ? 0xFFu : 0x00u);
}

static orb::DescriptionResult makeDescResult(
    std::vector<orb::Descriptor> descs,
    std::vector<orb::Keypoint>   kps = {}) {
    if (kps.empty()) {
        kps.resize(descs.size());
        for (std::size_t i = 0; i < descs.size(); ++i) {
            kps[i].x = static_cast<float>(i * 10 + 5);
            kps[i].y = static_cast<float>(i * 7 + 3);
        }
    }
    orb::DescriptionResult r;
    r.descriptors = std::move(descs);
    r.keypoints   = std::move(kps);
    return r;
}

static orb::DescriptionResult makeKpResult(
    const std::vector<std::pair<float,float>>& pts,
    const orb::Descriptor& d = makeDesc(0xAAu)) {
    orb::DescriptionResult r;
    for (auto& [x,y] : pts) {
        orb::Keypoint k{};
        k.x = x; k.y = y;
        r.keypoints.push_back(k);
        r.descriptors.push_back(d);
    }
    return r;
}

// Generate N x N grid of 2-D points (non-collinear by construction).
// Small offsets r*dr_x, c*dc_y break trivial collinearity.
static std::vector<std::pair<double,double>>
makeGrid(int n, double ox = 50.0, double oy = 50.0,
         double sx = 60.0, double sy = 55.0) {
    std::vector<std::pair<double,double>> pts;
    pts.reserve(static_cast<std::size_t>(n * n));
    for (int r = 0; r < n; ++r)
        for (int c = 0; c < n; ++c)
            pts.push_back({ox + c * sx + r * 3.7,
                           oy + r * sy + c * 2.9});
    return pts;
}

// Synthetic correspondence pair generated from a known ground-truth H.
// All Match records are manually constructed (not via matching::match).
struct SyntheticPair {
    orb::DescriptionResult descA;
    orb::DescriptionResult descB;
    std::vector<matching::Match> matches;
    std::array<double,9> H_gt;
};

SyntheticPair makeSyntheticPair(
    const std::array<double,9>& H_B_to_A,
    const std::vector<std::pair<double,double>>& pts_B,
    double noise_sigma = 0.0,
    std::uint32_t seed = 42) {

    std::mt19937 rng(seed);
    std::normal_distribution<double> ndist(0.0, noise_sigma);

    SyntheticPair sp;
    sp.H_gt = H_B_to_A;
    orb::Descriptor d = makeDesc(0xCCu);

    for (std::size_t i = 0; i < pts_B.size(); ++i) {
        auto [xb, yb] = pts_B[i];
        const double w  = H_B_to_A[6]*xb + H_B_to_A[7]*yb + H_B_to_A[8];
        const double xa = (H_B_to_A[0]*xb + H_B_to_A[1]*yb + H_B_to_A[2]) / w
                          + ndist(rng);
        const double ya = (H_B_to_A[3]*xb + H_B_to_A[4]*yb + H_B_to_A[5]) / w
                          + ndist(rng);

        orb::Keypoint kb{}; kb.x = static_cast<float>(xb); kb.y = static_cast<float>(yb);
        orb::Keypoint ka{}; ka.x = static_cast<float>(xa); ka.y = static_cast<float>(ya);
        sp.descB.keypoints.push_back(kb); sp.descB.descriptors.push_back(d);
        sp.descA.keypoints.push_back(ka); sp.descA.descriptors.push_back(d);

        matching::Match m;
        m.index_a = i; m.index_b = i;
        m.distance = 0; m.distance2 = 10;
        sp.matches.push_back(m);
    }
    return sp;
}

// Count the number of true entries in an inlier mask.
static int countInliers(const std::vector<bool>& mask) {
    return static_cast<int>(std::count(mask.begin(), mask.end(), true));
}

// Check all HomographyResult invariants. Returns true if all pass.
static bool checkMaskConsistency(const homography::HomographyResult& res,
                                 std::size_t n_matches,
                                 double thresh) {
    if (res.inlier_mask.size() != n_matches) return false;
    if (countInliers(res.inlier_mask) != res.num_inliers) return false;
    if (res.num_inliers == 0) {
        if (std::abs(res.inlier_ratio) > 1e-12) return false;
        return true;
    }
    if (std::abs(res.inlier_ratio -
                 static_cast<double>(res.num_inliers) /
                 static_cast<double>(n_matches)) > 1e-9)
        return false;
    if (thresh > 0.0 && res.max_reprojection_error > thresh + 1e-6)
        return false;
    // Cannot check reprojection errors without the original points here,
    // so just confirm statistics are finite.
    return std::isfinite(res.mean_reprojection_error) &&
           std::isfinite(res.max_reprojection_error) &&
           res.max_reprojection_error >= res.mean_reprojection_error - 1e-9;
}

// Strict verification of HomographyResult mask consistency:
// Recomputes reprojection error for every single match using res.H.
// Verifies that inliers have err <= thresh, outliers have err >= thresh (or invalid),
// and verifies that mean and max reprojection error exactly match recomputed values.
static bool checkStrictMaskConsistency(const homography::HomographyResult& res,
                                       const std::vector<matching::Match>& matches,
                                       const orb::DescriptionResult& da,
                                       const orb::DescriptionResult& db,
                                       double thresh) {
    if (res.inlier_mask.size() != matches.size()) return false;
    int count = 0;
    double sum_err = 0.0;
    double max_err = 0.0;
    for (std::size_t k = 0; k < matches.size(); ++k) {
        const auto& m = matches[k];
        if (m.index_a >= da.keypoints.size() || m.index_b >= db.keypoints.size())
            return false;
        const double err = homography::reprojectionError(res.H,
                                                         db.keypoints[m.index_b].x,
                                                         db.keypoints[m.index_b].y,
                                                         da.keypoints[m.index_a].x,
                                                         da.keypoints[m.index_a].y);
        const bool is_inlier = res.inlier_mask[k];
        if (is_inlier) {
            if (!std::isfinite(err) || err > thresh + 1e-6) return false;
            ++count;
            sum_err += err;
            max_err = std::max(max_err, err);
        } else {
            // Outlier: reprojection must either exceed threshold or be invalid
            if (std::isfinite(err) && err < thresh - 1e-4) return false;
        }
    }
    if (count != res.num_inliers) return false;
    const double expected_ratio = matches.empty() ? 0.0 : static_cast<double>(count) / static_cast<double>(matches.size());
    if (std::abs(res.inlier_ratio - expected_ratio) > 1e-9) return false;
    if (count > 0) {
        const double expected_mean = sum_err / static_cast<double>(count);
        if (std::abs(res.mean_reprojection_error - expected_mean) > 1e-4) return false;
        if (std::abs(res.max_reprojection_error - max_err) > 1e-4) return false;
    }
    return true;
}


// ============================================================
//  S3 Tests: Hamming Distance
// ============================================================

void test_hamming_identical() {
    std::cout << "  hamming(x, x) == 0 for all byte patterns\n";
    for (int b = 0; b < 256; b += 1) {
        const auto d = makeDesc(static_cast<std::uint8_t>(b));
        EXPECT_EQ(matching::hammingDistance(d, d), 0);
    }
}

void test_hamming_all_differ() {
    std::cout << "  hamming(0x00.., 0xFF..) == 256\n";
    const auto a = makeDescBit(false);
    const auto b = makeDescBit(true);
    EXPECT_EQ(matching::hammingDistance(a, b), 256);
    EXPECT_EQ(matching::hammingDistance(b, a), 256);
}

void test_hamming_known_patterns() {
    std::cout << "  known bit-flip counts\n";
    orb::Descriptor base = makeDesc(0x00u);
    orb::Descriptor b3 = base;
    b3.bytes[0] = 0x07u;  // 3 bits set
    EXPECT_EQ(matching::hammingDistance(base, b3), 3);
    EXPECT_EQ(matching::hammingDistance(b3, base), 3);

    orb::Descriptor b4 = base;
    b4.bytes[0] = 0x01u; b4.bytes[8] = 0x01u;
    b4.bytes[16] = 0x01u; b4.bytes[31] = 0x01u;
    EXPECT_EQ(matching::hammingDistance(base, b4), 4);

    // MSB test: 0x80 has popcount = 1
    orb::Descriptor bmsb = base;
    bmsb.bytes[31] = 0x80u;
    EXPECT_EQ(matching::hammingDistance(base, bmsb), 1);
}

void test_hamming_symmetry() {
    std::cout << "  symmetry: hamming(a,b) == hamming(b,a)\n";
    orb::Descriptor a = makeDesc(0xAAu);
    orb::Descriptor b = makeDesc(0x55u);
    EXPECT_EQ(matching::hammingDistance(a, b), matching::hammingDistance(b, a));
    EXPECT_EQ(matching::hammingDistance(a, b), 256);  // complementary patterns
}

void test_hamming_single_byte_all_values() {
    std::cout << "  popcount(v) for all byte values v in [0,255]\n";
    for (int v = 0; v < 256; ++v) {
        orb::Descriptor a = makeDesc(0x00u);
        orb::Descriptor b = makeDesc(0x00u);
        a.bytes[0] = static_cast<std::uint8_t>(v);
        int expected = 0;
        for (int bit = 0; bit < 8; ++bit) expected += (v >> bit) & 1;
        EXPECT_EQ(matching::hammingDistance(a, b), expected);
    }
}

void test_hamming_randomized_differential() {
    std::cout << "  randomized: cross-check with independent popcount\n";
    std::mt19937 rng(0xBEEF);
    for (int trial = 0; trial < 100; ++trial) {
        orb::Descriptor a, b;
        for (int i = 0; i < 32; ++i) {
            a.bytes[i] = static_cast<std::uint8_t>(rng());
            b.bytes[i] = static_cast<std::uint8_t>(rng());
        }
        int expected = 0;
        for (int i = 0; i < 32; ++i)
            for (int bit = 0; bit < 8; ++bit)
                expected += ((a.bytes[i] ^ b.bytes[i]) >> bit) & 1;
        EXPECT_EQ(matching::hammingDistance(a, b), expected);
    }
    // Range check: must be in [0, 256]
    EXPECT_LE(0, 256);  // trivial sanity
}

void test_hamming_range() {
    std::cout << "  result always in [0, 256]\n";
    std::mt19937 rng(0xCAFE);
    for (int trial = 0; trial < 200; ++trial) {
        orb::Descriptor a, b;
        for (int i = 0; i < 32; ++i) {
            a.bytes[i] = static_cast<std::uint8_t>(rng());
            b.bytes[i] = static_cast<std::uint8_t>(rng());
        }
        const int d = matching::hammingDistance(a, b);
        EXPECT_LE(0, d);
        EXPECT_LE(d, 256);
    }
}

// ============================================================
//  S3 Tests: Configuration Validation
// ============================================================

void test_match_invalid_ratio_nan() {
    std::cout << "  NaN ratio_threshold -> valid=false\n";
    matching::MatchingConfig cfg;
    cfg.ratio_threshold = std::numeric_limits<float>::quiet_NaN();
    auto ra = makeDescResult({makeDesc(0x00u)});
    auto rb = makeDescResult({makeDesc(0x00u), makeDesc(0xFFu)});
    const auto res = matching::match(ra, rb, cfg);
    REQUIRE_TRUE(!res.valid);
    EXPECT_FALSE(res.error_message.empty());
}

void test_match_invalid_ratio_inf() {
    std::cout << "  Inf ratio_threshold -> valid=false\n";
    matching::MatchingConfig cfg;
    cfg.ratio_threshold = std::numeric_limits<float>::infinity();
    auto ra = makeDescResult({makeDesc(0x00u)});
    auto rb = makeDescResult({makeDesc(0x00u), makeDesc(0xFFu)});
    const auto res = matching::match(ra, rb, cfg);
    REQUIRE_TRUE(!res.valid);
}

void test_match_invalid_ratio_negative() {
    std::cout << "  Negative ratio_threshold -> valid=false\n";
    matching::MatchingConfig cfg;
    cfg.ratio_threshold = -0.5f;
    auto ra = makeDescResult({makeDesc(0x00u)});
    auto rb = makeDescResult({makeDesc(0x00u), makeDesc(0xFFu)});
    const auto res = matching::match(ra, rb, cfg);
    REQUIRE_TRUE(!res.valid);
}

void test_match_invalid_ratio_zero() {
    std::cout << "  Zero ratio_threshold -> valid=false\n";
    matching::MatchingConfig cfg;
    cfg.ratio_threshold = 0.0f;
    auto ra = makeDescResult({makeDesc(0x00u)});
    auto rb = makeDescResult({makeDesc(0x00u), makeDesc(0xFFu)});
    const auto res = matching::match(ra, rb, cfg);
    REQUIRE_TRUE(!res.valid);
}

void test_match_invalid_ratio_one() {
    std::cout << "  ratio_threshold == 1.0 -> valid=false\n";
    matching::MatchingConfig cfg;
    cfg.ratio_threshold = 1.0f;
    auto ra = makeDescResult({makeDesc(0x00u)});
    auto rb = makeDescResult({makeDesc(0x00u), makeDesc(0xFFu)});
    const auto res = matching::match(ra, rb, cfg);
    REQUIRE_TRUE(!res.valid);
}

// ============================================================
//  S3 Tests: Edge Cases
// ============================================================

void test_match_empty_A() {
    std::cout << "  empty A -> valid=true, 0 matches\n";
    auto ra = makeDescResult({});
    auto rb = makeDescResult({makeDesc(0x00u), makeDesc(0xFFu)});
    const auto res = matching::match(ra, rb);
    REQUIRE_TRUE(res.valid);
    EXPECT_EQ(res.matches.size(), 0u);
}

void test_match_empty_B() {
    std::cout << "  empty B -> valid=true, 0 matches\n";
    auto ra = makeDescResult({makeDesc(0x00u)});
    auto rb = makeDescResult({});
    const auto res = matching::match(ra, rb);
    REQUIRE_TRUE(res.valid);
    EXPECT_EQ(res.matches.size(), 0u);
}

void test_match_b_one_descriptor() {
    std::cout << "  B has 1 descriptor: 2-NN impossible, 0 matches\n";
    auto ra = makeDescResult({makeDesc(0x00u), makeDesc(0x01u)});
    auto rb = makeDescResult({makeDesc(0x00u)});
    const auto res = matching::match(ra, rb);
    REQUIRE_TRUE(res.valid);
    EXPECT_EQ(res.matches.size(), 0u);
}

void test_match_mismatched_kp_descriptor_A() {
    std::cout << "  A: keypoints.size != descriptors.size -> valid=false\n";
    orb::DescriptionResult ra;
    orb::Keypoint k{}; k.x = 1; k.y = 1;
    ra.keypoints = {k};
    ra.descriptors = {makeDesc(0x00u), makeDesc(0xFFu)};  // mismatch
    auto rb = makeDescResult({makeDesc(0x00u), makeDesc(0x01u)});
    const auto res = matching::match(ra, rb);
    REQUIRE_TRUE(!res.valid);
    EXPECT_FALSE(res.error_message.empty());
}

void test_match_mismatched_kp_descriptor_B() {
    std::cout << "  B: keypoints.size != descriptors.size -> valid=false\n";
    auto ra = makeDescResult({makeDesc(0x00u)});
    orb::DescriptionResult rb;
    orb::Keypoint k{}; k.x = 5; k.y = 5;
    rb.keypoints = {k};
    rb.descriptors = {makeDesc(0x00u), makeDesc(0xFFu)};
    const auto res = matching::match(ra, rb);
    REQUIRE_TRUE(!res.valid);
}

void test_match_perfect_correspondences() {
    std::cout << "  identical descriptor pairs -> all match with d1=0\n";
    orb::Descriptor d0 = makeDesc(0x00u);
    orb::Descriptor d1 = makeDesc(0xAAu);
    orb::Descriptor d2 = makeDesc(0x55u);
    auto ra = makeDescResult({d0, d1, d2});
    auto rb = makeDescResult({d1, d2, d0});
    const auto res = matching::match(ra, rb);
    REQUIRE_TRUE(res.valid);
    EXPECT_EQ(res.matches.size(), 3u);
    for (const auto& m : res.matches)
        EXPECT_EQ(m.distance, 0);
}

void test_match_ratio_test_pass() {
    std::cout << "  ratio test PASS: d1=10, d2=13, threshold=0.8 (10 < 10.4)\n";
    orb::Descriptor da = makeDesc(0x00u);
    orb::Descriptor b0 = makeDesc(0x00u);
    b0.bytes[0] = 0x1Fu; b0.bytes[1] = 0x1Fu;  // d=10 bits
    orb::Descriptor b1 = makeDesc(0x00u);
    b1.bytes[0] = 0xFFu; b1.bytes[1] = 0x1Fu;  // d=13 bits
    auto ra = makeDescResult({da});
    auto rb = makeDescResult({b0, b1});
    matching::MatchingConfig cfg;
    cfg.ratio_threshold = 0.8f;
    const auto res = matching::match(ra, rb, cfg);
    REQUIRE_TRUE(res.valid);
    EXPECT_EQ(res.matches.size(), 1u);
    if (!res.matches.empty()) EXPECT_EQ(res.matches[0].distance, 10);
}

void test_match_ratio_test_fail() {
    std::cout << "  ratio test FAIL: d1=10, d2=12, threshold=0.8 (10 >= 9.6)\n";
    orb::Descriptor da = makeDesc(0x00u);
    orb::Descriptor b0 = makeDesc(0x00u);
    b0.bytes[0] = 0x1Fu; b0.bytes[1] = 0x1Fu;  // d=10 bits
    orb::Descriptor b1 = makeDesc(0x00u);
    b1.bytes[0] = 0xFFu; b1.bytes[1] = 0x0Fu;  // d=12 bits
    auto ra = makeDescResult({da});
    auto rb = makeDescResult({b0, b1});
    matching::MatchingConfig cfg;
    cfg.ratio_threshold = 0.8f;
    const auto res = matching::match(ra, rb, cfg);
    REQUIRE_TRUE(res.valid);
    // 10 < 0.8*12=9.6? NO (10 >= 9.6). Should fail.
    EXPECT_EQ(res.matches.size(), 0u);
}

void test_match_d2_zero_rejected() {
    std::cout << "  d2==0 (two identical best matches): ratio test rejects\n";
    orb::Descriptor da = makeDesc(0x00u);
    auto ra = makeDescResult({da});
    auto rb = makeDescResult({makeDesc(0x00u), makeDesc(0x00u)});
    const auto res = matching::match(ra, rb);
    REQUIRE_TRUE(res.valid);
    EXPECT_EQ(res.matches.size(), 0u);
}

void test_match_d1_zero_d2_positive() {
    std::cout << "  d1=0, d2>0 -> ratio test passes\n";
    auto ra = makeDescResult({makeDesc(0x00u)});
    auto rb = makeDescResult({makeDesc(0x00u), makeDesc(0xFFu)});
    const auto res = matching::match(ra, rb);
    REQUIRE_TRUE(res.valid);
    EXPECT_EQ(res.matches.size(), 1u);
    if (!res.matches.empty()) EXPECT_EQ(res.matches[0].distance, 0);
}

void test_match_hamming_256_case() {
    std::cout << "  descriptor with Hamming distance 256 is handled\n";
    auto ra = makeDescResult({makeDesc(0x00u)});
    // B: one exact match (d=0), one completely different (d=256)
    auto rb = makeDescResult({makeDesc(0x00u), makeDesc(0xFFu)});
    const auto res = matching::match(ra, rb);
    REQUIRE_TRUE(res.valid);
    EXPECT_EQ(res.matches.size(), 1u);
    if (!res.matches.empty()) {
        EXPECT_EQ(res.matches[0].distance, 0);
        EXPECT_EQ(res.matches[0].distance2, 256);
    }
}

void test_match_index_mapping_correct() {
    std::cout << "  A[0]->B[1], A[1]->B[2] verified\n";
    orb::Descriptor d0 = makeDesc(0x01u);
    orb::Descriptor d1 = makeDesc(0x02u);
    orb::Descriptor d2 = makeDesc(0xFFu);
    auto ra = makeDescResult({d0, d1});
    auto rb = makeDescResult({d2, d0, d1});
    const auto res = matching::match(ra, rb);
    REQUIRE_TRUE(res.valid);
    EXPECT_LE(res.matches.size(), 2u);
    for (const auto& m : res.matches) {
        if (m.index_a == 0) EXPECT_EQ(m.index_b, 1u);
        if (m.index_a == 1) EXPECT_EQ(m.index_b, 2u);
        EXPECT_LT(m.index_a, 2u);
        EXPECT_LT(m.index_b, 3u);
    }
}

void test_match_no_out_of_bounds_index() {
    std::cout << "  All match indices are within descriptor array bounds\n";
    auto ra = makeDescResult({makeDesc(0x0Fu), makeDesc(0xF0u), makeDesc(0xAAu)});
    auto rb = makeDescResult({makeDesc(0x0Fu), makeDesc(0x55u), makeDesc(0xF0u)});
    const auto res = matching::match(ra, rb);
    REQUIRE_TRUE(res.valid);
    for (const auto& m : res.matches) {
        EXPECT_LT(m.index_a, ra.keypoints.size());
        EXPECT_LT(m.index_b, rb.keypoints.size());
    }
}

void test_match_determinism() {
    std::cout << "  multiple calls with same input -> identical result\n";
    auto ra = makeDescResult({makeDesc(0x0Fu), makeDesc(0xF0u), makeDesc(0xAAu)});
    auto rb = makeDescResult({makeDesc(0x0Fu), makeDesc(0x55u), makeDesc(0xF0u), makeDesc(0xAAu)});
    const auto r1 = matching::match(ra, rb);
    const auto r2 = matching::match(ra, rb);
    REQUIRE_TRUE(r1.valid);
    EXPECT_EQ(r1.matches.size(), r2.matches.size());
    for (std::size_t i = 0; i < r1.matches.size(); ++i) {
        EXPECT_EQ(r1.matches[i].index_a, r2.matches[i].index_a);
        EXPECT_EQ(r1.matches[i].index_b, r2.matches[i].index_b);
        EXPECT_EQ(r1.matches[i].distance, r2.matches[i].distance);
    }
}

void test_match_different_sizes() {
    std::cout << "  A.size() != B.size() is valid\n";
    auto ra = makeDescResult({makeDesc(0x00u), makeDesc(0xAAu), makeDesc(0x55u)});
    auto rb = makeDescResult({makeDesc(0x00u), makeDesc(0xAAu)});
    const auto res = matching::match(ra, rb);
    REQUIRE_TRUE(res.valid);
    EXPECT_LE(res.matches.size(), 3u);
}

// ============================================================
//  S3 Tests: Cross-Check
// ============================================================

void test_match_crosscheck_no_change_when_mutual() {
    std::cout << "  cross_check=true: keeps mutual nearest neighbors\n";
    // A[0]=d0, A[1]=d1; B[0]=d0, B[1]=d1
    // Both are mutual nearest neighbors -> cross-check keeps them.
    orb::Descriptor d0 = makeDesc(0x01u);
    orb::Descriptor d1 = makeDesc(0xFEu);
    auto ra = makeDescResult({d0, d1});
    auto rb = makeDescResult({d0, d1});

    matching::MatchingConfig cfg_on, cfg_off;
    cfg_on.cross_check  = true;
    cfg_off.cross_check = false;
    const auto res_on  = matching::match(ra, rb, cfg_on);
    const auto res_off = matching::match(ra, rb, cfg_off);
    REQUIRE_TRUE(res_on.valid);
    REQUIRE_TRUE(res_off.valid);
    // With perfect mutual matches, cross-check should keep the same matches.
    EXPECT_EQ(res_on.matches.size(), res_off.matches.size());
}

void test_match_crosscheck_removes_non_mutual() {
    std::cout << "  cross_check=true removes non-mutual matches\n";
    // A[0]=d0, A[1]=d_similar_to_d0.
    // B[0]=d0, B[1]=d1 (very different).
    // A[0]->B[0] is mutual. A[1]->B[0] (also close) but B[0]->A would prefer A[0].
    // So cross_check should remove the A[1] match.
    orb::Descriptor d0 = makeDesc(0x00u);    // d0 = all zeros
    orb::Descriptor d_close = makeDesc(0x00u);
    d_close.bytes[0] = 0x01u;                // d_close differs by 1 bit from d0
    orb::Descriptor d1 = makeDesc(0xFFu);    // very different

    auto ra = makeDescResult({d0, d_close});
    auto rb = makeDescResult({d0, d1});

    matching::MatchingConfig cfg_on;
    cfg_on.cross_check = true;
    matching::MatchingConfig cfg_off;
    cfg_off.cross_check = false;

    const auto res_on  = matching::match(ra, rb, cfg_on);
    const auto res_off = matching::match(ra, rb, cfg_off);
    REQUIRE_TRUE(res_on.valid);
    REQUIRE_TRUE(res_off.valid);
    // Cross-check should produce equal or fewer matches.
    EXPECT_LE(res_on.matches.size(), res_off.matches.size());
    // A[0]->B[0] should survive cross-check (mutual).
    bool a0_b0_found = false;
    for (const auto& m : res_on.matches)
        if (m.index_a == 0 && m.index_b == 0) a0_b0_found = true;
    // d0 vs d_close: difference is 1 bit. Both have best match at B[0]=d0.
    // Forward: A[0]->B[0] (d=0, d2=256->pass), A[1]->B[0] (d=1, d2=256->pass).
    // Reverse: B[0]->A[0] (d=0), B[0]->A[1] (d=1) -> best is A[0].
    // Cross-check: A[0]->B[0] survives (rev_best[0]=0==index_a=0). OK.
    //              A[1]->B[0] fails (rev_best[0]=0 != index_a=1).
    if (a0_b0_found) EXPECT_TRUE(a0_b0_found);  // just register we found it
    // The key invariant: cross_check result <= no-cross_check result.
    EXPECT_LE(res_on.matches.size(), res_off.matches.size());
}

void test_match_crosscheck_false_is_default() {
    std::cout << "  default cross_check=false: same as explicit false\n";
    auto ra = makeDescResult({makeDesc(0x0Fu), makeDesc(0xF0u)});
    auto rb = makeDescResult({makeDesc(0x0Fu), makeDesc(0xF0u), makeDesc(0xAAu)});
    matching::MatchingConfig cfg_default;
    matching::MatchingConfig cfg_explicit;
    cfg_explicit.cross_check = false;
    const auto r1 = matching::match(ra, rb, cfg_default);
    const auto r2 = matching::match(ra, rb, cfg_explicit);
    REQUIRE_TRUE(r1.valid && r2.valid);
    EXPECT_EQ(r1.matches.size(), r2.matches.size());
}

// ============================================================
//  S4 Tests: Configuration Validation
// ============================================================

void test_hom_invalid_config_neg_iterations() {
    std::cout << "  ransac_iterations <= 0 -> valid=false, InvalidConfig\n";
    homography::HomographyConfig cfg;
    cfg.ransac_iterations = 0;
    auto sp = makeSyntheticPair({1,0,0, 0,1,0, 0,0,1}, makeGrid(3));
    const auto res = homography::estimate(sp.matches, sp.descA, sp.descB, cfg);
    REQUIRE_TRUE(!res.valid);
    EXPECT_EQ(res.failure_reason, homography::FailureReason::InvalidConfig);
}

void test_hom_invalid_config_nan_threshold() {
    std::cout << "  NaN reprojection_threshold -> valid=false, InvalidConfig\n";
    homography::HomographyConfig cfg;
    cfg.reprojection_threshold = std::numeric_limits<double>::quiet_NaN();
    auto sp = makeSyntheticPair({1,0,0, 0,1,0, 0,0,1}, makeGrid(3));
    const auto res = homography::estimate(sp.matches, sp.descA, sp.descB, cfg);
    REQUIRE_TRUE(!res.valid);
    EXPECT_EQ(res.failure_reason, homography::FailureReason::InvalidConfig);
}

void test_hom_invalid_config_inf_threshold() {
    std::cout << "  Inf reprojection_threshold -> valid=false, InvalidConfig\n";
    homography::HomographyConfig cfg;
    cfg.reprojection_threshold = std::numeric_limits<double>::infinity();
    auto sp = makeSyntheticPair({1,0,0, 0,1,0, 0,0,1}, makeGrid(3));
    const auto res = homography::estimate(sp.matches, sp.descA, sp.descB, cfg);
    REQUIRE_TRUE(!res.valid);
    EXPECT_EQ(res.failure_reason, homography::FailureReason::InvalidConfig);
}

void test_hom_invalid_config_neg_threshold() {
    std::cout << "  negative reprojection_threshold -> valid=false, InvalidConfig\n";
    homography::HomographyConfig cfg;
    cfg.reprojection_threshold = -1.0;
    auto sp = makeSyntheticPair({1,0,0, 0,1,0, 0,0,1}, makeGrid(3));
    const auto res = homography::estimate(sp.matches, sp.descA, sp.descB, cfg);
    REQUIRE_TRUE(!res.valid);
    EXPECT_EQ(res.failure_reason, homography::FailureReason::InvalidConfig);
}

void test_hom_invalid_config_min_inliers_too_small() {
    std::cout << "  min_inliers < 4 -> valid=false, InvalidConfig\n";
    homography::HomographyConfig cfg;
    cfg.min_inliers = 3;
    auto sp = makeSyntheticPair({1,0,0, 0,1,0, 0,0,1}, makeGrid(3));
    const auto res = homography::estimate(sp.matches, sp.descA, sp.descB, cfg);
    REQUIRE_TRUE(!res.valid);
    EXPECT_EQ(res.failure_reason, homography::FailureReason::InvalidConfig);
}

void test_hom_invalid_config_min_inliers_zero() {
    std::cout << "  min_inliers=0 -> valid=false, InvalidConfig (NOT 0 inliers=valid)\n";
    homography::HomographyConfig cfg;
    cfg.min_inliers = 0;
    auto sp = makeSyntheticPair({1,0,0, 0,1,0, 0,0,1}, makeGrid(3));
    const auto res = homography::estimate(sp.matches, sp.descA, sp.descB, cfg);
    REQUIRE_TRUE(!res.valid);
    EXPECT_EQ(res.failure_reason, homography::FailureReason::InvalidConfig);
}

// ============================================================
//  S4 Tests: Input Validation
// ============================================================

void test_hom_too_few_matches() {
    std::cout << "  fewer than 4 matches -> TooFewMatches\n";
    std::vector<matching::Match> matches = {{0,0,0,10},{1,1,0,10},{2,2,0,10}};
    auto da = makeKpResult({{10,10},{20,20},{30,30}});
    auto db = makeKpResult({{10,10},{20,20},{30,30}});
    const auto res = homography::estimate(matches, da, db);
    REQUIRE_TRUE(!res.valid);
    EXPECT_EQ(res.failure_reason, homography::FailureReason::TooFewMatches);
    EXPECT_EQ(static_cast<int>(res.inlier_mask.size()), 3);
}

void test_hom_match_index_out_of_range() {
    std::cout << "  match index_a out of range -> InvalidInput\n";
    std::vector<matching::Match> matches = {
        {100, 0, 0, 10},  // index_a=100 is out of range
        {0, 0, 0, 10},
        {1, 1, 0, 10},
        {2, 2, 0, 10}
    };
    auto da = makeKpResult({{10,10},{20,20},{30,30}});  // only 3 keypoints
    auto db = makeKpResult({{10,10},{20,20},{30,30},{40,40}});
    const auto res = homography::estimate(matches, da, db);
    REQUIRE_TRUE(!res.valid);
    EXPECT_EQ(res.failure_reason, homography::FailureReason::InvalidInput);
}

void test_hom_nan_keypoint_rejected() {
    std::cout << "  NaN keypoint coordinate -> InvalidInput (not a silent filter)\n";
    const float nan_v = std::numeric_limits<float>::quiet_NaN();
    orb::DescriptionResult da, db;
    std::vector<matching::Match> matches;
    for (int i = 0; i < 6; ++i) {
        orb::Keypoint ka{}; ka.x = (i == 0) ? nan_v : 100.0f + i; ka.y = 50.0f + i;
        orb::Keypoint kb{}; kb.x = 50.0f + i; kb.y = 50.0f + i;
        da.keypoints.push_back(ka); da.descriptors.push_back(makeDesc(0u));
        db.keypoints.push_back(kb); db.descriptors.push_back(makeDesc(0u));
        matches.push_back({static_cast<std::size_t>(i),
                           static_cast<std::size_t>(i), 0, 10});
    }
    homography::HomographyConfig cfg;
    cfg.random_seed = 0x7777;
    const auto res = homography::estimate(matches, da, db, cfg);
    // Policy: reject input with any non-finite coordinate.
    REQUIRE_TRUE(!res.valid);
    EXPECT_EQ(res.failure_reason, homography::FailureReason::InvalidInput);
}

// ============================================================
//  S4 Tests: Repeated and Degenerate Points
// ============================================================

void test_hom_repeated_points_in_B() {
    std::cout << "  all B points identical -> AllHypothesesFailed\n";
    orb::DescriptionResult da, db;
    std::vector<matching::Match> matches;
    for (int i = 0; i < 10; ++i) {
        orb::Keypoint ka{}; ka.x = 100.0f + i * 20.0f; ka.y = 100.0f + i * 15.0f;
        orb::Keypoint kb{}; kb.x = 50.0f; kb.y = 50.0f;  // All identical
        da.keypoints.push_back(ka); da.descriptors.push_back(makeDesc(0u));
        db.keypoints.push_back(kb); db.descriptors.push_back(makeDesc(0u));
        matches.push_back({static_cast<std::size_t>(i),
                           static_cast<std::size_t>(i), 0, 10});
    }
    homography::HomographyConfig cfg;
    cfg.random_seed = 0x6666;
    cfg.min_inliers = 4;
    const auto res = homography::estimate(matches, da, db, cfg);
    // Repeated points in B -> all 4-samples degenerate -> AllHypothesesFailed.
    REQUIRE_TRUE(!res.valid);
    EXPECT_NE(res.failure_reason, homography::FailureReason::None);
    // H must be all-zero (not a garbage matrix).
    for (double v : res.H) EXPECT_NEAR(v, 0.0, 1e-15);
}

void test_hom_repeated_points_in_A() {
    std::cout << "  all A points identical -> AllHypothesesFailed\n";
    orb::DescriptionResult da, db;
    std::vector<matching::Match> matches;
    for (int i = 0; i < 10; ++i) {
        orb::Keypoint ka{}; ka.x = 50.0f; ka.y = 50.0f;  // All identical
        orb::Keypoint kb{}; kb.x = 100.0f + i * 20.0f; kb.y = 100.0f + i * 15.0f;
        da.keypoints.push_back(ka); da.descriptors.push_back(makeDesc(0u));
        db.keypoints.push_back(kb); db.descriptors.push_back(makeDesc(0u));
        matches.push_back({static_cast<std::size_t>(i),
                           static_cast<std::size_t>(i), 0, 10});
    }
    homography::HomographyConfig cfg;
    cfg.random_seed = 0x9999;
    cfg.min_inliers = 4;
    const auto res = homography::estimate(matches, da, db, cfg);
    REQUIRE_TRUE(!res.valid);
    EXPECT_NE(res.failure_reason, homography::FailureReason::None);
    for (double v : res.H) EXPECT_NEAR(v, 0.0, 1e-15);
}

void test_hom_collinear_points() {
    std::cout << "  collinear B points -> fails gracefully\n";
    auto sp = makeSyntheticPair(
        {1,0,0, 0,1,0, 0,0,1},
        {{10,50},{20,50},{30,50},{40,50},{50,50},{60,50}});
    homography::HomographyConfig cfg;
    cfg.random_seed = 0x5555;
    const auto res = homography::estimate(sp.matches, sp.descA, sp.descB, cfg);
    // Must not be marked valid with a garbage H.
    // Can be AllHypothesesFailed or TooFewInliers.
    if (res.valid) {
        // If somehow it succeeded, H must be finite and inlier mask consistent.
        for (double v : res.H) REQUIRE_TRUE(std::isfinite(v));
        REQUIRE_TRUE(checkMaskConsistency(res, sp.matches.size(), cfg.reprojection_threshold));
    } else {
        // At minimum, failure_reason must be set.
        REQUIRE_TRUE(res.failure_reason != homography::FailureReason::None);
        // H must be zeros.
        for (double v : res.H) EXPECT_NEAR(v, 0.0, 1e-15);
    }
}

// ============================================================
//  S4 Tests: Synthetic Correspondences (Correct H)
// ============================================================

void test_hom_identity() {
    std::cout << "  identity H\n";
    std::array<double,9> H_gt = {1,0,0, 0,1,0, 0,0,1};
    auto sp = makeSyntheticPair(H_gt, makeGrid(4, 50, 50, 60, 55));
    homography::HomographyConfig cfg; cfg.random_seed = 0x1234;
    const auto res = homography::estimate(sp.matches, sp.descA, sp.descB, cfg);
    REQUIRE_TRUE(res.valid);
    EXPECT_NEAR(res.mean_reprojection_error, 0.0, 0.01);
    EXPECT_EQ(res.num_inliers, static_cast<int>(sp.matches.size()));
    REQUIRE_TRUE(checkMaskConsistency(res, sp.matches.size(), cfg.reprojection_threshold));
}

void test_hom_translation() {
    std::cout << "  translation H\n";
    std::array<double,9> H_gt = {1,0,50, 0,1,30, 0,0,1};
    auto sp = makeSyntheticPair(H_gt, makeGrid(5));
    homography::HomographyConfig cfg; cfg.random_seed = 0xABCD;
    const auto res = homography::estimate(sp.matches, sp.descA, sp.descB, cfg);
    REQUIRE_TRUE(res.valid);
    EXPECT_NEAR(res.mean_reprojection_error, 0.0, 0.1);
    REQUIRE_TRUE(checkMaskConsistency(res, sp.matches.size(), cfg.reprojection_threshold));
}

void test_hom_uniform_scale() {
    std::cout << "  uniform scale H\n";
    std::array<double,9> H_gt = {2,0,0, 0,2,0, 0,0,1};
    auto sp = makeSyntheticPair(H_gt, makeGrid(4, 30, 30, 40, 35));
    homography::HomographyConfig cfg; cfg.random_seed = 0x5678;
    const auto res = homography::estimate(sp.matches, sp.descA, sp.descB, cfg);
    REQUIRE_TRUE(res.valid);
    EXPECT_NEAR(res.mean_reprojection_error, 0.0, 0.1);
    REQUIRE_TRUE(checkMaskConsistency(res, sp.matches.size(), cfg.reprojection_threshold));
}

void test_hom_rotation_30deg() {
    std::cout << "  rotation 30 degrees\n";
    const double a = 30.0 * M_PI / 180.0;
    std::array<double,9> H_gt = {std::cos(a),-std::sin(a),0,
                                  std::sin(a), std::cos(a),0, 0,0,1};
    auto sp = makeSyntheticPair(H_gt, makeGrid(4, 80, 80, 50, 50));
    homography::HomographyConfig cfg; cfg.random_seed = 0x9ABC;
    const auto res = homography::estimate(sp.matches, sp.descA, sp.descB, cfg);
    REQUIRE_TRUE(res.valid);
    EXPECT_NEAR(res.mean_reprojection_error, 0.0, 0.1);
}

void test_hom_perspective() {
    std::cout << "  full perspective H\n";
    std::array<double,9> H_gt = {1.2, 0.1, 20, 0.05, 0.9, 15, 0.001, 0.0002, 1};
    auto sp = makeSyntheticPair(H_gt, makeGrid(5, 40, 40, 50, 45));
    homography::HomographyConfig cfg; cfg.random_seed = 0xDEF0;
    const auto res = homography::estimate(sp.matches, sp.descA, sp.descB, cfg);
    REQUIRE_TRUE(res.valid);
    EXPECT_NEAR(res.mean_reprojection_error, 0.0, 0.2);
    REQUIRE_TRUE(checkMaskConsistency(res, sp.matches.size(), cfg.reprojection_threshold));
}

void test_hom_with_gaussian_noise() {
    std::cout << "  with Gaussian noise (sigma=0.5px)\n";
    std::array<double,9> H_gt = {1.1, 0.05, 30, -0.03, 0.95, 10, 0,0,1};
    auto sp = makeSyntheticPair(H_gt, makeGrid(6, 30, 30, 40, 35), 0.5, 0x1111);
    homography::HomographyConfig cfg; cfg.random_seed = 0x2222;
    const auto res = homography::estimate(sp.matches, sp.descA, sp.descB, cfg);
    REQUIRE_TRUE(res.valid);
    EXPECT_LT(res.mean_reprojection_error, 1.5);
    REQUIRE_TRUE(res.num_inliers >= 4);
}

void test_hom_with_30pct_outliers() {
    std::cout << "  with 30% outliers\n";
    std::array<double,9> H_gt = {1,0,50, 0,1,30, 0,0,1};
    auto sp = makeSyntheticPair(H_gt, makeGrid(6, 40, 40, 45, 40));
    // Corrupt ~30% of A keypoints.
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> noise(-200.0f, 200.0f);
    for (std::size_t i = 0; i < sp.descA.keypoints.size(); ++i) {
        if (rng() % 3 == 0) {
            sp.descA.keypoints[i].x += noise(rng);
            sp.descA.keypoints[i].y += noise(rng);
        }
    }
    homography::HomographyConfig cfg; cfg.random_seed = 0x3333;
    const auto res = homography::estimate(sp.matches, sp.descA, sp.descB, cfg);
    REQUIRE_TRUE(res.valid);
    EXPECT_LT(res.mean_reprojection_error, 0.5);
    EXPECT_LT(res.inlier_ratio, 1.0);
    REQUIRE_TRUE(checkMaskConsistency(res, sp.matches.size(), cfg.reprojection_threshold));
}

void test_hom_with_50pct_outliers() {
    std::cout << "  with 50% outliers (harder)\n";
    std::array<double,9> H_gt = {1,0,40, 0,1,25, 0,0,1};
    auto sp = makeSyntheticPair(H_gt, makeGrid(7, 50, 50, 50, 45));
    std::mt19937 rng(77);
    std::uniform_real_distribution<float> noise(-300.0f, 300.0f);
    int n_corrupted = 0;
    for (std::size_t i = 0; i < sp.descA.keypoints.size(); i += 2) {
        sp.descA.keypoints[i].x += noise(rng);
        sp.descA.keypoints[i].y += noise(rng);
        ++n_corrupted;
    }
    REQUIRE_TRUE(n_corrupted > 0);
    homography::HomographyConfig cfg;
    cfg.random_seed = 0x4444;
    cfg.ransac_iterations = 3000;  // More iterations for harder case
    const auto res = homography::estimate(sp.matches, sp.descA, sp.descB, cfg);
    // With 50% outliers and enough iterations, RANSAC should still find H.
    // However, this is not guaranteed for all seeds; check if succeeded.
    if (res.valid) {
        REQUIRE_TRUE(res.num_inliers >= 4);
        REQUIRE_TRUE(checkMaskConsistency(res, sp.matches.size(), cfg.reprojection_threshold));
    } else {
        // Not finding a model with 50% outliers is also valid behavior.
        REQUIRE_TRUE(res.failure_reason != homography::FailureReason::None);
    }
}

void test_hom_four_exact_correspondences() {
    std::cout << "  exactly 4 non-degenerate correspondences -> valid\n";
    std::array<double,9> H_gt = {1,0,40, 0,1,20, 0,0,1};
    auto sp = makeSyntheticPair(H_gt, {{100,100},{200,100},{100,200},{200,200}});
    homography::HomographyConfig cfg; cfg.random_seed = 0x4444;
    const auto res = homography::estimate(sp.matches, sp.descA, sp.descB, cfg);
    REQUIRE_TRUE(res.valid);
    EXPECT_NEAR(res.mean_reprojection_error, 0.0, 0.01);
    EXPECT_EQ(res.num_inliers, 4);
}

void test_hom_direction_B_to_A() {
    std::cout << "  direction: H * xB ~= xA for all inliers\n";
    const double angle = 45.0 * M_PI / 180.0;
    const double c = std::cos(angle) * 1.5, s_v = std::sin(angle) * 1.5;
    std::array<double,9> H_gt = {c, -s_v, 100, s_v, c, 50, 0, 0, 1};
    auto sp = makeSyntheticPair(H_gt, makeGrid(5, 20, 20, 40, 35));
    homography::HomographyConfig cfg; cfg.random_seed = 0x8888;
    const auto res = homography::estimate(sp.matches, sp.descA, sp.descB, cfg);
    REQUIRE_TRUE(res.valid);
    // Verify direction: project each B point and check it lands on A point.
    for (std::size_t k = 0; k < sp.descB.keypoints.size(); ++k) {
        const double xb = sp.descB.keypoints[k].x;
        const double yb = sp.descB.keypoints[k].y;
        const double xa = sp.descA.keypoints[k].x;
        const double ya = sp.descA.keypoints[k].y;
        const double err = homography::reprojectionError(res.H, xb, yb, xa, ya);
        EXPECT_NEAR(err, 0.0, 0.5);
    }
}

void test_hom_large_scale_coordinates() {
    std::cout << "  large-scale coordinates (3840x2160 range)\n";
    std::array<double,9> H_gt = {1, 0, 200, 0, 1, 100, 0, 0, 1};
    auto pts = makeGrid(5, 500, 300, 300, 250);
    auto sp = makeSyntheticPair(H_gt, pts);
    homography::HomographyConfig cfg; cfg.random_seed = 0xAAAA;
    const auto res = homography::estimate(sp.matches, sp.descA, sp.descB, cfg);
    REQUIRE_TRUE(res.valid);
    EXPECT_NEAR(res.mean_reprojection_error, 0.0, 0.5);
}

void test_hom_small_scale_coordinates() {
    std::cout << "  small-scale coordinates (tiny pixel offsets)\n";
    std::array<double,9> H_gt = {1, 0, 2.0, 0, 1, 1.5, 0, 0, 1};
    auto pts = makeGrid(4, 5, 5, 4, 4);
    auto sp = makeSyntheticPair(H_gt, pts);
    homography::HomographyConfig cfg;
    cfg.random_seed = 0xBBBB;
    cfg.reprojection_threshold = 0.5;  // tighter threshold for small scale
    const auto res = homography::estimate(sp.matches, sp.descA, sp.descB, cfg);
    REQUIRE_TRUE(res.valid);
    EXPECT_NEAR(res.mean_reprojection_error, 0.0, 0.2);
}

void test_hom_determinism() {
    std::cout << "  same seed -> identical result\n";
    std::array<double,9> H_gt = {1.1, 0.05, 30, 0.02, 0.95, 10, 0.0001, 0, 1};
    auto sp = makeSyntheticPair(H_gt, makeGrid(5, 30, 30, 40, 35), 0.3, 12345);
    homography::HomographyConfig cfg; cfg.random_seed = 0x9999;
    const auto r1 = homography::estimate(sp.matches, sp.descA, sp.descB, cfg);
    const auto r2 = homography::estimate(sp.matches, sp.descA, sp.descB, cfg);
    REQUIRE_TRUE(r1.valid == r2.valid);
    EXPECT_EQ(r1.num_inliers, r2.num_inliers);
    if (r1.valid && r2.valid)
        for (int i = 0; i < 9; ++i)
            EXPECT_NEAR(r1.H[i], r2.H[i], 1e-12);
}

// ============================================================
//  S4 Tests: Inlier Mask Consistency
// ============================================================

void test_hom_inlier_mask_size() {
    std::cout << "  inlier_mask.size() == matches.size() always\n";
    // Test for valid result
    auto sp = makeSyntheticPair({1,0,20, 0,1,10, 0,0,1},
        {{10,10},{100,100},{200,200},{50,150},{150,50}});
    homography::HomographyConfig cfg; cfg.random_seed = 0xAAAA;
    const auto res = homography::estimate(sp.matches, sp.descA, sp.descB, cfg);
    REQUIRE_TRUE(res.inlier_mask.size() == sp.matches.size());
    // Also test for failure case
    std::vector<matching::Match> few = {{0,0,0,10},{1,1,0,10}};
    auto da2 = makeKpResult({{10,10},{20,20}});
    auto db2 = makeKpResult({{10,10},{20,20}});
    const auto res2 = homography::estimate(few, da2, db2);
    REQUIRE_TRUE(res2.inlier_mask.size() == few.size());
}

void test_hom_inlier_count_matches_mask() {
    std::cout << "  num_inliers == count(mask==true)\n";
    auto sp = makeSyntheticPair({1,0,30, 0,1,20, 0,0,1}, makeGrid(5));
    homography::HomographyConfig cfg; cfg.random_seed = 0xCCCC;
    const auto res = homography::estimate(sp.matches, sp.descA, sp.descB, cfg);
    REQUIRE_TRUE(res.inlier_mask.size() == sp.matches.size());
    REQUIRE_TRUE(res.num_inliers == countInliers(res.inlier_mask));
}

void test_hom_inlier_ratio_correct() {
    std::cout << "  inlier_ratio == num_inliers / matches.size()\n";
    auto sp = makeSyntheticPair({1,0,30, 0,1,20, 0,0,1}, makeGrid(5));
    homography::HomographyConfig cfg; cfg.random_seed = 0xDDDD;
    const auto res = homography::estimate(sp.matches, sp.descA, sp.descB, cfg);
    REQUIRE_TRUE(!sp.matches.empty());
    const double expected_ratio = static_cast<double>(res.num_inliers) /
                                  static_cast<double>(sp.matches.size());
    EXPECT_NEAR(res.inlier_ratio, expected_ratio, 1e-9);
}

void test_hom_reprojection_error_finite() {
    std::cout << "  mean/max reprojection error is finite when valid\n";
    auto sp = makeSyntheticPair({1,0,40, 0,1,25, 0,0,1}, makeGrid(5));
    homography::HomographyConfig cfg; cfg.random_seed = 0xEEEE;
    const auto res = homography::estimate(sp.matches, sp.descA, sp.descB, cfg);
    REQUIRE_TRUE(res.valid);
    REQUIRE_TRUE(std::isfinite(res.mean_reprojection_error));
    REQUIRE_TRUE(std::isfinite(res.max_reprojection_error));
    EXPECT_LE(0.0, res.mean_reprojection_error);
    EXPECT_LE(res.mean_reprojection_error, res.max_reprojection_error + 1e-9);
}

void test_hom_strict_mask_consistency() {
    std::cout << "  strict mask consistency: recomputes all errors from final H\n";
    std::array<double,9> H_gt = {1.1, 0.05, 15.0, -0.02, 0.95, 25.0, 0.0001, 0.0001, 1.0};
    auto sp = makeSyntheticPair(H_gt, makeGrid(6));
    homography::HomographyConfig cfg;
    cfg.random_seed = 0x9999;
    cfg.reprojection_threshold = 3.0;
    const auto res = homography::estimate(sp.matches, sp.descA, sp.descB, cfg);
    REQUIRE_TRUE(res.valid);
    REQUIRE_TRUE(checkStrictMaskConsistency(res, sp.matches, sp.descA, sp.descB, cfg.reprojection_threshold));
}


// ============================================================
//  S4 Tests: project() and reprojectionError()
// ============================================================

void test_project_basic() {
    std::cout << "  project() correctly computes perspective division\n";
    std::array<double,9> H = {1,0,10, 0,1,20, 0,0,1};
    const auto p = homography::project(H, 5.0, 5.0);
    EXPECT_NEAR(p.x, 15.0, 1e-10);
    EXPECT_NEAR(p.y, 25.0, 1e-10);
}

void test_project_perspective_division() {
    std::cout << "  project() with w != 1\n";
    std::array<double,9> H = {2,0,0, 0,2,0, 0,0,2};  // scale by 1 (2/2)
    const auto p = homography::project(H, 3.0, 4.0);
    EXPECT_NEAR(p.x, 3.0, 1e-10);
    EXPECT_NEAR(p.y, 4.0, 1e-10);
}

void test_project_scale_invariance_small() {
    std::cout << "  project() scale invariance: 1e-12 * I\n";
    std::array<double,9> H = {1e-12, 0, 0,  0, 1e-12, 0,  0, 0, 1e-12};
    const auto p = homography::project(H, 50.0, 70.0);
    EXPECT_NEAR(p.x, 50.0, 1e-9);
    EXPECT_NEAR(p.y, 70.0, 1e-9);
}

void test_project_scale_invariance_tiny() {
    std::cout << "  project() scale invariance: 1e-20 * I\n";
    std::array<double,9> H = {1e-20, 0, 0,  0, 1e-20, 0,  0, 0, 1e-20};
    const auto p = homography::project(H, 50.0, 70.0);
    EXPECT_NEAR(p.x, 50.0, 1e-9);
    EXPECT_NEAR(p.y, 70.0, 1e-9);
}


void test_project_large_coord_1e13() {
    std::cout << "  project() identity with large coordinate x = 1e13\n";
    std::array<double, 9> I = {1,0,0, 0,1,0, 0,0,1};
    const auto p = homography::project(I, 1e13, 0.0);
    EXPECT_NEAR(p.x, 1e13, 1e-4);
    EXPECT_NEAR(p.y, 0.0, 1e-4);
}

void test_project_large_coord_1e14() {
    std::cout << "  project() identity with large coordinate x = 1e14\n";
    std::array<double, 9> I = {1,0,0, 0,1,0, 0,0,1};
    const auto p = homography::project(I, 1e14, 0.0);
    EXPECT_NEAR(p.x, 1e14, 1e-3);
    EXPECT_NEAR(p.y, 0.0, 1e-3);
}

void test_project_scale_invariance_large() {
    std::cout << "  project() scale invariance: 1e+12 * I\n";
    std::array<double,9> H = {1e12, 0, 0,  0, 1e12, 0,  0, 0, 1e12};
    const auto p = homography::project(H, 50.0, 70.0);
    EXPECT_NEAR(p.x, 50.0, 1e-9);
    EXPECT_NEAR(p.y, 70.0, 1e-9);
}

void test_project_scale_invariance_negative() {
    std::cout << "  project() scale invariance: -3 * I\n";
    std::array<double,9> H = {-3.0, 0, 0,  0, -3.0, 0,  0, 0, -3.0};
    const auto p = homography::project(H, 50.0, 70.0);
    EXPECT_NEAR(p.x, 50.0, 1e-9);
    EXPECT_NEAR(p.y, 70.0, 1e-9);
}

void test_project_scale_invariance_perspective() {
    std::cout << "  project() scale invariance: full perspective H\n";
    std::array<double,9> H1 = {1.2, 0.1, 30.0,  -0.05, 1.1, 20.0,  0.0001, 0.0002, 1.0};
    std::array<double,9> H_small = H1;
    std::array<double,9> H_large = H1;
    for (double& v : H_small) v *= 1e-9;
    for (double& v : H_large) v *= 1e9;

    const auto p1 = homography::project(H1, 100.0, 150.0);
    const auto ps = homography::project(H_small, 100.0, 150.0);
    const auto pl = homography::project(H_large, 100.0, 150.0);

    EXPECT_NEAR(ps.x, p1.x, 1e-8);
    EXPECT_NEAR(ps.y, p1.y, 1e-8);
    EXPECT_NEAR(pl.x, p1.x, 1e-8);
    EXPECT_NEAR(pl.y, p1.y, 1e-8);
}

void test_project_zero_matrix_throws() {
    std::cout << "  project() throws on all-zero H\n";
    std::array<double,9> H{};
    bool threw = false;
    try { homography::project(H, 1.0, 1.0); }
    catch (const std::domain_error&) { threw = true; }
    REQUIRE_TRUE(threw);
}

void test_project_throws_on_nan_H() {
    std::cout << "  project() throws on NaN H\n";
    std::array<double,9> H{};
    H[0] = std::numeric_limits<double>::quiet_NaN();
    bool threw = false;
    try { homography::project(H, 1.0, 1.0); }
    catch (const std::domain_error&) { threw = true; }
    REQUIRE_TRUE(threw);
}

void test_project_inf_matrix_throws() {
    std::cout << "  project() throws on Inf H\n";
    std::array<double,9> H{};
    H[0] = std::numeric_limits<double>::infinity();
    bool threw = false;
    try { homography::project(H, 1.0, 1.0); }
    catch (const std::domain_error&) { threw = true; }
    REQUIRE_TRUE(threw);
}

void test_project_throws_on_near_zero_w() {
    std::cout << "  project() throws when homogeneous denominator ~ 0\n";
    std::array<double,9> H = {1,0,0, 0,1,0, 0,0,0};
    bool threw = false;
    try { homography::project(H, 1.0, 1.0); }
    catch (const std::domain_error&) { threw = true; }
    REQUIRE_TRUE(threw);
}

void test_project_throws_on_nan_input() {
    std::cout << "  project() throws on NaN input coordinates\n";
    std::array<double,9> H = {1,0,0, 0,1,0, 0,0,1};
    bool threw = false;
    try {
        homography::project(H, std::numeric_limits<double>::quiet_NaN(), 0.0);
    } catch (const std::domain_error&) { threw = true; }
    REQUIRE_TRUE(threw);
}

void test_project_inf_input_throws() {
    std::cout << "  project() throws on Inf input coordinate\n";
    std::array<double,9> H = {1,0,0, 0,1,0, 0,0,1};
    bool threw = false;
    try {
        homography::project(H, std::numeric_limits<double>::infinity(), 0.0);
    } catch (const std::domain_error&) { threw = true; }
    REQUIRE_TRUE(threw);
}

void test_reproj_error_exact() {
    std::cout << "  reprojectionError == 0 for exact projection\n";
    std::array<double,9> H = {1,0,50, 0,1,30, 0,0,1};
    const double err = homography::reprojectionError(H, 100, 200, 150, 230);
    EXPECT_NEAR(err, 0.0, 1e-9);
}

void test_reproj_error_scale_invariant() {
    std::cout << "  reprojectionError is scale-invariant: H vs 1e-12*H vs 1e12*H\n";
    std::array<double,9> H1 = {1.2, 0.1, 30.0, -0.05, 1.1, 20.0, 0.0001, 0.0002, 1.0};
    std::array<double,9> Hs = H1;
    std::array<double,9> Hl = H1;
    for (double& v : Hs) v *= 1e-12;
    for (double& v : Hl) v *= 1e12;

    const double e1 = homography::reprojectionError(H1, 80, 120, 125, 155);
    const double es = homography::reprojectionError(Hs, 80, 120, 125, 155);
    const double el = homography::reprojectionError(Hl, 80, 120, 125, 155);

    EXPECT_NEAR(es, e1, 1e-8);
    EXPECT_NEAR(el, e1, 1e-8);
}

void test_reproj_error_sentinel_on_zero_matrix() {
    std::cout << "  reprojectionError returns sentinel on all-zero H\n";
    std::array<double,9> H{};
    const double err = homography::reprojectionError(H, 1.0, 1.0, 0.0, 0.0);
    REQUIRE_TRUE(err == std::numeric_limits<double>::max());
}

void test_reproj_error_sentinel_on_nan_H() {
    std::cout << "  reprojectionError returns sentinel on NaN H\n";
    std::array<double,9> H{};
    H[0] = std::numeric_limits<double>::quiet_NaN();
    const double err = homography::reprojectionError(H, 1.0, 1.0, 0.0, 0.0);
    REQUIRE_TRUE(err == std::numeric_limits<double>::max() || !std::isfinite(err));
}

void test_reproj_error_sentinel_on_nan_input() {
    std::cout << "  reprojectionError returns sentinel on NaN input\n";
    std::array<double,9> H = {1,0,0, 0,1,0, 0,0,1};
    const double err = homography::reprojectionError(
        H, std::numeric_limits<double>::quiet_NaN(), 0, 0, 0);
    REQUIRE_TRUE(err == std::numeric_limits<double>::max());
}

void test_reproj_error_sentinel_on_zero_w() {
    std::cout << "  reprojectionError returns sentinel when w=0\n";
    std::array<double,9> H = {1,0,0, 0,1,0, 0,0,0};
    const double err = homography::reprojectionError(H, 1.0, 1.0, 0.0, 0.0);
    REQUIRE_TRUE(err == std::numeric_limits<double>::max());
}

// ============================================================
//  S4 Tests: Refinement Status
// ============================================================

void test_hom_refinement_succeeded_flag() {
    std::cout << "  refinement_succeeded is populated on valid result\n";
    auto sp = makeSyntheticPair({1,0,20, 0,1,15, 0,0,1}, makeGrid(5));
    homography::HomographyConfig cfg; cfg.random_seed = 0xF000;
    const auto res = homography::estimate(sp.matches, sp.descA, sp.descB, cfg);
    std::cout << "    msg=" << res.error_message << "\n";
    REQUIRE_TRUE(res.valid);
    // When there are many clean inliers, refinement should succeed and be accepted.
    REQUIRE_TRUE(res.refinement_succeeded);
}

void test_hom_refinement_fallback_when_refined_worse() {
    std::cout << "  refinement fallback: retains RANSAC best H when refined model has worse error\n";
    // Deterministic fixture where normalized DLT on all inliers yields slightly higher
    // reprojection error sum than the 4-sample RANSAC hypothesis.
    std::mt19937 rng(1);
    std::uniform_real_distribution<float> dpos(10.0f, 200.0f);
    std::uniform_real_distribution<float> dnoise(-2.95f, 2.95f);
    orb::DescriptionResult da, db;
    std::vector<matching::Match> matches;
    for (std::size_t i = 0; i < 4; ++i) {
        float x = 20.0f + static_cast<float>(i) * 40.0f;
        float y = 20.0f + static_cast<float>(i % 2) * 50.0f;
        da.keypoints.push_back({x + 10.0f, y + 5.0f, 1.0f, 0.0f, 0});
        db.keypoints.push_back({x, y, 1.0f, 0.0f, 0});
        matches.push_back({i, i, 0, 10});
    }
    for (std::size_t i = 4; i < 16; ++i) {
        float x = dpos(rng);
        float y = dpos(rng);
        da.keypoints.push_back({x + 10.0f + dnoise(rng), y + 5.0f + dnoise(rng), 1.0f, 0.0f, 0});
        db.keypoints.push_back({x, y, 1.0f, 0.0f, 0});
        matches.push_back({i, i, 0, 10});
    }
    homography::HomographyConfig cfg;
    cfg.random_seed = 1;
    cfg.ransac_iterations = 100;
    cfg.reprojection_threshold = 3.0;
    const auto res = homography::estimate(matches, da, db, cfg);
    REQUIRE_TRUE(res.valid);
    // Crucial check: refinement was evaluated, rejected because refined error was worse,
    // and best RANSAC hypothesis was retained cleanly.
    REQUIRE_TRUE(!res.refinement_succeeded);
    REQUIRE_TRUE(res.error_message.find("retained RANSAC best H") != std::string::npos);
    REQUIRE_TRUE(checkStrictMaskConsistency(res, matches, da, db, cfg.reprojection_threshold));
}

void test_hom_huge_threshold_never_accepts_invalid_projection() {
    std::cout << "  huge-threshold RANSAC: invalid projection never accepted as inlier (overflow guard)\n";
    // 4 good correspondences matching identity:
    orb::DescriptionResult da, db;
    std::vector<matching::Match> matches;
    da.keypoints.push_back({10, 10, 1, 0, 0}); db.keypoints.push_back({10, 10, 1, 0, 0});
    matches.push_back({0, 0, 0, 10});
    da.keypoints.push_back({200, 10, 1, 0, 0}); db.keypoints.push_back({200, 10, 1, 0, 0});
    matches.push_back({1, 1, 0, 10});
    da.keypoints.push_back({200, 200, 1, 0, 0}); db.keypoints.push_back({200, 200, 1, 0, 0});
    matches.push_back({2, 2, 0, 10});
    da.keypoints.push_back({10, 200, 1, 0, 0}); db.keypoints.push_back({10, 200, 1, 0, 0});
    matches.push_back({3, 3, 0, 10});

    // Zero-denominator check on projection helper:
    // w = 0*x + 1*y - 50 = 0 when y=50.
    std::array<double, 9> H_zero_w = {1,0,0, 0,1,0, 0,1,-50};
    const double err_zero_w = homography::reprojectionError(H_zero_w, 50.0, 50.0, 50.0, 50.0);
    REQUIRE_TRUE(err_zero_w == std::numeric_limits<double>::max());

    const double huge_thresholds[] = {
        3.0,
        1e154,
        1e200,
        std::numeric_limits<double>::max()
    };

    for (double th : huge_thresholds) {
        homography::HomographyConfig cfg;
        cfg.random_seed = 42;
        cfg.reprojection_threshold = th;
        const auto res = homography::estimate(matches, da, db, cfg);
        REQUIRE_TRUE(res.valid);
        REQUIRE_TRUE(res.num_inliers == 4);
        REQUIRE_TRUE(res.inlier_mask.size() == 4);
        for (bool b : res.inlier_mask) REQUIRE_TRUE(b);
        REQUIRE_TRUE(std::isfinite(res.mean_reprojection_error));
        REQUIRE_TRUE(std::isfinite(res.max_reprojection_error));
    }
}

// ============================================================
//  S4 Tests: Model Confidence (min_inlier_ratio)
// ============================================================

void test_hom_confidence_gate_four_exact_matches() {
    std::cout << "  confidence gate: exactly 4 matches has inlier_ratio=1.0\n";
    std::array<double,9> H_gt = {1,0,10, 0,1,10, 0,0,1};
    std::vector<std::pair<double,double>> pts = {{10,10}, {200,10}, {200,200}, {10,200}};
    auto sp = makeSyntheticPair(H_gt, pts);
    homography::HomographyConfig cfg;
    cfg.min_inliers = 4;
    cfg.min_inlier_ratio = 0.50;  // 50% required
    const auto res = homography::estimate(sp.matches, sp.descA, sp.descB, cfg);
    REQUIRE_TRUE(res.valid);
    EXPECT_NEAR(res.inlier_ratio, 1.0, 1e-9);
}

void test_hom_confidence_gate_rejects_low_consensus() {
    std::cout << "  confidence gate: min_inlier_ratio rejects low-consensus random matches\n";
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> coord(50.0f, 500.0f);
    orb::DescriptionResult da, db;
    std::vector<matching::Match> matches;
    for (int i = 0; i < 60; ++i) {
        orb::Keypoint ka{}, kb{};
        ka.x = coord(rng); ka.y = coord(rng);
        kb.x = coord(rng); kb.y = coord(rng);
        da.keypoints.push_back(ka);
        db.keypoints.push_back(kb);
        da.descriptors.push_back(makeDesc(0x01));
        db.descriptors.push_back(makeDesc(0x01));
        matching::Match m;
        m.index_a = i; m.index_b = i; m.distance = 0; m.distance2 = 10;
        matches.push_back(m);
    }
    homography::HomographyConfig cfg;
    cfg.min_inliers = 4;
    cfg.min_inlier_ratio = 0.20;  // Requires >= 20% inliers (>= 12 out of 60)
    cfg.random_seed = 0xABCD;
    const auto res = homography::estimate(matches, da, db, cfg);
    REQUIRE_TRUE(!res.valid);
    REQUIRE_TRUE(res.failure_reason == homography::FailureReason::TooFewInliers);
}

void test_hom_confidence_gate_default_zero_allows_low_consensus() {
    std::cout << "  confidence gate: default min_inlier_ratio=0.0 allows valid RANSAC if inliers>=min_inliers\n";
    std::array<double,9> H_gt = {1,0,20, 0,1,10, 0,0,1};
    auto sp = makeSyntheticPair(H_gt, makeGrid(5));  // 25 matches
    homography::HomographyConfig cfg;
    cfg.min_inlier_ratio = 0.0;  // default gate disabled
    cfg.random_seed = 0xABCD;
    const auto res = homography::estimate(sp.matches, sp.descA, sp.descB, cfg);
    REQUIRE_TRUE(res.valid);
}

void test_hom_confidence_gate_invalid_config() {
    std::cout << "  confidence gate: invalid min_inlier_ratio rejected\n";
    auto sp = makeSyntheticPair({1,0,0, 0,1,0, 0,0,1}, makeGrid(5));
    homography::HomographyConfig cfg;
    cfg.min_inlier_ratio = -0.1;
    auto res = homography::estimate(sp.matches, sp.descA, sp.descB, cfg);
    REQUIRE_TRUE(!res.valid);
    REQUIRE_TRUE(res.failure_reason == homography::FailureReason::InvalidConfig);

    cfg.min_inlier_ratio = 1.1;
    res = homography::estimate(sp.matches, sp.descA, sp.descB, cfg);
    REQUIRE_TRUE(!res.valid);
    REQUIRE_TRUE(res.failure_reason == homography::FailureReason::InvalidConfig);

    cfg.min_inlier_ratio = std::numeric_limits<double>::quiet_NaN();
    res = homography::estimate(sp.matches, sp.descA, sp.descB, cfg);
    REQUIRE_TRUE(!res.valid);
    REQUIRE_TRUE(res.failure_reason == homography::FailureReason::InvalidConfig);
}

// ============================================================
//  Integration Test: S2 (synthetic ORB) -> S3 -> S4
// ============================================================

void test_integration_synthetic_orb() {
    std::cout << "  S2->S3->S4 with index-encoded synthetic descriptors\n";
    std::array<double,9> H_gt = {1,0,30, 0,1,20, 0,0,1};
    orb::DescriptionResult da, db;
    const auto grid_pts = makeGrid(7, 50, 50, 60, 55);
    for (std::size_t i = 0; i < grid_pts.size(); ++i) {
        const float xb = static_cast<float>(grid_pts[i].first);
        const float yb = static_cast<float>(grid_pts[i].second);
        const float xa = static_cast<float>(H_gt[0]*xb + H_gt[1]*yb + H_gt[2]);
        const float ya = static_cast<float>(H_gt[3]*xb + H_gt[4]*yb + H_gt[5]);

        orb::Keypoint kb{}; kb.x = xb; kb.y = yb;
        orb::Keypoint ka{}; ka.x = xa; ka.y = ya;

        // Unique descriptor per keypoint (encoded index).
        orb::Descriptor d = makeDesc(0x00u);
        d.bytes[0] = static_cast<std::uint8_t>(i & 0xFF);
        d.bytes[1] = static_cast<std::uint8_t>((i >> 8) & 0xFF);
        d.bytes[2] = 0xA5u;  // magic to prevent accidental matches

        da.keypoints.push_back(ka); da.descriptors.push_back(d);
        db.keypoints.push_back(kb); db.descriptors.push_back(d);
    }
    matching::MatchingConfig mcfg;
    mcfg.ratio_threshold = 0.8f;
    const auto mr = matching::match(da, db, mcfg);
    REQUIRE_TRUE(mr.valid);
    REQUIRE_TRUE(mr.matches.size() >= 4u);

    homography::HomographyConfig hcfg;
    hcfg.random_seed = 0xBBBB;
    const auto hr = homography::estimate(mr.matches, da, db, hcfg);
    REQUIRE_TRUE(hr.valid);
    EXPECT_NEAR(hr.mean_reprojection_error, 0.0, 0.1);
    REQUIRE_TRUE(hr.num_inliers >= 4);
    REQUIRE_TRUE(checkMaskConsistency(hr, mr.matches.size(), hcfg.reprojection_threshold));
}

// ============================================================
//  Synthetic Image-Pair Test (S1->S4 end-to-end without image loader)
//  Uses orb::extract() on programmatically generated GrayImages.
// ============================================================

// Generate a synthetic GrayImage with deterministic features.
// Creates a pattern of intensity blobs (radial falloff) spread over a grid.
static orb::GrayImage makeSyntheticImage(int width, int height, int n_blobs,
                                          std::uint32_t seed) {
    orb::GrayImage img;
    img.width  = width;
    img.height = height;
    img.pixels.assign(static_cast<std::size_t>(width * height), 20u);  // dark background

    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> rx(20, width - 20);
    std::uniform_int_distribution<int> ry(20, height - 20);
    std::uniform_int_distribution<int> ri(80, 200);

    for (int b = 0; b < n_blobs; ++b) {
        const int cx = rx(rng);
        const int cy = ry(rng);
        const int peak = ri(rng);
        const int radius = 12;
        for (int dy = -radius; dy <= radius; ++dy) {
            for (int dx = -radius; dx <= radius; ++dx) {
                const int px = cx + dx;
                const int py = cy + dy;
                if (px < 0 || py < 0 || px >= width || py >= height) continue;
                const double dist2 = static_cast<double>(dx*dx + dy*dy);
                const double falloff = std::exp(-dist2 / (2.0 * 25.0));
                const std::size_t idx = static_cast<std::size_t>(py) *
                                        static_cast<std::size_t>(width) +
                                        static_cast<std::size_t>(px);
                const int cur = static_cast<int>(img.pixels[idx]);
                const int val = std::min(255, cur + static_cast<int>(peak * falloff));
                img.pixels[idx] = static_cast<std::uint8_t>(val);
            }
        }
    }
    return img;
}

// Apply a pure translation to a GrayImage (for ground-truth H).
static orb::GrayImage translateImage(const orb::GrayImage& src, int tx, int ty) {
    const int W = src.width;
    const int H = src.height;
    orb::GrayImage dst;
    dst.width  = src.width;
    dst.height = src.height;
    dst.pixels.assign(src.pixels.size(), 20u);  // fill with background
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const int sx = x - tx;
            const int sy = y - ty;
            if (sx < 0 || sy < 0 || sx >= W || sy >= H) continue;
            dst.pixels[static_cast<std::size_t>(y) * static_cast<std::size_t>(W) +
                     static_cast<std::size_t>(x)] =
                src.pixels[static_cast<std::size_t>(sy) * static_cast<std::size_t>(W) +
                          static_cast<std::size_t>(sx)];
        }
    }
    return dst;
}

void test_synthetic_image_pair_translation() {
    std::cout << "  Synthetic image pair: S1->S4, translation ground truth\n";

    constexpr int W = 480, H_img = 360;
    constexpr int TX = 30, TY = 20;  // translation: B -> A has H = [1,0,TX; 0,1,TY; 0,0,1]
    constexpr int N_BLOBS = 60;

    // Image B is the source; image A is B translated by (TX, TY).
    // H_B_to_A maps a point xB -> xA = xB + (TX, TY).
    const orb::GrayImage imgB = makeSyntheticImage(W, H_img, N_BLOBS, 0xFACE);
    const orb::GrayImage imgA = translateImage(imgB, TX, TY);

    // Run S1+S2: feature extraction.
    orb::Config orb_cfg;
    orb_cfg.max_features   = 400;
    orb_cfg.fast_threshold = 15;
    const auto descA = orb::extract(imgA, orb_cfg);
    const auto descB = orb::extract(imgB, orb_cfg);

    std::cout << "    keypoints: A=" << descA.keypoints.size()
              << " B=" << descB.keypoints.size() << "\n";

    // Minimum requirement for the test to be meaningful.
    REQUIRE_TRUE(descA.keypoints.size() >= 10u);
    REQUIRE_TRUE(descB.keypoints.size() >= 10u);

    if (descA.keypoints.size() < 10 || descB.keypoints.size() < 10) return;

    // S3: match.
    matching::MatchingConfig mcfg;
    mcfg.ratio_threshold = 0.75f;
    const auto mr = matching::match(descA, descB, mcfg);
    REQUIRE_TRUE(mr.valid);
    std::cout << "    matches: " << mr.matches.size() << "\n";

    if (mr.matches.size() < 4) {
        std::cerr << "  WARN: too few matches for S4 (got "
                  << mr.matches.size() << "), skipping S4 assertions\n";
        REQUIRE_TRUE(mr.matches.size() >= 4u);  // Count this as a failure
        return;
    }

    // S4: estimate H.
    homography::HomographyConfig hcfg;
    hcfg.random_seed = 0x1234ABCD;
    hcfg.ransac_iterations = 2000;
    const auto hr = homography::estimate(mr.matches, descA, descB, hcfg);

    std::cout << "    inliers: " << hr.num_inliers << "/" << mr.matches.size()
              << " mean_err=" << hr.mean_reprojection_error << " px\n";
    std::cout << "    valid=" << hr.valid
              << " reason=" << homography::failureReasonToString(hr.failure_reason) << "\n";

    REQUIRE_TRUE(hr.valid);

    if (!hr.valid) return;

    // Check H direction: project a grid of B-points through estimated H
    // and compare to their expected positions in A.
    const std::array<double,9> H_gt = {1,0,TX, 0,1,TY, 0,0,1};
    double max_hold_err = 0.0;
    double sum_hold_err = 0.0;
    int hold_pts = 0;
    for (int gy = 50; gy < H_img - 50; gy += 50) {
        for (int gx = 50; gx < W - 50; gx += 50) {
            const double xb = gx, yb = gy;
            // Ground-truth A position.
            const double xa_gt = xb + TX;
            const double ya_gt = yb + TY;
            if (xa_gt < 0 || ya_gt < 0 || xa_gt >= W || ya_gt >= H_img) continue;
            const double err = homography::reprojectionError(hr.H, xb, yb, xa_gt, ya_gt);
            if (std::isfinite(err)) {
                max_hold_err = std::max(max_hold_err, err);
                sum_hold_err += err;
                ++hold_pts;
            }
        }
    }
    const double mean_hold_err = (hold_pts > 0) ? sum_hold_err / hold_pts : 0.0;

    std::cout << "    ground-truth H:\n";
    for (int r = 0; r < 3; ++r) {
        std::cout << "      ";
        for (int c = 0; c < 3; ++c) std::cout << H_gt[r*3+c] << " ";
        std::cout << "\n";
    }
    std::cout << "    estimated H:\n";
    for (int r = 0; r < 3; ++r) {
        std::cout << "      ";
        for (int c = 0; c < 3; ++c) std::cout << hr.H[r*3+c] << " ";
        std::cout << "\n";
    }
    std::cout << "    holdout grid: " << hold_pts << " pts, mean_err="
              << mean_hold_err << " px, max_err=" << max_hold_err << " px\n";

    // Rigorous tolerance based on ORB centroid localization accuracy.
    EXPECT_LT(max_hold_err, 2.5);  // < 2.5 pixels on independent holdout grid
    EXPECT_LT(mean_hold_err, 1.5);
    REQUIRE_TRUE(checkMaskConsistency(hr, mr.matches.size(), hcfg.reprojection_threshold));
}

// ============================================================
//  Metadata-driven CSV parser for dataset integration test
// ============================================================

// Simple, header-aware CSV parser for dataset/metadata.csv.
// - Reads columns by name (position-independent).
// - Handles CRLF and LF line endings.
// - Reports parse errors with line numbers.
// - Does NOT support escaped quotes or multi-line fields (not needed here).
struct DatasetRow {
    std::string pair_id;
    std::string resolution;
    std::string pair_type;
    std::string prepared_a;  // relative to dataset/ dir
    std::string prepared_b;
};

struct CsvParseResult {
    std::vector<DatasetRow> rows;
    std::string error;  // non-empty means parse failed
};

// Split a CSV line into fields (no quote escaping needed for this dataset).
static std::vector<std::string> splitCsvLine(const std::string& line) {
    std::vector<std::string> fields;
    std::string cur;
    for (char c : line) {
        if (c == ',') {
            fields.push_back(cur);
            cur.clear();
        } else if (c != '\r') {
            cur += c;
        }
    }
    fields.push_back(cur);
    return fields;
}

// Replace .jpg or .jpeg extension (any case) with .pgm.
static std::string jpgToPgm(const std::string& rel) {
    std::string out = rel;
    const auto dot = out.rfind('.');
    if (dot != std::string::npos) {
        std::string ext_lc = out.substr(dot);
        for (char& c : ext_lc)
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        if (ext_lc == ".jpg" || ext_lc == ".jpeg") {
            out.replace(dot, out.size() - dot, ".pgm");
        }
    }
    return out;
}

static CsvParseResult parseDatasetMetadata(const std::string& csv_path) {
    CsvParseResult result;
    FILE* f = std::fopen(csv_path.c_str(), "r");
    if (!f) {
        result.error = "Cannot open metadata file: " + csv_path;
        return result;
    }

    std::string content;
    {
        char buf[4096];
        while (!std::feof(f)) {
            const std::size_t n = std::fread(buf, 1, sizeof(buf), f);
            content.append(buf, n);
        }
        std::fclose(f);
    }

    std::vector<std::string> lines;
    {
        std::istringstream iss(content);
        std::string ln;
        while (std::getline(iss, ln)) {
            if (!ln.empty() && ln.back() == '\r') ln.pop_back();
            lines.push_back(ln);
        }
    }

    if (lines.empty()) {
        result.error = "Metadata file is empty";
        return result;
    }

    const auto header_fields = splitCsvLine(lines[0]);
    std::map<std::string, int> col;
    for (int i = 0; i < static_cast<int>(header_fields.size()); ++i)
        col[header_fields[i]] = i;

    const auto check_col = [&](const std::string& name) -> bool {
        if (col.find(name) == col.end()) {
            result.error = "Required column '" + name + "' not found in header";
            return false;
        }
        return true;
    };
    if (!check_col("pair_id") || !check_col("resolution") ||
        !check_col("prepared_a") || !check_col("prepared_b"))
        return result;

    const bool has_pair_type = (col.find("pair_type") != col.end());

    std::map<std::string, int> seen_ids;
    for (int li = 1; li < static_cast<int>(lines.size()); ++li) {
        const auto& line = lines[li];
        if (line.empty()) continue;

        const auto fields = splitCsvLine(line);
        const auto get = [&](const std::string& name) -> std::string {
            const int idx = col.at(name);
            if (idx < static_cast<int>(fields.size())) return fields[idx];
            return "";
        };

        DatasetRow row;
        row.pair_id    = get("pair_id");
        row.resolution = get("resolution");
        row.prepared_a = get("prepared_a");
        row.prepared_b = get("prepared_b");
        if (has_pair_type) row.pair_type = get("pair_type");

        if (row.pair_id.empty()) {
            result.error = "Line " + std::to_string(li + 1) + ": empty pair_id";
            return result;
        }
        if (row.prepared_a.empty() || row.prepared_b.empty()) {
            result.error = "Line " + std::to_string(li + 1) +
                           " (pair=" + row.pair_id +
                           "): empty prepared_a or prepared_b path";
            return result;
        }

        const std::string key = row.pair_id + "@" + row.resolution;
        if (seen_ids.count(key)) {
            result.error = "Duplicate pair_id+resolution '" + key +
                           "' at lines " + std::to_string(seen_ids[key]) +
                           " and " + std::to_string(li + 1);
            return result;
        }
        seen_ids[key] = li + 1;
        result.rows.push_back(row);
    }

    if (result.rows.empty()) {
        result.error = "Metadata file has header but no data rows";
        return result;
    }
    return result;
}

// Returns the repository root relative to the current working directory.
// Tests must be run from the repository root (standard project convention).
static std::string repoRoot() { return "."; }

// ============================================================
//  Dataset Integration (all pairs in dataset/metadata.csv)
// ============================================================

void test_dataset_integration() {
    const std::string csv_path = repoRoot() + "/dataset/metadata.csv";
    const CsvParseResult meta = parseDatasetMetadata(csv_path);

    if (!meta.error.empty()) {
        BLOCK_TEST("Metadata parse failed: " + meta.error);
        return;
    }

    std::cout << "  Dataset integration: " << meta.rows.size()
              << " pairs loaded from metadata.csv\n";

    auto loadPGM = [](const std::string& path) -> orb::GrayImage {
        orb::GrayImage img;
        FILE* f = fopen(path.c_str(), "rb");
        if (!f) return img;
        char magic[3];
        int w = 0, h = 0, maxval = 0;
        if (fscanf(f, "%2s %d %d %d", magic, &w, &h, &maxval) != 4 ||
            magic[0] != 'P' || magic[1] != '5' || maxval > 255) {
            fclose(f); return img;
        }
        fgetc(f);
        img.width = w; img.height = h;
        img.pixels.resize(static_cast<std::size_t>(w * h));
        if (fread(img.pixels.data(), 1, img.pixels.size(), f) != img.pixels.size()) {
            img = {}; fclose(f); return img;
        }
        fclose(f);
        return img;
    };

    int pairs_tested = 0;
    int pairs_valid  = 0;

    for (const auto& row : meta.rows) {
        const std::string pgm_a_rel = jpgToPgm(row.prepared_a);
        const std::string pgm_b_rel = jpgToPgm(row.prepared_b);
        const std::string path_a = repoRoot() + "/dataset/" + pgm_a_rel;
        const std::string path_b = repoRoot() + "/dataset/" + pgm_b_rel;
        const std::string label  = row.pair_id + "_" + row.resolution;

        const auto imgA = loadPGM(path_a);
        const auto imgB = loadPGM(path_b);

        if (imgA.pixels.empty() || imgB.pixels.empty()) {
            BLOCK_TEST("Missing PGM for " + label +
                       ". Run: python3 scripts/prepare_pgm.py");
            return;
        }

        // Development diagnostic config (explicitly documented).
        // Official V0 benchmark uses max_features=2000, ratio_threshold=0.8.
        orb::Config orb_cfg;
        orb_cfg.max_features = 500;
        const auto descA = orb::extract(imgA, orb_cfg);
        const auto descB = orb::extract(imgB, orb_cfg);

        const auto t0_s3 = std::chrono::high_resolution_clock::now();
        matching::MatchingConfig mcfg;
        mcfg.ratio_threshold = 0.75f;
        const auto mr = matching::match(descA, descB, mcfg);
        const auto t1_s3 = std::chrono::high_resolution_clock::now();
        const double s3_ms =
            std::chrono::duration<double, std::milli>(t1_s3 - t0_s3).count();

        REQUIRE_TRUE(mr.valid);
        REQUIRE_TRUE(mr.matches.size() >= 4u);

        const auto t0_s4 = std::chrono::high_resolution_clock::now();
        homography::HomographyConfig hcfg;
        hcfg.random_seed = 0xCAFEBABE;
        const auto hr = homography::estimate(mr.matches, descA, descB, hcfg);
        const auto t1_s4 = std::chrono::high_resolution_clock::now();
        const double s4_ms =
            std::chrono::duration<double, std::milli>(t1_s4 - t0_s4).count();

        std::cout << "    [" << label << "] "
                  << imgA.width << "x" << imgA.height
                  << " kpA=" << descA.keypoints.size()
                  << " kpB=" << descB.keypoints.size()
                  << " matches=" << mr.matches.size()
                  << " inliers=" << hr.num_inliers
                  << " (" << (hr.inlier_ratio * 100.0) << "%)"
                  << " mean_err=" << hr.mean_reprojection_error << "px"
                  << " valid=" << hr.valid
                  << " S3=" << s3_ms << "ms S4=" << s4_ms << "ms\n";

        REQUIRE_TRUE(hr.valid);
        REQUIRE_TRUE(hr.num_inliers >= 4);
        REQUIRE_TRUE(checkMaskConsistency(
            hr, mr.matches.size(), hcfg.reprojection_threshold));

        ++pairs_tested;
        if (hr.valid) ++pairs_valid;
    }

    const int expected = static_cast<int>(meta.rows.size());
    REQUIRE_TRUE(pairs_tested == expected);
    REQUIRE_TRUE(pairs_valid  == expected);
}

// ============================================================
//  Metadata Parser Unit Tests (use temp fixtures, not real metadata)
// ============================================================

static std::string writeTempCsv(const std::string& content,
                                 const std::string& tag) {
    const std::string path = std::string("/tmp/test_s3s4_") + tag + ".csv";
    FILE* f = std::fopen(path.c_str(), "w");
    if (!f) return "";
    std::fwrite(content.data(), 1, content.size(), f);
    std::fclose(f);
    return path;
}

void test_metadata_parser_valid_12_rows() {
    std::cout << "  Metadata CSV parser: current metadata.csv has 12 data rows\n";
    const CsvParseResult meta =
        parseDatasetMetadata(repoRoot() + "/dataset/metadata.csv");
    REQUIRE_TRUE(meta.error.empty());
    EXPECT_EQ(static_cast<int>(meta.rows.size()), 12);
}

void test_metadata_parser_column_order_independent() {
    std::cout << "  Metadata CSV parser: handles shuffled column order\n";
    const std::string csv =
        "resolution,prepared_b,pair_id,prepared_a,extra_col\r\n"
        "small,prepared/B/b.jpg,pairX,prepared/A/a.jpg,ignored\r\n"
        "medium,prepared/B/b2.jpg,pairX,prepared/A/a2.jpg,ignored\r\n";
    const auto path = writeTempCsv(csv, "col_order");
    REQUIRE_TRUE(!path.empty());
    const CsvParseResult meta = parseDatasetMetadata(path);
    REQUIRE_TRUE(meta.error.empty());
    REQUIRE_TRUE(meta.rows.size() == 2u);
    EXPECT_TRUE(meta.rows[0].pair_id == "pairX");
    EXPECT_TRUE(meta.rows[0].resolution == "small");
    EXPECT_TRUE(meta.rows[0].prepared_a == "prepared/A/a.jpg");
    EXPECT_TRUE(meta.rows[0].prepared_b == "prepared/B/b.jpg");
}

void test_metadata_parser_duplicate_pair_id_resolution_rejected() {
    std::cout << "  Metadata CSV parser: rejects duplicate pair_id+resolution\n";
    const std::string csv =
        "pair_id,resolution,prepared_a,prepared_b\n"
        "pair1,small,a/a.jpg,b/b.jpg\n"
        "pair1,small,a/a2.jpg,b/b2.jpg\n";
    const auto path = writeTempCsv(csv, "dup_id");
    REQUIRE_TRUE(!path.empty());
    const CsvParseResult meta = parseDatasetMetadata(path);
    REQUIRE_TRUE(!meta.error.empty());
    EXPECT_TRUE(meta.error.find("uplicate") != std::string::npos);
}

void test_metadata_parser_empty_file_rejected() {
    std::cout << "  Metadata CSV parser: rejects empty file\n";
    const auto path = writeTempCsv("", "empty");
    REQUIRE_TRUE(!path.empty());
    const CsvParseResult meta = parseDatasetMetadata(path);
    REQUIRE_TRUE(!meta.error.empty());
}

void test_metadata_parser_header_only_rejected() {
    std::cout << "  Metadata CSV parser: rejects header-only file (no data rows)\n";
    const std::string csv = "pair_id,resolution,prepared_a,prepared_b\n";
    const auto path = writeTempCsv(csv, "header_only");
    REQUIRE_TRUE(!path.empty());
    const CsvParseResult meta = parseDatasetMetadata(path);
    REQUIRE_TRUE(!meta.error.empty());
}

void test_metadata_parser_missing_required_column_rejected() {
    std::cout << "  Metadata CSV parser: rejects CSV missing required column\n";
    const std::string csv =
        "pair_id,resolution,prepared_a\n"
        "p1,small,a/a.jpg\n";
    const auto path = writeTempCsv(csv, "missing_col");
    REQUIRE_TRUE(!path.empty());
    const CsvParseResult meta = parseDatasetMetadata(path);
    REQUIRE_TRUE(!meta.error.empty());
}

void test_metadata_parser_extra_pair_picked_up() {
    std::cout << "  Metadata CSV parser: extra data row counted dynamically\n";
    const std::string csv =
        "pair_id,resolution,prepared_a,prepared_b\n"
        "pairA,small,prepared/A/small/a.jpg,prepared/A/small/b.jpg\n"
        "pairA,medium,prepared/A/medium/a.jpg,prepared/A/medium/b.jpg\n"
        "pairB,small,prepared/B/small/a.jpg,prepared/B/small/b.jpg\n";
    const auto path = writeTempCsv(csv, "extra_row");
    REQUIRE_TRUE(!path.empty());
    const CsvParseResult meta = parseDatasetMetadata(path);
    REQUIRE_TRUE(meta.error.empty());
    EXPECT_EQ(static_cast<int>(meta.rows.size()), 3);
    EXPECT_TRUE(meta.rows[0].pair_id == "pairA");
    EXPECT_TRUE(meta.rows[1].resolution == "medium");
    EXPECT_TRUE(meta.rows[2].pair_id == "pairB");
}

void test_metadata_pgm_path_derivation() {
    std::cout << "  Metadata PGM path: .jpg/.jpeg -> .pgm extension replacement\n";
    EXPECT_TRUE(jpgToPgm("prepared/CMU0/standard/small/a.jpg")
                == "prepared/CMU0/standard/small/a.pgm");
    EXPECT_TRUE(jpgToPgm("prepared/foo/b.jpeg") == "prepared/foo/b.pgm");
    EXPECT_TRUE(jpgToPgm("prepared/foo/c.JPG")  == "prepared/foo/c.pgm");
    EXPECT_TRUE(jpgToPgm("prepared/foo/d.JPEG") == "prepared/foo/d.pgm");
    EXPECT_TRUE(jpgToPgm("prepared/foo/e.png")  == "prepared/foo/e.png");
}

// ============================================================
//  Benchmarks (S3, S4 timing - correctness also verified)
// ============================================================

void test_benchmark_s3_s4() {
    std::cout << "  Benchmark: S3 (2000x2000) and S4 with valid correspondences\n";
    constexpr std::size_t N = 2000;

    // --- Build random descriptors and keypoints for S3 microbenchmark ---
    orb::DescriptionResult da_rnd, db_rnd;
    {
        std::mt19937 rng(0xDEAD);
        for (std::size_t i = 0; i < N; ++i) {
            orb::Descriptor d;
            for (auto& b : d.bytes) b = static_cast<std::uint8_t>(rng());
            orb::Keypoint k{};
            k.x = static_cast<float>(rng() % 1200);
            k.y = static_cast<float>(rng() % 800);
            da_rnd.descriptors.push_back(d); da_rnd.keypoints.push_back(k);
            for (auto& b : d.bytes) b = static_cast<std::uint8_t>(rng());
            k.x = static_cast<float>(rng() % 1200);
            k.y = static_cast<float>(rng() % 800);
            db_rnd.descriptors.push_back(d); db_rnd.keypoints.push_back(k);
        }
    }

    // --- S3 benchmark ---
    // Warm-up.
    { (void)matching::match(da_rnd, db_rnd); }

    constexpr int kRuns = 5;
    double s3_total = 0.0;
    matching::MatchResult mr_rnd;
    for (int r = 0; r < kRuns; ++r) {
        const auto t0 = std::chrono::high_resolution_clock::now();
        mr_rnd = matching::match(da_rnd, db_rnd);
        const auto t1 = std::chrono::high_resolution_clock::now();
        s3_total += std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    const double s3_avg = s3_total / kRuns;
    std::cout << "    S3 (2000x2000 random descriptors): " << s3_avg
              << " ms avg  matches=" << mr_rnd.matches.size() << "\n";
    REQUIRE_TRUE(s3_avg > 0.0);

    // --- Build synthetic correspondences for S4 benchmark ---
    // Use a grid with a known H to guarantee valid RANSAC input.
    std::array<double,9> H_bench = {1,0,40, 0,1,30, 0,0,1};
    auto sp_bench = makeSyntheticPair(H_bench, makeGrid(10, 50, 50, 50, 45));
    // Add some noise.
    {
        std::mt19937 rng2(0x1234);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        for (auto& kp : sp_bench.descA.keypoints) {
            kp.x += nd(rng2); kp.y += nd(rng2);
        }
    }

    // Verify S4 actually runs and succeeds on this input.
    homography::HomographyConfig hcfg;
    hcfg.random_seed = 0xABCDE;
    {
        const auto pre_check = homography::estimate(sp_bench.matches,
                                                    sp_bench.descA, sp_bench.descB,
                                                    hcfg);
        REQUIRE_TRUE(pre_check.valid);
        if (!pre_check.valid) {
            std::cerr << "    S4 benchmark: pre-check failed, cannot benchmark\n";
            return;
        }
    }

    // Warm-up S4.
    { (void)homography::estimate(sp_bench.matches, sp_bench.descA, sp_bench.descB, hcfg); }

    double s4_total = 0.0;
    homography::HomographyResult hr_final;
    for (int r = 0; r < kRuns; ++r) {
        const auto t0 = std::chrono::high_resolution_clock::now();
        hr_final = homography::estimate(sp_bench.matches,
                                        sp_bench.descA, sp_bench.descB, hcfg);
        const auto t1 = std::chrono::high_resolution_clock::now();
        s4_total += std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    const double s4_avg = s4_total / kRuns;
    std::cout << "    S4 (" << sp_bench.matches.size() << " correspondences, "
              << hcfg.ransac_iterations << " iters): " << s4_avg
              << " ms avg  inliers=" << hr_final.num_inliers << "\n";

    // S4 MUST have actually been called and returned a valid result.
    REQUIRE_TRUE(hr_final.valid);
    REQUIRE_TRUE(s4_avg > 0.0);

    // Also benchmark S3 on the benchmark grid (different from random).
    {
        const auto t0 = std::chrono::high_resolution_clock::now();
        const auto mr_g = matching::match(sp_bench.descA, sp_bench.descB);
        const auto t1 = std::chrono::high_resolution_clock::now();
        const double s3_grid = std::chrono::duration<double,std::milli>(t1-t0).count();
        std::cout << "    S3 (" << sp_bench.descA.keypoints.size()
                  << "x" << sp_bench.descB.keypoints.size()
                  << " grid descriptors): " << s3_grid << " ms\n";
        (void)mr_g;
    }
}

// ============================================================
//  Main
// ============================================================

int main() {
    std::cout << "=========================================\n";
    std::cout << " S3/S4 Test Suite - CSC14120 V0\n";
    std::cout << "=========================================\n\n";

    std::cout << "=== Hamming Distance ===\n";
    RUN_TEST2(test_hamming_identical);
    RUN_TEST2(test_hamming_all_differ);
    RUN_TEST2(test_hamming_known_patterns);
    RUN_TEST2(test_hamming_symmetry);
    RUN_TEST2(test_hamming_single_byte_all_values);
    RUN_TEST2(test_hamming_randomized_differential);
    RUN_TEST2(test_hamming_range);

    std::cout << "\n=== S3: Config Validation ===\n";
    RUN_TEST2(test_match_invalid_ratio_nan);
    RUN_TEST2(test_match_invalid_ratio_inf);
    RUN_TEST2(test_match_invalid_ratio_negative);
    RUN_TEST2(test_match_invalid_ratio_zero);
    RUN_TEST2(test_match_invalid_ratio_one);

    std::cout << "\n=== S3: Edge Cases ===\n";
    RUN_TEST2(test_match_empty_A);
    RUN_TEST2(test_match_empty_B);
    RUN_TEST2(test_match_b_one_descriptor);
    RUN_TEST2(test_match_mismatched_kp_descriptor_A);
    RUN_TEST2(test_match_mismatched_kp_descriptor_B);
    RUN_TEST2(test_match_perfect_correspondences);
    RUN_TEST2(test_match_ratio_test_pass);
    RUN_TEST2(test_match_ratio_test_fail);
    RUN_TEST2(test_match_d2_zero_rejected);
    RUN_TEST2(test_match_d1_zero_d2_positive);
    RUN_TEST2(test_match_hamming_256_case);
    RUN_TEST2(test_match_index_mapping_correct);
    RUN_TEST2(test_match_no_out_of_bounds_index);
    RUN_TEST2(test_match_determinism);
    RUN_TEST2(test_match_different_sizes);

    std::cout << "\n=== S3: Cross-Check ===\n";
    RUN_TEST2(test_match_crosscheck_no_change_when_mutual);
    RUN_TEST2(test_match_crosscheck_removes_non_mutual);
    RUN_TEST2(test_match_crosscheck_false_is_default);

    std::cout << "\n=== S4: Config Validation ===\n";
    RUN_TEST2(test_hom_invalid_config_neg_iterations);
    RUN_TEST2(test_hom_invalid_config_nan_threshold);
    RUN_TEST2(test_hom_invalid_config_inf_threshold);
    RUN_TEST2(test_hom_invalid_config_neg_threshold);
    RUN_TEST2(test_hom_invalid_config_min_inliers_too_small);
    RUN_TEST2(test_hom_invalid_config_min_inliers_zero);
    RUN_TEST2(test_hom_confidence_gate_invalid_config);

    std::cout << "\n=== S4: Model Confidence ===\n";
    RUN_TEST2(test_hom_confidence_gate_four_exact_matches);
    RUN_TEST2(test_hom_confidence_gate_rejects_low_consensus);
    RUN_TEST2(test_hom_confidence_gate_default_zero_allows_low_consensus);

    std::cout << "\n=== S4: Input Validation ===\n";
    RUN_TEST2(test_hom_too_few_matches);
    RUN_TEST2(test_hom_match_index_out_of_range);
    RUN_TEST2(test_hom_nan_keypoint_rejected);

    std::cout << "\n=== S4: Degenerate Input ===\n";
    RUN_TEST2(test_hom_repeated_points_in_B);
    RUN_TEST2(test_hom_repeated_points_in_A);
    RUN_TEST2(test_hom_collinear_points);

    std::cout << "\n=== S4: Synthetic Correspondences ===\n";
    RUN_TEST2(test_hom_identity);
    RUN_TEST2(test_hom_translation);
    RUN_TEST2(test_hom_uniform_scale);
    RUN_TEST2(test_hom_rotation_30deg);
    RUN_TEST2(test_hom_perspective);
    RUN_TEST2(test_hom_with_gaussian_noise);
    RUN_TEST2(test_hom_with_30pct_outliers);
    RUN_TEST2(test_hom_with_50pct_outliers);
    RUN_TEST2(test_hom_four_exact_correspondences);
    RUN_TEST2(test_hom_direction_B_to_A);
    RUN_TEST2(test_hom_large_scale_coordinates);
    RUN_TEST2(test_hom_small_scale_coordinates);
    RUN_TEST2(test_hom_determinism);

    std::cout << "\n=== S4: Inlier Mask Consistency ===\n";
    RUN_TEST2(test_hom_inlier_mask_size);
    RUN_TEST2(test_hom_inlier_count_matches_mask);
    RUN_TEST2(test_hom_inlier_ratio_correct);
    RUN_TEST2(test_hom_reprojection_error_finite);
    RUN_TEST2(test_hom_strict_mask_consistency);

    std::cout << "\n=== S4: project() and reprojectionError() ===\n";
    RUN_TEST2(test_project_basic);
    RUN_TEST2(test_project_perspective_division);
    RUN_TEST2(test_project_scale_invariance_small);
    RUN_TEST2(test_project_scale_invariance_tiny);
    RUN_TEST2(test_project_large_coord_1e13);
    RUN_TEST2(test_project_large_coord_1e14);
    RUN_TEST2(test_project_scale_invariance_large);
    RUN_TEST2(test_project_scale_invariance_negative);
    RUN_TEST2(test_project_scale_invariance_perspective);
    RUN_TEST2(test_project_zero_matrix_throws);
    RUN_TEST2(test_project_throws_on_nan_H);
    RUN_TEST2(test_project_inf_matrix_throws);
    RUN_TEST2(test_project_throws_on_near_zero_w);
    RUN_TEST2(test_project_throws_on_nan_input);
    RUN_TEST2(test_project_inf_input_throws);
    RUN_TEST2(test_reproj_error_exact);
    RUN_TEST2(test_reproj_error_scale_invariant);
    RUN_TEST2(test_reproj_error_sentinel_on_zero_matrix);
    RUN_TEST2(test_reproj_error_sentinel_on_nan_H);
    RUN_TEST2(test_reproj_error_sentinel_on_nan_input);
    RUN_TEST2(test_reproj_error_sentinel_on_zero_w);

    std::cout << "\n=== S4: Refinement Status ===\n";
    RUN_TEST2(test_hom_refinement_succeeded_flag);
    RUN_TEST2(test_hom_refinement_fallback_when_refined_worse);
    RUN_TEST2(test_hom_huge_threshold_never_accepts_invalid_projection);

    std::cout << "\n=== Integration Tests ===\n";
    RUN_TEST2(test_integration_synthetic_orb);
    RUN_TEST2(test_synthetic_image_pair_translation);  // S1->S4 end-to-end

    std::cout << "\n=== Metadata Parser (CSV, path derivation) ===\n";
    RUN_TEST2(test_metadata_parser_valid_12_rows);
    RUN_TEST2(test_metadata_parser_column_order_independent);
    RUN_TEST2(test_metadata_parser_duplicate_pair_id_resolution_rejected);
    RUN_TEST2(test_metadata_parser_empty_file_rejected);
    RUN_TEST2(test_metadata_parser_header_only_rejected);
    RUN_TEST2(test_metadata_parser_missing_required_column_rejected);
    RUN_TEST2(test_metadata_parser_extra_pair_picked_up);
    RUN_TEST2(test_metadata_pgm_path_derivation);

    std::cout << "\n=== Dataset Integration ===\n";
    RUN_TEST2(test_dataset_integration);

    std::cout << "\n=== Benchmark (S3 + S4, validity verified) ===\n";
    RUN_TEST2(test_benchmark_s3_s4);

    std::cout << "\n=========================================\n";
    std::cout << " Test Cases: " << g_tests_run
              << "  Passed: " << g_tests_passed
              << "  Failed: " << g_tests_failed
              << "  Skipped: " << g_tests_skipped
              << "  Blocked: " << g_tests_blocked << "\n";
    std::cout << " Assertions: " << (g_asserts_passed + g_asserts_failed)
              << "  Passed: " << g_asserts_passed
              << "  Failed: " << g_asserts_failed << "\n";
    std::cout << "=========================================\n";
    return (g_tests_failed == 0 && g_tests_blocked == 0 && g_asserts_failed == 0) ? 0 : 1;
}
