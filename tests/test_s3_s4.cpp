// Unit tests for Stage 3 (Feature Matching) and Stage 4 (Homography Estimation).
//
// Authors: Truong Quang Phat (23120318)
// Build:
//   clang++ -std=c++17 -Wall -Wextra -Wpedantic -O2 \
//       tests/test_s3_s4.cpp src/orb.cpp src/match_homography.cpp \
//       -o tests/test_s3_s4 && ./tests/test_s3_s4
//   (or with AddressSanitizer / UBSan:)
//   clang++ -std=c++17 -Wall -Wextra -Wpedantic -fsanitize=address,undefined \
//       tests/test_s3_s4.cpp src/orb.cpp src/match_homography.cpp \
//       -o tests/test_s3_s4_asan && ./tests/test_s3_s4_asan

#include "../src/match_homography.h"
#include "../src/orb.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

// ============================================================
//  Minimal test framework
// ============================================================

static int g_passed = 0;
static int g_failed = 0;

#define EXPECT_TRUE(expr) do { \
    if (!(expr)) { \
        std::cerr << "  FAIL [" << __LINE__ << "]: " #expr "\n"; \
        ++g_failed; \
    } else { ++g_passed; } \
} while(0)

#define EXPECT_EQ(a, b) EXPECT_TRUE((a) == (b))
#define EXPECT_NEAR(a, b, tol) EXPECT_TRUE(std::abs((double)(a) - (double)(b)) <= (tol))
#define EXPECT_FALSE(expr) EXPECT_TRUE(!(expr))

#define RUN_TEST(fn) do { \
    std::cout << "[ RUN ] " #fn "\n"; \
    fn(); \
    std::cout << "\n"; \
} while(0)

// ============================================================
//  Helpers
// ============================================================

using namespace stitching;

// Build a descriptor with all bytes set to `b`.
static orb::Descriptor makeDesc(std::uint8_t b) {
    orb::Descriptor d;
    d.bytes.fill(b);
    return d;
}

// Build a descriptor from a bit-pattern repeated across bytes.
static orb::Descriptor makeDescBit(bool v) {
    return makeDesc(v ? 0xFFu : 0x00u);
}

// Build a trivial DescriptionResult.
static orb::DescriptionResult makeDescResult(
    std::vector<orb::Descriptor> descs,
    std::vector<orb::Keypoint>   kps = {}) {
    if (kps.empty()) {
        kps.resize(descs.size());
        for (std::size_t i = 0; i < descs.size(); ++i) {
            kps[i].x = static_cast<float>(i);
            kps[i].y = static_cast<float>(i);
        }
    }
    orb::DescriptionResult r;
    r.descriptors = std::move(descs);
    r.keypoints   = std::move(kps);
    return r;
}

// Apply a 3×3 Homography to a 2-D point (available for future tests).
[[maybe_unused]]
static std::pair<double,double> applyH(const std::array<double,9>& H,
                                        double x, double y) {
    const double w = H[6]*x + H[7]*y + H[8];
    return {(H[0]*x + H[1]*y + H[2]) / w,
            (H[3]*x + H[4]*y + H[5]) / w};
}

// Make a DescriptionResult from a set of keypoint coordinates and identical descriptors.
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

// Generate synthetic correspondences from a ground-truth H.
// Returns a MatchResult and two DescriptionResults.
struct SyntheticPair {
    orb::DescriptionResult descA;
    orb::DescriptionResult descB;
    matching::MatchResult   mr;
    std::array<double,9>    H_gt; // H_B_to_A
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
        const double xa = (H_B_to_A[0]*xb + H_B_to_A[1]*yb + H_B_to_A[2]) / w + ndist(rng);
        const double ya = (H_B_to_A[3]*xb + H_B_to_A[4]*yb + H_B_to_A[5]) / w + ndist(rng);

        orb::Keypoint kb{}; kb.x = static_cast<float>(xb); kb.y = static_cast<float>(yb);
        orb::Keypoint ka{}; ka.x = static_cast<float>(xa); ka.y = static_cast<float>(ya);
        sp.descB.keypoints.push_back(kb);
        sp.descB.descriptors.push_back(d);
        sp.descA.keypoints.push_back(ka);
        sp.descA.descriptors.push_back(d);

        matching::Match m;
        m.index_a = i; m.index_b = i;
        m.distance = 0; m.distance2 = 10;
        sp.mr.matches.push_back(m);
    }
    return sp;
}

// ============================================================
//  S3 Tests: Hamming Distance
// ============================================================

void test_hamming_identical() {
    std::cout << "  hamming(x, x) == 0 for any x\n";
    for (int b = 0; b < 256; b += 17) {
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
    std::cout << "  hamming with known bit-flip counts\n";
    orb::Descriptor a = makeDesc(0x00u);
    orb::Descriptor b = makeDesc(0x00u);
    // Set exactly 3 bits in the first byte of b.
    b.bytes[0] = 0x07u; // 00000111 → 3 bits set
    EXPECT_EQ(matching::hammingDistance(a, b), 3);
    EXPECT_EQ(matching::hammingDistance(b, a), 3);

    // Set one bit in each of 4 bytes.
    orb::Descriptor c = makeDesc(0x00u);
    c.bytes[0] = 0x01u;
    c.bytes[5] = 0x01u;
    c.bytes[15] = 0x01u;
    c.bytes[31] = 0x01u;
    EXPECT_EQ(matching::hammingDistance(a, c), 4);
}

void test_hamming_symmetry() {
    std::cout << "  hamming is symmetric\n";
    orb::Descriptor a = makeDesc(0xAAu); // 10101010 pattern
    orb::Descriptor b = makeDesc(0x55u); // 01010101 pattern (complement)
    EXPECT_EQ(matching::hammingDistance(a, b), matching::hammingDistance(b, a));
    EXPECT_EQ(matching::hammingDistance(a, b), 256);
}

void test_hamming_single_byte_all_values() {
    std::cout << "  hamming(byte=v, 0x00) == popcount(v) for all v\n";
    for (int v = 0; v < 256; ++v) {
        orb::Descriptor a = makeDesc(0x00u);
        orb::Descriptor b = makeDesc(0x00u);
        a.bytes[0] = static_cast<std::uint8_t>(v);
        int expected = 0;
        for (int bit = 0; bit < 8; ++bit) expected += (v >> bit) & 1;
        EXPECT_EQ(matching::hammingDistance(a, b), expected);
    }
}

// ============================================================
//  S3 Tests: Matching
// ============================================================

void test_match_empty_inputs() {
    std::cout << "  match on empty A/B returns valid, empty matches\n";
    auto ra = makeDescResult({});
    auto rb = makeDescResult({makeDesc(0x00u), makeDesc(0xFFu)});
    const auto res = matching::match(ra, rb);
    EXPECT_TRUE(res.valid);
    EXPECT_EQ(res.matches.size(), 0u);
}

void test_match_b_one_descriptor() {
    std::cout << "  B with 1 descriptor: no ratio-test possible, 0 matches\n";
    auto ra = makeDescResult({makeDesc(0x00u), makeDesc(0x01u)});
    auto rb = makeDescResult({makeDesc(0x00u)});
    const auto res = matching::match(ra, rb);
    EXPECT_TRUE(res.valid);
    EXPECT_EQ(res.matches.size(), 0u);
}

void test_match_mismatched_kp_descriptor_counts() {
    std::cout << "  keypoint/descriptor count mismatch returns invalid\n";
    orb::DescriptionResult ra;
    ra.keypoints  = {{0,0,1,0,0,0}};
    ra.descriptors = {makeDesc(0x00u), makeDesc(0xFFu)}; // size mismatch

    orb::DescriptionResult rb = makeDescResult({makeDesc(0x00u), makeDesc(0x01u)});
    const auto res = matching::match(ra, rb);
    EXPECT_FALSE(res.valid);
}

void test_match_perfect_correspondences() {
    std::cout << "  identical descriptor pairs → all match with d1=0\n";
    orb::Descriptor d0 = makeDesc(0x00u);
    orb::Descriptor d1 = makeDesc(0xAAu);
    orb::Descriptor d2 = makeDesc(0x55u);
    // A has d0, d1, d2 in order.
    // B has d1, d2, d0 → expected matches: A[0]↔B[2], A[1]↔B[0], A[2]↔B[1].
    auto ra = makeDescResult({d0, d1, d2});
    auto rb = makeDescResult({d1, d2, d0});
    const auto res = matching::match(ra, rb);
    EXPECT_TRUE(res.valid);
    EXPECT_EQ(res.matches.size(), 3u);
    for (const auto& m : res.matches) {
        EXPECT_EQ(m.distance, 0);
    }
}

void test_match_ratio_test_threshold() {
    std::cout << "  ratio-test boundary: strict inequality\n";
    // A has one descriptor = 0x00.
    // B: best match distance d1=10, second = 10/0.8=12.5, so d2 should be 13
    // for a pass (10 < 0.8*13 = 10.4 → pass).
    // We do this by crafting descriptors with controlled distances.
    orb::Descriptor da = makeDesc(0x00u);
    // b0: distance=10 → 10 bits different in first 2 bytes
    orb::Descriptor b0 = makeDesc(0x00u);
    b0.bytes[0] = 0x1Fu; // 5 bits
    b0.bytes[1] = 0x1Fu; // 5 bits → total 10 bits
    // b1: distance=13
    orb::Descriptor b1 = makeDesc(0x00u);
    b1.bytes[0] = 0xFFu; // 8 bits
    b1.bytes[1] = 0x1Fu; // 5 bits → total 13 bits

    auto ra = makeDescResult({da});
    auto rb = makeDescResult({b0, b1});

    matching::MatchingConfig cfg;
    cfg.ratio_threshold = 0.8f;
    const auto res = matching::match(ra, rb, cfg);
    EXPECT_TRUE(res.valid);
    // d1=10, d2=13: 10 < 0.8*13 = 10.4 → PASS
    EXPECT_EQ(res.matches.size(), 1u);
    if (!res.matches.empty()) EXPECT_EQ(res.matches[0].distance, 10);
}

void test_match_d2_zero() {
    std::cout << "  d2==0 (two identical best): ratio test rejects\n";
    // Both B descriptors identical to A → d1=0, d2=0 → reject.
    orb::Descriptor da = makeDesc(0x00u);
    auto ra = makeDescResult({da});
    auto rb = makeDescResult({makeDesc(0x00u), makeDesc(0x00u)});
    const auto res = matching::match(ra, rb);
    EXPECT_TRUE(res.valid);
    EXPECT_EQ(res.matches.size(), 0u);
}

void test_match_d1_zero_d2_positive() {
    std::cout << "  d1=0, d2>0 → ratio test passes\n";
    orb::Descriptor da = makeDesc(0x00u);
    orb::Descriptor b1 = makeDesc(0x00u);      // d1=0
    orb::Descriptor b2 = makeDesc(0xFFu);      // d2=256
    auto ra = makeDescResult({da});
    auto rb = makeDescResult({b1, b2});
    const auto res = matching::match(ra, rb);
    EXPECT_TRUE(res.valid);
    EXPECT_EQ(res.matches.size(), 1u);
    if (!res.matches.empty()) EXPECT_EQ(res.matches[0].distance, 0);
}

void test_match_different_sizes() {
    std::cout << "  A.size() != B.size() is fine\n";
    auto ra = makeDescResult({makeDesc(0x00u), makeDesc(0xAAu), makeDesc(0x55u)});
    auto rb = makeDescResult({makeDesc(0x00u), makeDesc(0xAAu)});
    const auto res = matching::match(ra, rb);
    EXPECT_TRUE(res.valid);
    EXPECT_TRUE(res.matches.size() <= 3u);
}

void test_match_determinism() {
    std::cout << "  match is deterministic across multiple calls\n";
    auto ra = makeDescResult({makeDesc(0x0Fu), makeDesc(0xF0u), makeDesc(0xAAu)});
    auto rb = makeDescResult({makeDesc(0x0Fu), makeDesc(0x55u), makeDesc(0xF0u), makeDesc(0xAAu)});
    const auto r1 = matching::match(ra, rb);
    const auto r2 = matching::match(ra, rb);
    EXPECT_EQ(r1.matches.size(), r2.matches.size());
    for (std::size_t i = 0; i < r1.matches.size(); ++i) {
        EXPECT_EQ(r1.matches[i].index_a, r2.matches[i].index_a);
        EXPECT_EQ(r1.matches[i].index_b, r2.matches[i].index_b);
        EXPECT_EQ(r1.matches[i].distance, r2.matches[i].distance);
    }
}

void test_match_index_mapping() {
    std::cout << "  match preserves correct index mapping\n";
    // A[0]=d0, A[1]=d1; B[0]=d2(unrelated), B[1]=d0, B[2]=d1
    orb::Descriptor d0 = makeDesc(0x01u);
    orb::Descriptor d1 = makeDesc(0x02u);
    orb::Descriptor d2 = makeDesc(0xFFu);
    auto ra = makeDescResult({d0, d1});
    auto rb = makeDescResult({d2, d0, d1});
    const auto res = matching::match(ra, rb);
    EXPECT_TRUE(res.valid);
    for (const auto& m : res.matches) {
        if (m.index_a == 0) EXPECT_EQ(m.index_b, 1u);
        if (m.index_a == 1) EXPECT_EQ(m.index_b, 2u);
    }
}

// ============================================================
//  S4 Tests: Normalized DLT and RANSAC
// ============================================================

void test_homography_identity() {
    std::cout << "  identity H: xA == xB\n";
    std::array<double,9> I = {1,0,0, 0,1,0, 0,0,1};
    std::vector<std::pair<double,double>> pts = {{10,20},{100,50},{300,400},{50,300}};
    auto sp = makeSyntheticPair(I, pts);
    homography::HomographyConfig cfg;
    cfg.random_seed = 0x1234;
    const auto res = homography::estimate(sp.mr.matches, sp.descA, sp.descB, cfg);
    EXPECT_TRUE(res.valid);
    if (res.valid) {
        EXPECT_NEAR(res.mean_reprojection_error, 0.0, 0.01);
        EXPECT_EQ(res.num_inliers, static_cast<int>(pts.size()));
    }
}

// Helper: generate a scattered 2-D grid of n×n points starting at (ox,oy)
// with spacing (sx, sy), then optionally add jitter to break collinearity.
static std::vector<std::pair<double,double>>
makeGrid(int n, double ox = 50.0, double oy = 50.0,
         double sx = 60.0, double sy = 55.0) {
    std::vector<std::pair<double,double>> pts;
    pts.reserve(static_cast<std::size_t>(n * n));
    for (int r = 0; r < n; ++r)
        for (int c = 0; c < n; ++c)
            pts.push_back({ox + c * sx + r * 3.0,
                           oy + r * sy + c * 2.5});
    return pts;
}

void test_homography_translation() {
    std::cout << "  translation H: xA = xB + (tx, ty)\n";
    std::array<double,9> H_gt = {1,0,50, 0,1,30, 0,0,1};
    auto sp = makeSyntheticPair(H_gt, makeGrid(5));
    homography::HomographyConfig cfg;
    cfg.random_seed = 0xABCD;
    const auto res = homography::estimate(sp.mr.matches, sp.descA, sp.descB, cfg);
    EXPECT_TRUE(res.valid);
    if (res.valid) EXPECT_NEAR(res.mean_reprojection_error, 0.0, 0.1);
}

void test_homography_uniform_scale() {
    std::cout << "  uniform scale H\n";
    std::array<double,9> H_gt = {2,0,0, 0,2,0, 0,0,1};
    auto sp = makeSyntheticPair(H_gt, makeGrid(4, 30, 30, 40, 35));
    homography::HomographyConfig cfg;
    cfg.random_seed = 0x5678;
    const auto res = homography::estimate(sp.mr.matches, sp.descA, sp.descB, cfg);
    EXPECT_TRUE(res.valid);
    if (res.valid) EXPECT_NEAR(res.mean_reprojection_error, 0.0, 0.1);
}

void test_homography_rotation() {
    std::cout << "  pure rotation H (30 degrees)\n";
    const double angle = 30.0 * M_PI / 180.0;
    const double c = std::cos(angle), s = std::sin(angle);
    std::array<double,9> H_gt = {c,-s,0, s,c,0, 0,0,1};
    auto sp = makeSyntheticPair(H_gt, makeGrid(4, 80, 80, 50, 50));
    homography::HomographyConfig cfg;
    cfg.random_seed = 0x9ABC;
    const auto res = homography::estimate(sp.mr.matches, sp.descA, sp.descB, cfg);
    EXPECT_TRUE(res.valid);
    if (res.valid) EXPECT_NEAR(res.mean_reprojection_error, 0.0, 0.1);
}

void test_homography_perspective() {
    std::cout << "  perspective H\n";
    // A genuine perspective transform.
    std::array<double,9> H_gt = {1.2, 0.1, 20,
                                  0.05, 0.9, 15,
                                  0.001, 0.0002, 1};
    auto sp = makeSyntheticPair(H_gt, makeGrid(5, 40, 40, 50, 45));
    homography::HomographyConfig cfg;
    cfg.random_seed = 0xDEF0;
    const auto res = homography::estimate(sp.mr.matches, sp.descA, sp.descB, cfg);
    EXPECT_TRUE(res.valid);
    if (res.valid) EXPECT_NEAR(res.mean_reprojection_error, 0.0, 0.2);
}

void test_homography_with_noise() {
    std::cout << "  H estimation with Gaussian noise (sigma=0.5px)\n";
    std::array<double,9> H_gt = {1.1, 0.05, 30, -0.03, 0.95, 10, 0,0,1};
    auto sp = makeSyntheticPair(H_gt, makeGrid(6, 30, 30, 40, 35), 0.5, 0x1111);
    homography::HomographyConfig cfg;
    cfg.random_seed = 0x2222;
    const auto res = homography::estimate(sp.mr.matches, sp.descA, sp.descB, cfg);
    EXPECT_TRUE(res.valid);
    // With small noise, error should stay below reprojection threshold.
    if (res.valid) EXPECT_TRUE(res.mean_reprojection_error < 1.5);
}

void test_homography_with_outliers() {
    std::cout << "  H estimation with 30% outliers\n";
    std::array<double,9> H_gt = {1,0,50, 0,1,30, 0,0,1};
    auto sp = makeSyntheticPair(H_gt, makeGrid(6, 40, 40, 45, 40));

    // Corrupt ~30% of matches with random offsets.
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> noise(-200.0f, 200.0f);
    for (std::size_t i = 0; i < sp.descA.keypoints.size(); ++i) {
        if (rng() % 3 == 0) {
            sp.descA.keypoints[i].x += noise(rng);
            sp.descA.keypoints[i].y += noise(rng);
        }
    }
    homography::HomographyConfig cfg;
    cfg.random_seed = 0x3333;
    const auto res = homography::estimate(sp.mr.matches, sp.descA, sp.descB, cfg);
    EXPECT_TRUE(res.valid);
    if (res.valid) {
        EXPECT_TRUE(res.inlier_ratio > 0.5);
        EXPECT_NEAR(res.mean_reprojection_error, 0.0, 0.5);
    }
}

void test_homography_too_few_matches() {
    std::cout << "  fewer than 4 matches → valid=false, TooFewMatches\n";
    std::vector<matching::Match> matches = {{0,0,0,10},{1,1,0,10},{2,2,0,10}};
    orb::DescriptionResult da = makeKpResult({{10,10},{20,20},{30,30}});
    orb::DescriptionResult db = makeKpResult({{10,10},{20,20},{30,30}});
    const auto res = homography::estimate(matches, da, db);
    EXPECT_FALSE(res.valid);
    EXPECT_TRUE(res.failure_reason == homography::FailureReason::TooFewMatches);
    EXPECT_EQ(static_cast<int>(res.inlier_mask.size()), 3);
}

void test_homography_four_valid_correspondences() {
    std::cout << "  exactly 4 non-degenerate correspondences → valid\n";
    std::array<double,9> H_gt = {1,0,40, 0,1,20, 0,0,1};
    auto sp = makeSyntheticPair(H_gt,
        {{100,100},{200,100},{100,200},{200,200}});
    homography::HomographyConfig cfg;
    cfg.random_seed = 0x4444;
    const auto res = homography::estimate(sp.mr.matches, sp.descA, sp.descB, cfg);
    EXPECT_TRUE(res.valid);
    if (res.valid) EXPECT_NEAR(res.mean_reprojection_error, 0.0, 0.01);
}

void test_homography_degenerate_collinear() {
    std::cout << "  collinear points → AllHypothesesFailed or TooFewInliers\n";
    // All points on a horizontal line — degenerate.
    auto sp = makeSyntheticPair(
        {1,0,0, 0,1,0, 0,0,1},
        {{10,50},{20,50},{30,50},{40,50},{50,50},{60,50}});
    homography::HomographyConfig cfg;
    cfg.random_seed = 0x5555;
    const auto res = homography::estimate(sp.mr.matches, sp.descA, sp.descB, cfg);
    // Either fails or succeeds with very few inliers if points are truly collinear.
    // We just check it doesn't crash and doesn't silently return a broken H.
    if (res.valid) {
        // If it "succeeded", make sure the H is finite.
        for (double v : res.H) EXPECT_TRUE(std::isfinite(v));
    } else {
        EXPECT_TRUE(res.failure_reason != homography::FailureReason::None);
    }
}

void test_homography_repeated_points() {
    std::cout << "  repeated point pairs → fails gracefully\n";
    // 10 matches, all from the same (xb,yb) → degenerate sample.
    std::vector<matching::Match> matches;
    orb::DescriptionResult da, db;
    for (int i = 0; i < 10; ++i) {
        orb::Keypoint ka{}; ka.x = 100.0f + i; ka.y = 100.0f;
        orb::Keypoint kb{}; kb.x = 50.0f;       kb.y = 50.0f; // All same
        da.keypoints.push_back(ka); da.descriptors.push_back(makeDesc(0u));
        db.keypoints.push_back(kb); db.descriptors.push_back(makeDesc(0u));
        matches.push_back({static_cast<std::size_t>(i),
                           static_cast<std::size_t>(i), 0, 10});
    }
    homography::HomographyConfig cfg;
    cfg.random_seed = 0x6666;
    const auto res = homography::estimate(matches, da, db, cfg);
    // Should fail or return a degenerate model.
    if (res.valid) {
        for (double v : res.H) EXPECT_TRUE(std::isfinite(v));
    }
}

void test_homography_nan_input() {
    std::cout << "  NaN in keypoints → handled (no crash, valid==false or filtered)\n";
    std::vector<matching::Match> matches;
    orb::DescriptionResult da, db;
    const float nan = std::numeric_limits<float>::quiet_NaN();
    for (int i = 0; i < 5; ++i) {
        orb::Keypoint ka{}; ka.x = (i == 0) ? nan : 100.0f + i; ka.y = 100.0f;
        orb::Keypoint kb{}; kb.x = 50.0f + i; kb.y = 50.0f + i;
        da.keypoints.push_back(ka); da.descriptors.push_back(makeDesc(0u));
        db.keypoints.push_back(kb); db.descriptors.push_back(makeDesc(0u));
        matches.push_back({static_cast<std::size_t>(i),
                           static_cast<std::size_t>(i), 0, 10});
    }
    homography::HomographyConfig cfg;
    cfg.random_seed = 0x7777;
    // Must not crash.
    const auto res = homography::estimate(matches, da, db, cfg);
    if (res.valid) {
        for (double v : res.H) EXPECT_TRUE(std::isfinite(v));
    }
}

void test_homography_direction_B_to_A() {
    std::cout << "  direction check: H projects xB → xA\n";
    // H: rotate 45 degrees, scale 1.5, translate (100, 50).
    const double angle = 45.0 * M_PI / 180.0;
    const double c = std::cos(angle) * 1.5, s_v = std::sin(angle) * 1.5;
    std::array<double,9> H_gt = {c, -s_v, 100,
                                  s_v,  c,  50,
                                  0,  0,   1};
    auto sp = makeSyntheticPair(H_gt, makeGrid(5, 20, 20, 40, 35));
    homography::HomographyConfig cfg;
    cfg.random_seed = 0x8888;
    const auto res = homography::estimate(sp.mr.matches, sp.descA, sp.descB, cfg);
    EXPECT_TRUE(res.valid);
    if (!res.valid) return;
    // Verify: for each known point in B, projecting with estimated H should
    // approximately equal the corresponding point in A.
    for (std::size_t k = 0; k < sp.descB.keypoints.size(); ++k) {
        const double xb = sp.descB.keypoints[k].x;
        const double yb = sp.descB.keypoints[k].y;
        const double xa = sp.descA.keypoints[k].x;
        const double ya = sp.descA.keypoints[k].y;
        const double err = homography::reprojectionError(res.H, xb, yb, xa, ya);
        EXPECT_NEAR(err, 0.0, 0.5); // Allow small error due to noise-free DLT
    }
}

void test_homography_determinism() {
    std::cout << "  same seed → same result (deterministic RANSAC)\n";
    std::array<double,9> H_gt = {1.1, 0.05, 30, 0.02, 0.95, 10, 0.0001, 0, 1};
    auto sp = makeSyntheticPair(H_gt, makeGrid(5, 30, 30, 40, 35), 0.3, 12345);
    homography::HomographyConfig cfg;
    cfg.random_seed = 0x9999;
    const auto r1 = homography::estimate(sp.mr.matches, sp.descA, sp.descB, cfg);
    const auto r2 = homography::estimate(sp.mr.matches, sp.descA, sp.descB, cfg);
    EXPECT_EQ(r1.valid, r2.valid);
    EXPECT_EQ(r1.num_inliers, r2.num_inliers);
    if (r1.valid && r2.valid) {
        for (int i = 0; i < 9; ++i)
            EXPECT_NEAR(r1.H[i], r2.H[i], 1e-12);
    }
}

void test_homography_inlier_mask_consistency() {
    std::cout << "  inlier_mask.size() == matches.size()\n";
    std::array<double,9> H_gt = {1,0,20, 0,1,10, 0,0,1};
    auto sp = makeSyntheticPair(H_gt,
        {{10,10},{100,100},{200,200},{50,150},{150,50}});
    homography::HomographyConfig cfg;
    cfg.random_seed = 0xAAAA;
    const auto res = homography::estimate(sp.mr.matches, sp.descA, sp.descB, cfg);
    EXPECT_EQ(res.inlier_mask.size(), sp.mr.matches.size());
}

void test_homography_project_function() {
    std::cout << "  project() correctly computes perspective division\n";
    std::array<double,9> H = {1,0,10, 0,1,20, 0,0,1};
    const auto p = homography::project(H, 5.0, 5.0);
    EXPECT_NEAR(p.x, 15.0, 1e-10);
    EXPECT_NEAR(p.y, 25.0, 1e-10);
}

void test_homography_reprojection_error_fn() {
    std::cout << "  reprojectionError returns 0 for exact projection\n";
    std::array<double,9> H = {1,0,50, 0,1,30, 0,0,1};
    const double xb = 100, yb = 200;
    const double xa = 150, ya = 230;
    const double err = homography::reprojectionError(H, xb, yb, xa, ya);
    EXPECT_NEAR(err, 0.0, 1e-9);
}

// ============================================================
//  Integration test: S2 → S3 → S4 with synthetic ORB output
// ============================================================

void test_integration_s2_to_s4() {
    std::cout << "  Integration: S2→S3→S4 with synthetic DescriptionResults\n";
    // Create two images worth of keypoints/descriptors where we know matches.
    std::array<double,9> H_gt = {1,0,30, 0,1,20, 0,0,1};

    // 7×7 grid of non-collinear point pairs.
    orb::DescriptionResult da, db;
    const auto grid_pts = makeGrid(7, 50, 50, 60, 55);
    for (std::size_t i = 0; i < grid_pts.size(); ++i) {
        const float xb = static_cast<float>(grid_pts[i].first);
        const float yb = static_cast<float>(grid_pts[i].second);

        const double xa = H_gt[0]*xb + H_gt[1]*yb + H_gt[2];
        const double ya = H_gt[3]*xb + H_gt[4]*yb + H_gt[5];

        orb::Keypoint kb{}; kb.x = xb; kb.y = yb;
        orb::Keypoint ka{}; ka.x = static_cast<float>(xa); ka.y = static_cast<float>(ya);

        // Use descriptor that encodes the index for easy identification.
        orb::Descriptor d = makeDesc(0x00u);
        d.bytes[0] = static_cast<std::uint8_t>(i);
        d.bytes[1] = static_cast<std::uint8_t>(i >> 8u);

        da.keypoints.push_back(ka); da.descriptors.push_back(d);
        db.keypoints.push_back(kb); db.descriptors.push_back(d);
    }

    // S3: match
    matching::MatchingConfig mcfg;
    mcfg.ratio_threshold = 0.8f;
    const auto mr = matching::match(da, db, mcfg);
    EXPECT_TRUE(mr.valid);
    EXPECT_TRUE(mr.matches.size() >= 4u);

    // S4: estimate H
    homography::HomographyConfig hcfg;
    hcfg.random_seed = 0xBBBB;
    const auto hr = homography::estimate(mr.matches, da, db, hcfg);
    EXPECT_TRUE(hr.valid);
    if (hr.valid) {
        EXPECT_NEAR(hr.mean_reprojection_error, 0.0, 0.1);
        EXPECT_TRUE(hr.num_inliers >= 4);
    }
}

// ============================================================
//  Profiling helper
// ============================================================

void test_profiling_timing() {
    std::cout << "  Profiling S3 and S4 timing (2000 descriptors each)\n";
    const std::size_t N = 2000;
    orb::DescriptionResult da, db;
    std::mt19937 rng(0xDEAD);
    for (std::size_t i = 0; i < N; ++i) {
        orb::Descriptor d;
        for (auto& b : d.bytes) b = static_cast<std::uint8_t>(rng());
        orb::Keypoint k{}; k.x = static_cast<float>(rng()%1200); k.y = static_cast<float>(rng()%800);
        da.descriptors.push_back(d); da.keypoints.push_back(k);
        for (auto& b : d.bytes) b = static_cast<std::uint8_t>(rng());
        k.x = static_cast<float>(rng()%1200); k.y = static_cast<float>(rng()%800);
        db.descriptors.push_back(d); db.keypoints.push_back(k);
    }

    // Warm-up.
    matching::match(da, db);

    // S3 timing (5 runs).
    constexpr int kRuns = 5;
    double s3_total = 0.0;
    matching::MatchResult mr;
    for (int r = 0; r < kRuns; ++r) {
        const auto t0 = std::chrono::high_resolution_clock::now();
        mr = matching::match(da, db);
        const auto t1 = std::chrono::high_resolution_clock::now();
        s3_total += std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    std::cout << "    S3 avg over " << kRuns << " runs: "
              << s3_total / kRuns << " ms  (matches=" << mr.matches.size() << ")\n";

    // S4 timing (5 runs) — only if we have enough matches.
    if (mr.matches.size() >= 4) {
        double s4_total = 0.0;
        for (int r = 0; r < kRuns; ++r) {
            const auto t0 = std::chrono::high_resolution_clock::now();
            homography::estimate(mr.matches, da, db);
            const auto t1 = std::chrono::high_resolution_clock::now();
            s4_total += std::chrono::duration<double, std::milli>(t1 - t0).count();
        }
        std::cout << "    S4 avg over " << kRuns << " runs: "
                  << s4_total / kRuns << " ms\n";
    }
    EXPECT_TRUE(s3_total > 0.0); // Just confirm it ran.
}

// ============================================================
//  Main
// ============================================================

int main() {
    std::cout << "========================================\n";
    std::cout << " S3/S4 Test Suite – CSC14120 V0\n";
    std::cout << "========================================\n\n";

    std::cout << "=== Hamming Distance ===\n";
    RUN_TEST(test_hamming_identical);
    RUN_TEST(test_hamming_all_differ);
    RUN_TEST(test_hamming_known_patterns);
    RUN_TEST(test_hamming_symmetry);
    RUN_TEST(test_hamming_single_byte_all_values);

    std::cout << "\n=== Feature Matching (S3) ===\n";
    RUN_TEST(test_match_empty_inputs);
    RUN_TEST(test_match_b_one_descriptor);
    RUN_TEST(test_match_mismatched_kp_descriptor_counts);
    RUN_TEST(test_match_perfect_correspondences);
    RUN_TEST(test_match_ratio_test_threshold);
    RUN_TEST(test_match_d2_zero);
    RUN_TEST(test_match_d1_zero_d2_positive);
    RUN_TEST(test_match_different_sizes);
    RUN_TEST(test_match_determinism);
    RUN_TEST(test_match_index_mapping);

    std::cout << "\n=== Homography Estimation (S4) ===\n";
    RUN_TEST(test_homography_identity);
    RUN_TEST(test_homography_translation);
    RUN_TEST(test_homography_uniform_scale);
    RUN_TEST(test_homography_rotation);
    RUN_TEST(test_homography_perspective);
    RUN_TEST(test_homography_with_noise);
    RUN_TEST(test_homography_with_outliers);
    RUN_TEST(test_homography_too_few_matches);
    RUN_TEST(test_homography_four_valid_correspondences);
    RUN_TEST(test_homography_degenerate_collinear);
    RUN_TEST(test_homography_repeated_points);
    RUN_TEST(test_homography_nan_input);
    RUN_TEST(test_homography_direction_B_to_A);
    RUN_TEST(test_homography_determinism);
    RUN_TEST(test_homography_inlier_mask_consistency);
    RUN_TEST(test_homography_project_function);
    RUN_TEST(test_homography_reprojection_error_fn);

    std::cout << "\n=== Integration Tests ===\n";
    RUN_TEST(test_integration_s2_to_s4);

    std::cout << "\n=== Profiling (informational, not pass/fail) ===\n";
    RUN_TEST(test_profiling_timing);

    std::cout << "\n========================================\n";
    std::cout << " Results: " << g_passed << " passed, "
              << g_failed << " failed.\n";
    std::cout << "========================================\n";
    return (g_failed == 0) ? 0 : 1;
}
