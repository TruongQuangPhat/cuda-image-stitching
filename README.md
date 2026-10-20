# CUDA Image Stitching

An image stitching project developed for **CSC14120 – Parallel Programming**. The project implements a sequential C++ baseline (V0) and aims to accelerate selected stages using CUDA.

## Image Stitching Pipeline

The pipeline consists of six stages:

1. **Feature Detection (S1):** Detect image keypoints using ORB.
2. **Feature Description (S2):** Generate rotation-aware binary descriptors.
3. **Feature Matching (S3):** Match corresponding keypoints using Hamming distance.
4. **Homography Estimation (S4):** Estimate geometric transformation using RANSAC and Normalized DLT.
5. **Image Warping (S5):** Align images using inverse mapping and bilinear interpolation.
6. **Image Blending (S6):** Combine aligned images using feather blending.

## ORB Feature Extraction (S1–S2)

Implemented in `src/orb.h` and `src/orb.cpp` using sequential C++17.

### Stage 1 – Feature Detection

Stage 1 identifies distinctive image locations through:

- Grayscale conversion and Gaussian image pyramid construction.
- FAST-9 corner detection.
- Non-Maximum Suppression (NMS).
- Harris corner ranking and multiscale keypoint selection.

**Output:** Keypoints containing coordinates, scale, octave, and response score, along with the image pyramid.

### Stage 2 – Feature Description

Stage 2 generates descriptors from the detected keypoints through:

- Orientation estimation using Intensity Centroid.
- Rotated BRIEF descriptor generation.
- 256-bit (32-byte) binary representation.

**Output:** Oriented keypoints and their corresponding binary descriptors.

### Public API

| Function | Responsibility |
|---|---|
| `orb::detect()` | Stage 1 – Feature Detection |
| `orb::describe()` | Stage 2 – Feature Description |
| `orb::extract()` | Complete ORB Feature Extraction |

The output of `describe()` is a `DescriptionResult` containing two aligned collections: `keypoints` and `descriptors`. Each descriptor corresponds to the keypoint at the same index.

These features are passed to Stage 3 for matching using Hamming distance.

## Implementation

- **Language:** C++17
- **V0:** Sequential CPU implementation
- **Feature Extraction:** ORB (FAST + Harris + Rotated BRIEF)
- **Feature Matching:** Brute-Force Hamming + Lowe's 2-NN Ratio Test
- **Homography Estimation:** Hartley Normalized DLT + RANSAC + Inlier Refinement
- **Future Development:** CUDA acceleration of selected pipeline stages

## Feature Matching (S3)

Implemented in [`src/match_homography.h`](src/match_homography.h) and
[`src/match_homography.cpp`](src/match_homography.cpp).

Namespace: `stitching::matching`.

### Stage 3 – Feature Matching

Stage 3 finds keypoint correspondences between two images through:

- Hamming distance computation: `popcount(descriptor_A XOR descriptor_B)` (256-bit).
- Brute-force 2-nearest-neighbor search: for each descriptor in A, finds the two
  closest descriptors in B by Hamming distance.
- Lowe's ratio test: accepts a match only when `d1 < ratio_threshold × d2`
  (default `ratio_threshold = 0.8`).

**Output:** A `MatchResult` containing accepted `Match` records, each with
`index_a`, `index_b` (indices into the respective `DescriptionResult`),
`distance` (d1), and `distance2` (d2).

### Public API – S3

| Function / Type | Responsibility |
|---|---|
| `matching::hammingDistance(a, b)` | Hamming distance between two 256-bit descriptors (0–256) |
| `matching::match(desc_a, desc_b, config)` | Brute-force 2-NN matching with ratio test |
| `matching::MatchingConfig` | `ratio_threshold` (default 0.8), `cross_check` (default false) |
| `matching::Match` | `index_a`, `index_b`, `distance`, `distance2` |
| `matching::MatchResult` | `matches`, `valid`, `error_message` |

---

## Homography Estimation (S4)

Implemented in the same `src/match_homography.h` / `src/match_homography.cpp`.

Namespace: `stitching::homography`.

### Stage 4 – Homography Estimation

Stage 4 estimates the geometric transformation between image A and image B through:

- **Hartley Normalized DLT:** Normalizes point coordinates to improve numerical
  conditioning before solving the homogeneous linear system via Jacobi eigendecomposition
  of A^T A.
- **RANSAC:** Iteratively samples 4 correspondences, tests degeneracy (collinearity
  and repeated points), estimates a candidate H, and selects the best model by
  inlier count then by total inlier reprojection error.
- **Re-estimation from inliers:** The final Homography is computed by running
  Normalized DLT on all inliers of the best RANSAC model.
- **Homography direction:** `x_A ~ H_B_to_A × x_B` (maps image-B points to image-A).

**Output:** A `HomographyResult` containing `H` (3×3 row-major), `valid`,
`failure_reason`, `inlier_mask`, `num_inliers`, `inlier_ratio`,
`mean_reprojection_error`, and `max_reprojection_error`.

### Public API – S4

| Function / Type | Responsibility |
|---|---|
| `homography::estimate(matches, desc_a, desc_b, config)` | Estimate H_B_to_A from matched correspondences |
| `homography::project(H, x, y)` | Project a 2-D point through H with scale-invariant normalization |
| `homography::reprojectionError(H, xb, yb, xa, ya)` | Directional Euclidean reprojection error in pixels: \|\|x_A - project(H, x_B)\|\| |
| `homography::HomographyConfig` | `ransac_iterations` (2000), `reprojection_threshold` (3.0 px), `min_inliers` (4), `min_inlier_ratio` (0.0), `random_seed` |
| `homography::FailureReason` | `None`, `InvalidConfig`, `InvalidInput`, `TooFewMatches`, `AllHypothesesFailed`, `TooFewInliers`, `RefinementFailed` |
| `homography::HomographyResult` | `H`, `valid`, `failure_reason`, `error_message`, `inlier_mask`, `num_inliers`, `inlier_ratio`, `mean_reprojection_error`, `max_reprojection_error`, `refinement_succeeded` |

---

## Building and Testing

### Image Preparation (Dataset Integration)

Before running the full dataset integration suite, convert the dataset JPEG images to PGM fixtures using the provided Python script (requires Python 3 + Pillow):

```bash
# Convert dataset images from JPEG to PGM (does not alter original JPEGs):
python3 scripts/prepare_pgm.py
```

### Running Test Suite

You can execute the automated check script directly:

```bash
./scripts/run_s3_s4_checks.sh
```

Or manually run:

```bash
# Compile S3 and S4 test suite (standard C++17, zero external CV dependencies):
clang++ -std=c++17 -Wall -Wextra -Wpedantic -Werror -O2 \
    tests/test_s3_s4.cpp src/orb.cpp src/match_homography.cpp \
    -o tests/test_s3_s4

./tests/test_s3_s4   # -> 101/101 test cases passed, 1551/1551 assertions passed (0 failed, 0 blocked)

# With AddressSanitizer and UndefinedBehaviorSanitizer:
clang++ -std=c++17 -Wall -Wextra -Wpedantic -Werror -O1 \
    -fsanitize=address,undefined -g \
    tests/test_s3_s4.cpp src/orb.cpp src/match_homography.cpp \
    -o tests/test_s3_s4_asan

./tests/test_s3_s4_asan  # -> 101/101 passed, 0 memory leaks, 0 UB, clean exit
```

---

## Integration Notes for S5–S6

### Pipeline Handoff Code Example

The handoff from S3 to S4, and from S4 to S5 (Image Warping) requires validating both intermediate results:

```cpp
#include "src/match_homography.h"

// 1. Configure and run S3 (Feature Matching)
stitching::matching::MatchingConfig matching_config;
matching_config.ratio_threshold = 0.8f;   // Standard Lowe's ratio test threshold
matching_config.cross_check = false;     // Mutual nearest neighbor filtering

const auto mr = stitching::matching::match(desc_a, desc_b, matching_config);
if (!mr.valid) {
    // Handle matching failure (e.g. fewer than 2 descriptors in one image)
    // Error description available in mr.error_message
    return;
}

// 2. Configure and run S4 (Homography Estimation)
stitching::homography::HomographyConfig homography_config;
homography_config.ransac_iterations = 2000;
homography_config.reprojection_threshold = 3.0; // In pixels
homography_config.min_inliers = 4;
// Optional model confidence gate (default 0.0 = disabled):
// If enabled by team agreement (e.g., 0.15), must be consistent across V0 and CUDA versions.
homography_config.min_inlier_ratio = 0.0;

const auto hr = stitching::homography::estimate(
    mr.matches, desc_a, desc_b, homography_config);

// 3. Mandatory validity check before using Homography
if (!hr.valid) {
    // hr.failure_reason (TooFewMatches, AllHypothesesFailed, TooFewInliers, etc.)
    // hr.error_message describes the exact reason.
    return;
}

// 4. Homography convention and usage in S5 Warping:
// - Direction: H_B_to_A maps points from image B to image A: x_A ~ hr.H * x_B.
// - Representation: hr.H is a std::array<double, 9> in row-major order:
//     [ H[0] H[1] H[2] ]
//     [ H[3] H[4] H[5] ]
//     [ H[6] H[7] H[8] ]
// - Inverse warping convention: to sample pixel values from image B onto the canvas of image A,
//   compute x_B ~ inv(hr.H) * x_A and perform bilinear interpolation.
// - Inlier correspondences: hr.inlier_mask[k] == true <=> mr.matches[k] is an inlier.
```

### Model Confidence and Validation Notes

- **Mathematical Validity vs. Visual Quality:** `hr.valid == true` guarantees that RANSAC found a mathematically consistent homography meeting the inlier threshold on the given features. It does not guarantee that the final stitched panorama will be visually seamless or free of parallax ghosting (which depends on scene planarity, parallax, and S5/S6 warping and blending).
- **Confidence Gate (`min_inlier_ratio`):** By default `0.0` (disabled). If the team decides to enforce a minimum inlier ratio consensus threshold, the same value must be used consistently across V0 and CUDA implementations.
- **Pipeline Configurations:**
  - *Development Diagnostic Configuration (in `tests/test_s3_s4.cpp`):* Uses `max_features = 500`, `ratio_threshold = 0.75f`, RANSAC 2000 iterations to verify integration quickly (~10 ms per pair).
  - *Official V0 Baseline Configuration:* Expected to use `max_features = 2000`, `ratio_threshold = 0.8f`, RANSAC 2000 iterations once finalized by the team. Diagnostic benchmarks should not be conflated with official baseline speedup benchmarks.