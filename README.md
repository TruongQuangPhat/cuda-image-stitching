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
| `homography::project(H, x, y)` | Project a 2-D point through H (homogeneous division) |
| `homography::reprojectionError(H, xb, yb, xa, ya)` | Euclidean reprojection error in pixels |
| `homography::HomographyConfig` | `ransac_iterations` (2000), `reprojection_threshold` (3.0 px), `min_inliers` (4), `random_seed` |
| `homography::HomographyResult` | Complete estimation result and diagnostics |
| `homography::FailureReason` | `TooFewMatches`, `AllHypothesesFailed`, `RefinementFailed`, `TooFewInliers` |

---

## Building and Testing

```bash
# Compile S3 and S4 together (no extra dependencies beyond the STL):
clang++ -std=c++17 -Wall -Wextra -O2 \
    tests/test_s3_s4.cpp src/orb.cpp src/match_homography.cpp \
    -o tests/test_s3_s4

./tests/test_s3_s4   # → 381 passed, 0 failed

# With AddressSanitizer and UndefinedBehaviorSanitizer:
clang++ -std=c++17 -Wall -Wextra -O1 \
    -fsanitize=address,undefined \
    tests/test_s3_s4.cpp src/orb.cpp src/match_homography.cpp \
    -o tests/test_s3_s4_asan

./tests/test_s3_s4_asan  # → 381 passed, 0 failed, no sanitizer errors
```

---

## Integration Notes for S5–S6

The handoff from S4 to S5 (Image Warping) requires:

```cpp
#include "src/match_homography.h"

// 1. Run S3
stitching::matching::MatchResult mr =
    stitching::matching::match(desc_a, desc_b);

// 2. Run S4
stitching::homography::HomographyResult hr =
    stitching::homography::estimate(mr.matches, desc_a, desc_b);

// 3. Check validity before using H
if (!hr.valid) {
    // hr.failure_reason and hr.error_message describe why.
    return;
}

// 4. Use H_B_to_A: x_A ~ hr.H * x_B
// hr.H is a std::array<double, 9> in row-major order.
// hr.inlier_mask[k] == true  ↔  mr.matches[k] is an inlier.
```