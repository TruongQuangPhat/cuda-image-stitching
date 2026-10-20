# CUDA Image Stitching

An image stitching project developed for **CSC14120 – Parallel Programming**. The project implements a sequential C++ baseline (V0) and aims to accelerate selected stages using CUDA.

## Image Stitching Pipeline

The complete pipeline consists of six stages:

1. **Feature Detection (S1):** Detect image keypoints using multiscale FAST corners and Harris score ranking.
2. **Feature Description (S2):** Generate rotation-aware 256-bit binary descriptors (Rotated BRIEF).
3. **Feature Matching (S3):** Match corresponding keypoints using Hamming distance and Lowe's 2-NN ratio test.
4. **Homography Estimation (S4):** Estimate geometric transformation using RANSAC and Normalized DLT.
5. **Image Warping (S5):** Align images using inverse projective mapping and bilinear interpolation.
6. **Image Blending (S6):** Combine aligned images using multiband or feather blending.

> **Note on stage numbering:** S1–S4 above represents the internal milestone numbering for the sequential baseline. In the official course project specification, Feature Matching corresponds to Stage 4 and Homography Estimation corresponds to Stage 5.

---

## Repository Structure

```text
cuda-image-stitching/
├── src/
│   ├── orb.h
│   ├── orb.cpp
│   ├── matching.h
│   ├── matching.cpp
│   ├── homography.h
│   └── homography.cpp
├── tests/
│   └── test_matching_homography.cpp
├── scripts/
│   ├── prepare_pgm.py
│   └── run_matching_homography_tests.sh
├── dataset/
│   ├── metadata.csv
│   ├── raw/
│   └── prepared/
├── .gitignore
└── README.md
```

---

## Module Overview & APIs

### 1. ORB Feature Extraction (S1–S2)

Implemented in [`src/orb.h`](src/orb.h) and [`src/orb.cpp`](src/orb.cpp).
Namespace: `stitching::orb`.

- **Stage 1 (Detection):** Grayscale conversion, Gaussian image pyramid, FAST-9 corner detection, non-maximum suppression (NMS), and Harris response filtering.
- **Stage 2 (Description):** Intensity centroid orientation assignment and steered 256-bit BRIEF binary descriptors computed on blurred pyramid levels.

```cpp
namespace stitching::orb {

struct GrayImage;
struct Config;
struct DetectionResult;
struct DescriptionResult;

// Convert packed RGB/RGBA buffer to owned grayscale:
GrayImage toGrayscale(const std::uint8_t* data, int width, int height, int channels, std::size_t row_stride_bytes = 0);

// Stage 1: Detect multiscale FAST corners and rank by Harris response:
DetectionResult detect(const GrayImage& grayscale, const Config& config = Config{});

// Stage 2: Calculate intensity centroid orientations and build 256-bit steered BRIEF descriptors:
DescriptionResult describe(const DetectionResult& detection, const Config& config = Config{});

// Stage 1 + Stage 2 convenience pipeline:
DescriptionResult extract(const GrayImage& grayscale, const Config& config = Config{});

}
```

#### ORB Usage Example

```cpp
#include "orb.h"

stitching::orb::Config config;

// gray_image is a valid stitching::orb::GrayImage (e.g. from orb::toGrayscale() or PGM loader):
auto detection = stitching::orb::detect(gray_image, config);
auto description = stitching::orb::describe(detection, config);

// Alternatively, run both stages together:
auto result = stitching::orb::extract(gray_image, config);
```

### 2. Feature Matching (S3)

Implemented in [`src/matching.h`](src/matching.h) and [`src/matching.cpp`](src/matching.cpp).
Namespace: `stitching::matching`.

- **Hamming Distance:** Bitwise XOR followed by 256-bit population count.
- **Brute-Force 2-NN Search:** Finds the two closest descriptors in image B for each descriptor in image A.
- **Lowe's Ratio Test:** Accepts match only when $d_1 < \text{ratio\_threshold} \times d_2$ (default threshold = 0.8).
- **Mutual Cross-Check (Optional):** Retains match $(i, j)$ only if descriptor $i$ is also the mutual nearest neighbor of descriptor $j$.

| Function / Type | Description |
|---|---|
| `matching::hammingDistance(a, b)` | Hamming distance between two 256-bit descriptors in $[0, 256]$ |
| `matching::match(desc_a, desc_b, config)` | Brute-force 2-NN matching with Lowe's ratio test and optional cross-check |
| `matching::MatchingConfig` | Configuration (`ratio_threshold = 0.8f`, `cross_check = false`) |
| `matching::Match` | Correspondence struct (`index_a`, `index_b`, `distance`, `distance2`) |
| `matching::MatchResult` | Result struct (`matches`, `valid`, `error_message`) |

### 3. Homography Estimation (S4)

Implemented in [`src/homography.h`](src/homography.h) and [`src/homography.cpp`](src/homography.cpp).
Namespace: `stitching::homography`.

- **Hartley Normalized DLT:** Translates coordinates to centroid and scales mean distance to $\sqrt{2}$ to ensure numerical stability during Jacobi null-space eigensolving.
- **Scale-Adaptive Degeneracy Detection:** Rejects collinear or coincident 4-point sample configurations before DLT evaluation.
- **Deterministic RANSAC:** Iteratively tests 4-point hypotheses using a fixed RNG seed (`0xC5C14120u`) for reproducibility.
- **Robust Inlier Evaluation:** Uses scale-invariant projection validity checks, avoiding floating-point overflow even with extreme thresholds.
- **Inlier Refinement:** Re-estimates $H$ from all consensus inliers using Normalized DLT, falling back to the best hypothesis if refinement fails.

| Function / Type | Description |
|---|---|
| `homography::estimate(matches, desc_a, desc_b, config)` | Estimates projective transformation $H_{B \to A}$ from matches |
| `homography::project(H, x, y)` | Projects point $(x, y)$ through $H$ with scale-invariant normalization |
| `homography::reprojectionError(H, xb, yb, xa, ya)` | Computes Euclidean reprojection error $\|x_A - \text{project}(H, x_B)\|$ |
| `homography::HomographyConfig` | RANSAC iterations (2000), reprojection threshold (3.0 px), min inliers (4), random seed |
| `homography::HomographyResult` | Matrix $H$ (row-major), `valid`, `inlier_mask`, `num_inliers`, `inlier_ratio`, error stats |

---

## Build Requirements

- **C++ Compiler:** Clang++ (LLVM $\ge$ 14) or GCC ($\ge$ 10) supporting **C++17**.
- **Build Flags:** `-std=c++17 -Wall -Wextra -Wpedantic -Werror -O2`.
- **Operating System:** Linux or macOS (POSIX compliant).
- **External Dependencies:** Zero third-party computer vision libraries (no OpenCV, Eigen, or CUDA required for CPU V0).
- **Python (Dataset conversion):** Python 3.8+ with `Pillow` (PIL) for generating PGM test fixtures.

---

## Build & Run Tests

### 1. Automated Verification Script

Run the automated script to prepare test fixtures, build both release and sanitizer targets in `build/`, and execute all 101 test cases:

```bash
# Full verification (Dataset prep + Release tests + ASan/UBSan):
./scripts/run_matching_homography_tests.sh

# Release-only verification (skips ASan/UBSan):
RUN_SANITIZERS=0 ./scripts/run_matching_homography_tests.sh
```

### 2. Manual Build Commands

To build and run manually into the git-ignored `build/` directory:

```bash
mkdir -p build

# Release build:
clang++ -std=c++17 -Wall -Wextra -Wpedantic -Werror -O2 \
    -Isrc \
    tests/test_matching_homography.cpp \
    src/orb.cpp \
    src/matching.cpp \
    src/homography.cpp \
    -o build/test_matching_homography

./build/test_matching_homography

# Sanitizer build (AddressSanitizer + UndefinedBehaviorSanitizer):
clang++ -std=c++17 -Wall -Wextra -Wpedantic -Werror -O1 \
    -fsanitize=address,undefined -g \
    -Isrc \
    tests/test_matching_homography.cpp \
    src/orb.cpp \
    src/matching.cpp \
    src/homography.cpp \
    -o build/test_matching_homography_asan

./build/test_matching_homography_asan
```

---

## Dataset Preparation

The benchmark dataset configuration is specified in `dataset/metadata.csv` (12 configurations covering CMU0, CMU1, and CMU2 across small, medium, and large resolutions).

Before executing dataset integration tests on a fresh clone or export:

```bash
# Converts referenced JPEG pairs to grayscale P5 PGM fixtures:
python3 scripts/prepare_pgm.py
```

---

## Stage 4 → Stage 5 (Image Warping) Handoff Guide

To integrate the estimated Homography into Stage 5 (Image Warping):

```cpp
#include "matching.h"
#include "homography.h"
#include "orb.h"

// Example function illustrating the S3 -> S4 -> S5 pipeline handoff:
bool computeWarpingHomography(
    const stitching::orb::DescriptionResult& desc_a,
    const stitching::orb::DescriptionResult& desc_b,
    stitching::homography::HomographyResult& out_homo_res) {

    // 1. Run Feature Matching (Stage 3)
    stitching::matching::MatchingConfig match_cfg;
    match_cfg.ratio_threshold = 0.8f;
    match_cfg.cross_check = false;

    const auto match_res = stitching::matching::match(desc_a, desc_b, match_cfg);

    // Note on match_res.valid:
    // match_res.valid is false ONLY when inputs or MatchingConfig are invalid
    // (e.g. ratio_threshold <= 0 or keypoint/descriptor size mismatch).
    if (!match_res.valid) {
        // Invalid matching input or configuration: inspect match_res.error_message
        return false;
    }

    // A valid matching result may still contain fewer than 4 matches if the scene has low overlap:
    if (match_res.matches.size() < 4) {
        // Insufficient correspondences for homography estimation (DLT requires >= 4)
        return false;
    }

    // 2. Run Homography Estimation (Stage 4)
    stitching::homography::HomographyConfig homo_cfg;
    homo_cfg.ransac_iterations = 2000;
    homo_cfg.reprojection_threshold = 3.0; // In pixels
    homo_cfg.min_inliers = 4;
    homo_cfg.min_inlier_ratio = 0.0;       // Default: 0.0 (disabled)

    const auto homo_res = stitching::homography::estimate(
        match_res.matches, desc_a, desc_b, homo_cfg);

    // 3. Mandatory validity gate before warping
    if (!homo_res.valid) {
        // FailureReason: TooFewMatches, AllHypothesesFailed, TooFewInliers, etc.
        // Inspect homo_res.failure_reason and homo_res.error_message
        return false;
    }

    out_homo_res = homo_res;
    return true;
}
```

### Homography Convention & Warping Specifications

- **Direction:** $H_{B \to A}$ maps points from image B to image A: $x_A \sim H_{B \to A} x_B$.
- **Storage:** `homo_res.H` is a `std::array<double, 9>` stored in row-major order:
  $$\begin{bmatrix} H[0] & H[1] & H[2] \\ H[3] & H[4] & H[5] \\ H[6] & H[7] & H[8] \end{bmatrix}$$
- **Inverse Warping:** To render image B onto the canvas of image A, inverse mapping is applied for each canvas pixel $(x_A, y_A)$: compute $x_B = \text{inv}(H) x_A$ and sample pixel intensities via bilinear interpolation in image B.
- **Inlier Correspondences:** `homo_res.inlier_mask[k] == true` indicates that `match_res.matches[k]` is an inlier consistent with $H$.
- **Mathematical Validity vs. Visual Quality:** `homo_res.valid == true` certifies that RANSAC discovered a mathematically consistent homography meeting the inlier thresholds. It does not guarantee that the final stitched panorama is free from parallax or illumination seams (which must be handled by Stage 5 warping and Stage 6 multiband/feather blending).

---

## Authors & Acknowledgments

- **Author:** Truong Quang Phat (23120318)
- **Course:** CSC14120 – Parallel Programming (VNU-HCM University of Science)
- **AI-Assisted Development:** Implemented with Antigravity (Google DeepMind). All algorithms, numerical safeguards, and test coverage have been independently verified per academic integrity requirements.