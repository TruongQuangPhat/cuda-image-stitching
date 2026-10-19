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
- **Future Development:** CUDA acceleration of selected pipeline stages