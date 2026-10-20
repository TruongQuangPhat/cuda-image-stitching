#!/usr/bin/env bash
# CSC14120 - S3/S4 Automated Verification Suite
# Runs PGM fixture preparation, standard C++17 build & test, and ASan/UBSan checks.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

echo "=========================================================="
echo " CSC14120: S3 Feature Matching & S4 Homography Estimation "
echo " Full Verification and Regression Test Suite              "
echo "=========================================================="

echo ""
echo "[1/4] Preparing PGM fixtures from dataset/metadata.csv..."
python3 scripts/prepare_pgm.py

echo ""
echo "[2/4] Building test suite (C++17, -Wall -Wextra -Wpedantic -Werror -O2)..."
clang++ -std=c++17 -Wall -Wextra -Wpedantic -Werror -O2 \
    tests/test_s3_s4.cpp src/orb.cpp src/match_homography.cpp \
    -o tests/test_s3_s4

echo ""
echo "[3/4] Running test suite..."
./tests/test_s3_s4

echo ""
echo "[4/4] Building and running with AddressSanitizer & UndefinedBehaviorSanitizer..."
clang++ -std=c++17 -Wall -Wextra -Wpedantic -Werror -O1 \
    -fsanitize=address,undefined -g \
    tests/test_s3_s4.cpp src/orb.cpp src/match_homography.cpp \
    -o tests/test_s3_s4_asan
./tests/test_s3_s4_asan

echo ""
echo "=========================================================="
echo " All checks passed successfully! Baseline V0 is verified. "
echo "=========================================================="
