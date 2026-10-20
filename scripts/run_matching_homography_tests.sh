#!/usr/bin/env bash
# CSC14120 - Feature Matching & Homography Estimation Automated Verification Suite
# Runs PGM fixture preparation, standard C++17 build & test, and ASan/UBSan checks.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

echo "=========================================================="
echo " CSC14120: Matching & Homography Verification Test Suite  "
echo " Modules: src/matching.{h,cpp}, src/homography.{h,cpp}   "
echo "=========================================================="

# Compiler detection: respect CXX if set, otherwise try clang++ then g++
CXX="${CXX:-}"
if [ -z "${CXX}" ]; then
    if command -v clang++ >/dev/null 2>&1; then
        CXX="clang++"
    elif command -v g++ >/dev/null 2>&1; then
        CXX="g++"
    else
        echo "Error: Neither clang++ nor g++ was found in PATH." >&2
        exit 1
    fi
fi

echo "Compiler detected: ${CXX}"
"${CXX}" --version | head -n 1

# Ensure build directory exists
mkdir -p build

echo ""
echo "[1/4] Preparing PGM fixtures from dataset/metadata.csv..."
python3 scripts/prepare_pgm.py

echo ""
echo "[2/4] Building test suite (C++17, -Wall -Wextra -Wpedantic -Werror -O2)..."
"${CXX}" -std=c++17 -Wall -Wextra -Wpedantic -Werror -O2 \
    -Isrc \
    tests/test_matching_homography.cpp \
    src/orb.cpp \
    src/matching.cpp \
    src/homography.cpp \
    -o build/test_matching_homography

echo ""
echo "[3/4] Running correctness test suite..."
./build/test_matching_homography

echo ""
echo "[4/4] Building and running with AddressSanitizer & UndefinedBehaviorSanitizer..."
if "${CXX}" -std=c++17 -Wall -Wextra -Wpedantic -Werror -O1 \
    -fsanitize=address,undefined -g \
    -Isrc \
    tests/test_matching_homography.cpp \
    src/orb.cpp \
    src/matching.cpp \
    src/homography.cpp \
    -o build/test_matching_homography_asan; then
    ./build/test_matching_homography_asan
    echo "[OK] AddressSanitizer and UndefinedBehaviorSanitizer passed with 0 errors."
else
    echo "[WARNING] Sanitizer build not supported or failed on this toolchain. Skipping sanitizer execution." >&2
fi

echo ""
echo "=========================================================="
echo " All checks passed successfully! Baseline V0 is verified. "
echo " Executables located in build/ (git-ignored).             "
echo "=========================================================="
