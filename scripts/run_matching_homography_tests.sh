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

RUN_SANITIZERS="${RUN_SANITIZERS:-1}"

# Tracking status of each stage
STATUS_DATASET="PENDING"
STATUS_RELEASE_BUILD="PENDING"
STATUS_RELEASE_TESTS="PENDING"
STATUS_SANITIZERS="PENDING"
OVERALL_STATUS="FAIL"

print_summary() {
    echo ""
    echo "=========================================================="
    echo " Matching & Homography Verification Summary"
    echo "----------------------------------------------------------"
    printf " Dataset preparation : %s\n" "${STATUS_DATASET}"
    printf " Release build       : %s\n" "${STATUS_RELEASE_BUILD}"
    printf " Release tests       : %s\n" "${STATUS_RELEASE_TESTS}"
    printf " Sanitizers          : %s\n" "${STATUS_SANITIZERS}"
    printf " Overall             : %s\n" "${OVERALL_STATUS}"
    echo "=========================================================="
}

# Trap unexpected errors to print summary before exit
trap 'if [ "${OVERALL_STATUS}" != "PASS" ] && [ "${OVERALL_STATUS}" != "PASS (release checks only)" ]; then print_summary; fi' EXIT

mkdir -p build

echo ""
echo "[1/4] Preparing PGM fixtures from dataset/metadata.csv..."
if python3 scripts/prepare_pgm.py; then
    STATUS_DATASET="PASS"
else
    STATUS_DATASET="FAIL"
    exit 1
fi

echo ""
echo "[2/4] Building release test suite (C++17, -Wall -Wextra -Wpedantic -Werror -O2)..."
rm -f build/test_matching_homography
if "${CXX}" -std=c++17 -Wall -Wextra -Wpedantic -Werror -O2 \
    -Isrc \
    tests/test_matching_homography.cpp \
    src/orb.cpp \
    src/matching.cpp \
    src/homography.cpp \
    -o build/test_matching_homography && [ -x build/test_matching_homography ]; then
    STATUS_RELEASE_BUILD="PASS"
else
    STATUS_RELEASE_BUILD="FAIL"
    exit 1
fi

echo ""
echo "[3/4] Running release correctness test suite..."
if ./build/test_matching_homography; then
    STATUS_RELEASE_TESTS="PASS"
else
    STATUS_RELEASE_TESTS="FAIL"
    exit 1
fi

echo ""
if [ "${RUN_SANITIZERS}" = "0" ]; then
    echo "[4/4] Sanitizer checks explicitly skipped (RUN_SANITIZERS=0)."
    STATUS_SANITIZERS="NOT TESTED (explicitly disabled)"
    OVERALL_STATUS="PASS (release checks only)"
    print_summary
    exit 0
else
    echo "[4/4] Building and running with AddressSanitizer & UndefinedBehaviorSanitizer..."
    rm -f build/test_matching_homography_asan
    if ! ("${CXX}" -std=c++17 -Wall -Wextra -Wpedantic -Werror -O1 \
        -fsanitize=address,undefined -g \
        -Isrc \
        tests/test_matching_homography.cpp \
        src/orb.cpp \
        src/matching.cpp \
        src/homography.cpp \
        -o build/test_matching_homography_asan && [ -x build/test_matching_homography_asan ]); then
        STATUS_SANITIZERS="FAIL (compilation failed)"
        exit 1
    fi

    if ! ./build/test_matching_homography_asan; then
        STATUS_SANITIZERS="FAIL (runtime error or sanitizer violation)"
        exit 1
    fi

    STATUS_SANITIZERS="PASS"
    OVERALL_STATUS="PASS"
    print_summary
    exit 0
fi
