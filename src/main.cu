// Authors: HUỲNH TẤN PHƯỚC (23120334), integration.
// CPU entry point for V0, compiled with g++ -x c++.

#include "versions.h"
#include "cuda_utils.cuh"

int main(int argc, char** argv) {
    return stitching::runV0(argc, argv);
}
