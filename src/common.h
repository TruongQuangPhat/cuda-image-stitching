// Authors: HUỲNH TẤN PHƯỚC (23120334), integration.
// PPM input/output declarations used by V0.

#pragma once
#include "warping.h"
#include <string>

namespace stitching::io {
warping::RgbImage readPpm(const std::string& path);
void writePpm(const std::string& path, const warping::RgbImage& image);
}  // namespace stitching::io
