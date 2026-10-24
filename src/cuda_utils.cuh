// Authors: HUỲNH TẤN PHƯỚC (23120334), integration.
// Compile-time CUDA detection; V0 does not require the CUDA runtime.

#pragma once
namespace stitching {
#if defined(__CUDACC__)
inline constexpr bool compiled_with_cuda = true;
#else
inline constexpr bool compiled_with_cuda = false;
#endif
}  // namespace stitching
