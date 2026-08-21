#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    const cudaError_t error__ = (call);                                      \
    if (error__ != cudaSuccess)                                              \
      throw std::runtime_error(std::string(#call) + ": " +                 \
                               cudaGetErrorString(error__));                 \
  } while (0)

namespace hbmv2 {

using packed_fp32 = uint4;

// A100-40 HBM launch configuration.
constexpr int kThreads = 256;
constexpr int kDefaultBlocksPerSm = 8;
constexpr int kDominantFixedGridBlocks = 864;  // 108 SMs x 8 CTAs/SM.

// One differential pass contains 64 explicit scalar (u32) loads per thread.
// R1/R2 select one/two passes, while a complete stream group reserves both.
constexpr int kScalarLoadsPerRound = 64;
constexpr int kMaxRounds = 2;
constexpr int kScalarLoadsPerGroup = kScalarLoadsPerRound * kMaxRounds;

struct Options {
  int device = 0;
  size_t mib = 1024;
  int blocksPerSm = kDefaultBlocksPerSm;
  int activeLoads = kScalarLoadsPerRound;
  std::string input = "random";
  int trials = 1;
};

[[noreturn]] inline void fail(const std::string& message) {
  throw std::runtime_error(message);
}

inline long long parseInteger(const char* text, const char* name) {
  char* end = nullptr;
  const long long value = std::strtoll(text, &end, 10);
  if (end == nullptr || *end != '\0') fail(std::string("invalid ") + name);
  return value;
}

inline int inputMode(const std::string& input) {
  if (input == "random") return 0;
  if (input == "zero") return 1;
  if (input == "one-point-one") return 2;
  fail("invalid --input (expected random, zero, or one-point-one)");
}

inline Options parseOptions(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto value = [&]() -> const char* {
      if (++i >= argc) fail("missing value for " + arg);
      return argv[i];
    };
    if (arg == "--device") options.device = static_cast<int>(parseInteger(value(), "device"));
    else if (arg == "--mib") options.mib = static_cast<size_t>(parseInteger(value(), "mib"));
    else if (arg == "--blocks-per-sm") options.blocksPerSm = static_cast<int>(parseInteger(value(), "blocks-per-sm"));
    else if (arg == "--active-loads") options.activeLoads = static_cast<int>(parseInteger(value(), "active-loads"));
    else if (arg == "--rounds") options.activeLoads = static_cast<int>(parseInteger(value(), "rounds")) * kScalarLoadsPerRound;
    else if (arg == "--trials") options.trials = static_cast<int>(parseInteger(value(), "trials"));
    else if (arg == "--input") options.input = value();
    else if (arg == "--help" || arg == "-h") fail("help");
    else fail("unknown option: " + arg);
  }
  if (options.device < 0 || options.mib == 0 || options.blocksPerSm < 1 ||
      options.trials < 1 || (options.activeLoads != 64 && options.activeLoads != 128))
    fail("invalid option value");
  inputMode(options.input);
  return options;
}

__device__ __forceinline__ std::uint32_t mixBits(std::uint32_t x) {
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  return x ^ (x >> 16);
}

// Initialize four deterministic FP32 bit patterns per packed uint4 element.
__global__ void initBits(packed_fp32* data, size_t count, int mode) {
  constexpr std::uint32_t kOnePointOne = 0x3f8ccccdu;
  for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count; i += static_cast<size_t>(gridDim.x) * blockDim.x) {
    const auto base = static_cast<std::uint32_t>(i * 4);
    data[i] = mode == 1 ? make_uint4(0, 0, 0, 0)
        : mode == 2 ? make_uint4(kOnePointOne, kOnePointOne, kOnePointOne, kOnePointOne)
                    : make_uint4(mixBits(base), mixBits(base + 1),
                                 mixBits(base + 2), mixBits(base + 3));
  }
}

inline void printUsage(const char* program, const char* description) {
  std::cout << "Usage: " << program
            << " [--device N] [--mib N] [--blocks-per-sm N]"
            << " [--active-loads 64|128 | --rounds 1|2]"
            << " [--trials N] [--input random|zero|one-point-one]\n"
            << description << '\n';
}

}  // namespace hbmv2
