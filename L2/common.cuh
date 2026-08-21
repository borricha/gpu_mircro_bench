#pragma once

#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    const cudaError_t error__ = (call);                                      \
    if (error__ != cudaSuccess) {                                            \
      std::cerr << "CUDA error: " << cudaGetErrorString(error__) << " ("    \
                << __FILE__ << ':' << __LINE__ << ")\n";                  \
      std::exit(EXIT_FAILURE);                                               \
    }                                                                        \
  } while (0)

namespace l2v2 {

// Preserve the v1 launch geometry: 200,000 CTAs x 1024 threads. One kernel
// launch already supplies enough CTAs to hide L2 latency, so this benchmark
// does not artificially constrain blocks per SM.
constexpr int kThreads = 1024;
constexpr int kLaunchBlocks = 200000;
constexpr int kLoadsPerPass = 64;
constexpr int kLoadPairsPerPass = kLoadsPerPass / 2;
constexpr int kMaxRounds = 2;
constexpr int kInitBlocks = 52;
constexpr int kWarmupLaunches = 15;

// Preserve the v1 storage and CTA-to-region mapping:
// N=128, block size=1024, region count=33 => 33 x 512 KiB = 16,896 KiB.
// This fits in an A100-40's L2 but is far larger than one SM's L1/TEX cache.
constexpr int kBlockRun = 33;
constexpr int kBlockSize = 1024;
constexpr int kN = 128;
constexpr size_t kDataBytes =
    static_cast<size_t>(kBlockRun) * kBlockSize * kN * sizeof(float);
constexpr size_t kDataElements = kDataBytes / sizeof(float);
// FP32 element stride between consecutive v1 inner-loop indices. Within one
// 512 KiB region, 64 pairs of 4 B loads cover the full region; every offset is
// a compile-time immediate in the generated SASS.
constexpr int kPairStride = kThreads * 2;
constexpr int kElementsPerRegion = kN * kBlockSize;
static_assert(kElementsPerRegion * sizeof(float) == 512 * 1024,
              "each L2 CTA region must be 512 KiB");
static_assert(kThreads == kBlockSize, "v1 mapping assumes one 1024-thread CTA");
static_assert(kDataBytes == 16896ull * 1024,
              "v1 L2 working-set size must remain 16,896 KiB");

struct Options {
  int device = 0;
  int activeLoads = kLoadsPerPass;
  // One v1-style 200,000-CTA launch is already about 20 ms and saturates L2.
  // Unlike L1 v2, this benchmark does not need an in-kernel iteration loop.
  unsigned int iterations = 1;
  std::string input = "random";
  std::string launchMode = "single";
  int graphNodes = 1;
  int graphReplays = 1;
};

[[noreturn]] inline void fail(const std::string& message) {
  throw std::runtime_error(message);
}

inline long long parseInteger(const char* value, const char* name) {
  char* end = nullptr;
  const long long parsed = std::strtoll(value, &end, 10);
  if (end == nullptr || *end != '\0') fail(std::string("invalid ") + name + ": " + value);
  return parsed;
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
    else if (arg == "--active-loads") options.activeLoads = static_cast<int>(parseInteger(value(), "active-loads"));
    else if (arg == "--rounds") options.activeLoads = static_cast<int>(parseInteger(value(), "rounds")) * kLoadsPerPass;
    else if (arg == "--iterations") options.iterations = static_cast<unsigned int>(parseInteger(value(), "iterations"));
    else if (arg == "--input") options.input = value();
    else if (arg == "--launch-mode") options.launchMode = value();
    else if (arg == "--graph-nodes") options.graphNodes = static_cast<int>(parseInteger(value(), "graph-nodes"));
    else if (arg == "--graph-replays") options.graphReplays = static_cast<int>(parseInteger(value(), "graph-replays"));
    else if (arg == "--help" || arg == "-h") fail("help");
    else fail("unknown option: " + arg);
  }
  if (options.device < 0 || options.iterations == 0 || options.graphNodes < 1 ||
      options.graphReplays < 1 || options.activeLoads < kLoadsPerPass ||
      options.activeLoads > kMaxRounds * kLoadsPerPass ||
      options.activeLoads % kLoadsPerPass != 0)
    fail("invalid option value");
  if (options.launchMode != "single" && options.launchMode != "graph")
    fail("invalid --launch-mode (expected single or graph)");
  inputMode(options.input);
  return options;
}

inline unsigned long long launchMultiplier(const Options& options) {
  return options.launchMode == "graph"
      ? static_cast<unsigned long long>(options.graphNodes) * options.graphReplays
      : 1ull;
}

template <typename Enqueue>
inline float timedLaunchSequence(const Options& options, Enqueue enqueue,
                                 const char* nvtxRangeName) {
  cudaStream_t stream{};
  cudaEvent_t start{}, stop{};
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  if (options.launchMode == "single") {
    CUDA_CHECK(cudaEventRecord(start, stream));
    enqueue(stream);
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
  } else {
    cudaGraph_t graph{};
    cudaGraphExec_t graphExec{};
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    for (int node = 0; node < options.graphNodes; ++node) enqueue(stream);
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
    nvtxRangePushA(nvtxRangeName);
    CUDA_CHECK(cudaEventRecord(start, stream));
    for (int replay = 0; replay < options.graphReplays; ++replay)
      CUDA_CHECK(cudaGraphLaunch(graphExec, stream));
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    nvtxRangePop();
    CUDA_CHECK(cudaGraphExecDestroy(graphExec));
    CUDA_CHECK(cudaGraphDestroy(graph));
  }
  float milliseconds = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaStreamDestroy(stream));
  return milliseconds;
}

__device__ __forceinline__ std::uint32_t mixBits(std::uint32_t x) {
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  return x ^ (x >> 16);
}

__global__ void initBits(float* data, size_t count, int mode) {
  constexpr std::uint32_t kOnePointOne = 0x3f8ccccdu;
  auto* words = reinterpret_cast<std::uint32_t*>(data);
  for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count; i += static_cast<size_t>(gridDim.x) * blockDim.x) {
    words[i] = mode == 1 ? 0u : (mode == 2 ? kOnePointOne
                                             : mixBits(static_cast<std::uint32_t>(i)));
  }
}

inline void printCommonUsage(const char* program, const char* description) {
  std::cout << "Usage: " << program
            << " [--device N] [--active-loads 64|128 | --rounds 1|2]"
            << " [--iterations N] [--input random|zero|one-point-one]"
            << " [--launch-mode single|graph] [--graph-nodes N] [--graph-replays N]\n"
            << description << '\n';
}

}  // namespace l2v2
