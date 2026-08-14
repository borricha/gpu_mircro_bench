#pragma once

#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

#include <cmath>
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
                << __FILE__ << ':' << __LINE__ << ")\\n";                  \
      std::exit(EXIT_FAILURE);                                               \
    }                                                                        \
  } while (0)

namespace l1v2 {

constexpr int kThreads = 512;
constexpr int kLoadsPerPass = 64;
constexpr int kPairsPerPass = kLoadsPerPass / 2;
constexpr int kMaxRounds = 4;
constexpr int kElementsPerSm = kThreads * kLoadsPerPass;
constexpr size_t kWorkingSetBytes =
    static_cast<size_t>(kElementsPerSm) * sizeof(float);  // 128 KiB
constexpr unsigned int kDefaultIterations = 1000000000u / (kElementsPerSm / 2) + 2u;

struct Options {
  int device = 0;
  int activeLoads = kLoadsPerPass;
  unsigned int iterations = kDefaultIterations;
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
  if (options.device < 0 || options.iterations == 0 || options.graphNodes < 1 || options.graphReplays < 1 ||
      options.activeLoads < kLoadsPerPass || options.activeLoads > kMaxRounds * kLoadsPerPass ||
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

// Capture a fixed sequence once and replay it. The NVTX range surrounds only
// graph replay so NSYS can show the kernel spacing inside the measured region.
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
            << " [--device N] [--active-loads 64|128|192|256 | --rounds 1..4]"
            << " [--iterations N] [--input random|zero|one-point-one]\n"
            << description << '\n';
}

}  // namespace l1v2
