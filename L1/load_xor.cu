// L1 v2 differential load kernel: one extra pass issues 64 LDG and 32 LOP3.
#include "common.cuh"

#include <iomanip>

namespace {

using namespace l1v2;

constexpr int kInitBlocks = 52;

// `ca` (cache all) requests that the global-load result be cached in L1/TEX.
// Keep the result as raw bits rather than interpreting it as FP32 so the
// experiment retains data-pattern-dependent switching behavior without FP work.
__device__ __forceinline__ std::uint32_t loadL1CachedWord(const float* address) {
  std::uint32_t value;
  asm volatile("ld.global.ca.u32 %0, [%1];"
               : "=r"(value) : "l"(address) : "memory");
  return value;
}

// One 3-input XOR maps to one LOP3. Because the load results feed this value,
// the compiler cannot eliminate the preceding loads as dead code.
__device__ __forceinline__ std::uint32_t xorThreeWords(
    std::uint32_t a, std::uint32_t b, std::uint32_t c) {
  std::uint32_t result;
  asm volatile("lop3.b32 %0, %1, %2, %3, 0x96;"
               : "=r"(result) : "r"(a), "r"(b), "r"(c));
  return result;
}

// One pass performs 64 loads and 32 LOP3 operations per thread. Each pair
// reads two FP32 words 64 KiB apart and consumes them with one LOP3. Template
// recursion fully unrolls the 32 pairs, keeping pair-loop index arithmetic out
// of the hot path.
template <int Pair, int Remaining>
__device__ __forceinline__ void runLoadXorPass(std::uint32_t& checksum,
                                                const float* threadBase) {
  constexpr int kPairOffsetElements = Pair * kThreads;
  constexpr int kHalfWorkingSetElements = kElementsPerSm / 2;
  const std::uint32_t lowerWord =
      loadL1CachedWord(threadBase + kPairOffsetElements);
  const std::uint32_t upperWord =
      loadL1CachedWord(threadBase + kHalfWorkingSetElements + kPairOffsetElements);
  checksum = xorThreeWords(checksum, lowerWord, upperWord);
  if constexpr (Remaining > 1)
    runLoadXorPass<Pair + 1, Remaining - 1>(checksum, threadBase);
}

// Core L1 v2 kernel.
//
// - Grid: one block per SM. Block: 512 threads (16 warps).
// - Data: a 128 KiB working set. Every block reads the same allocation, so
//   each SM repeatedly hits its own L1 after warm-up.
// - rounds=1 (r1) executes one pass; rounds=2 (r2) executes two passes.
//   Therefore r2-r1 adds exactly 64 LDG and 32 LOP3 per thread per iteration.
//
// The pass loop always evaluates kMaxRounds uniform branches. The xor-only
// control kernel has the same branch structure, so subtracting its slope
// removes the branch/control contribution during LDG attribution.
__global__ void l1LoadXorKernel(std::uint32_t* __restrict__ checksums,
                                const float* __restrict__ data, unsigned int iterations,
                                int runtimeZero, int rounds) {
  // Lanes in a warp start at consecutive words, producing coalesced sectors.
  const float* threadBase = data + threadIdx.x;
  std::uint32_t checksum = 0;
#pragma unroll 1
  for (unsigned int iteration = 0; iteration < iterations; ++iteration) {
    // A runtime value that is always zero prevents the address from becoming
    // a compile-time constant without changing the address or traffic.
    threadBase += runtimeZero;
#pragma unroll 1
    for (int pass = 0; pass < kMaxRounds; ++pass) {
      if (rounds > pass) runLoadXorPass<0, kPairsPerPass>(checksum, threadBase);
    }
  }
  // This condition is false in normal operation, so no store traffic is
  // expected. It still makes checksum observable and preserves the LDG/LOP3 chain.
  if (checksum == 0xffffffffu)
    checksums[blockIdx.x * blockDim.x + threadIdx.x] = checksum;
}

void launch(std::uint32_t* checksums, const float* data, int blocks,
            unsigned int iterations, int rounds, cudaStream_t stream) {
  if (rounds < 1 || rounds > kMaxRounds) fail("invalid round count");
  constexpr int kRuntimeZero = 0;
  l1LoadXorKernel<<<blocks, kThreads, 0, stream>>>(
      checksums, data, iterations, kRuntimeZero, rounds);
  CUDA_CHECK(cudaGetLastError());
}

void run(const Options& options) {
  CUDA_CHECK(cudaSetDevice(options.device));
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, options.device));
  const int blocks = prop.multiProcessorCount;
  const int rounds = options.activeLoads / kLoadsPerPass;
  float* data = nullptr;
  std::uint32_t* checksums = nullptr;
  CUDA_CHECK(cudaMalloc(&data, kWorkingSetBytes));
  CUDA_CHECK(cudaMalloc(&checksums, static_cast<size_t>(blocks) * kThreads * sizeof(std::uint32_t)));
  initBits<<<kInitBlocks, 256>>>(data, kElementsPerSm, inputMode(options.input));
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  const float milliseconds = timedLaunchSequence(
      options,
      [&](cudaStream_t stream) {
        launch(checksums, data, blocks, options.iterations, rounds, stream);
      },
      "l1v2_load_xor_graph_replay");

  const double issuedLoads = static_cast<double>(options.activeLoads) *
      options.iterations * blocks * kThreads * launchMultiplier(options);
  const double logicalBytes = issuedLoads * sizeof(float);
  const double bandwidth = logicalBytes / (milliseconds / 1000.0) / 1.0e9;
  std::cout << std::fixed << std::setprecision(3)
            << "kernel=l1LoadXorKernel rounds=" << rounds
            << " iterations=" << options.iterations
            << " launch_mode=" << options.launchMode
            << " graph_nodes=" << options.graphNodes
            << " graph_replays=" << options.graphReplays
            << " sm_count=" << blocks
            << " kernel_ms=" << milliseconds
            << " logical_bytes=" << std::setprecision(0) << logicalBytes
            << " logical_bw_gbs=" << std::setprecision(3) << bandwidth << '\n';

  CUDA_CHECK(cudaFree(checksums));
  CUDA_CHECK(cudaFree(data));
}

}  // namespace

int main(int argc, char** argv) {
  try {
    run(l1v2::parseOptions(argc, argv));
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    if (std::string(error.what()) == "help") {
      l1v2::printCommonUsage(argv[0], "load-xor: each pass adds 64 LDG + 32 LOP3.");
      return EXIT_SUCCESS;
    }
    std::cerr << "Error: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
