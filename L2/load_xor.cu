// L2 v2 differential kernel: preserve the v1 ordinary-global-load mapping
// and add exactly 64 LDG plus 32 LOP3 per extra pass.
#include "common.cuh"

#include <iomanip>

namespace {

using namespace l2v2;

// Keep the v1 ordinary global-load path: do not use .cg or __ldcg to bypass
// L1 allocation. This preserves the path that reached about 5 TB/s in v1.
__device__ __forceinline__ std::uint32_t loadGlobalWord(const float* address) {
  return __float_as_uint(*address);
}

// Keep the consumer pattern identical to L1: combine two loaded FP32 words
// with the running checksum in one three-input LOP3.
__device__ __forceinline__ std::uint32_t xorThreeWords(
    std::uint32_t checksum, std::uint32_t first, std::uint32_t second) {
  std::uint32_t result;
  asm volatile("lop3.b32 %0, %1, %2, %3, 0x96;"
               : "=r"(result) : "r"(checksum), "r"(first), "r"(second));
  return result;
}

// Template parameters make every load offset a compile-time constant. r1 and
// r2 share the same base-address prologue; r2 only appends 32 pairs from this
// static body, so changing active-load count does not add dynamic IMAD work.
template <int Pair, int Remaining>
__device__ __forceinline__ void runStaticLoadXorPairs(
    std::uint32_t& checksum, const float* threadRegionBase) {
  constexpr size_t kPairOffsetElements = static_cast<size_t>(Pair) * kPairStride;
  const std::uint32_t first =
      loadGlobalWord(threadRegionBase + kPairOffsetElements);
  const std::uint32_t second =
      loadGlobalWord(threadRegionBase + kPairOffsetElements + kBlockSize);
  checksum = xorThreeWords(checksum, first, second);
  if constexpr (Remaining > 1)
    runStaticLoadXorPairs<Pair + 1, Remaining - 1>(checksum, threadRegionBase);
}

// Core L2 v2 kernel.
//
// - Data: the v1 16,896 KiB FP32 working set, which fits in A100-40 L2.
// - Grid/block: the v1 200,000 CTA x 1024-thread geometry supplies enough
//   concurrent work to hide L2 latency.
// - The blockIdx.x % 33 region mapping and per-pair stride match v1. Thus the
//   r2 128-load address sequence is the original loadOnlyKernel sequence.
// - r1/r2 differ only in active passes. r2-r1 adds 64 ordinary LDG and 32
//   LOP3 per thread per kernel launch, matching the L1/HBM differential body.
template <int ActiveLoads>
__global__ void l2LoadXorKernel(const float* __restrict__ data,
                                int regionCount) {
  std::uint32_t checksum = 0;
  // Preserve the v1 CTA-to-region mapping while embedding pair offsets in the
  // template body. Each CTA owns one 512 KiB region; 33 regions form the
  // 16,896 KiB working set. The set is L2-resident but exceeds one SM's L1.
  const size_t regionOffset = static_cast<size_t>(blockIdx.x % regionCount) *
      kElementsPerRegion + threadIdx.x;
  const float* threadRegionBase = data + regionOffset;
  runStaticLoadXorPairs<0, kLoadPairsPerPass>(checksum, threadRegionBase);
  if constexpr (ActiveLoads == 2 * kLoadsPerPass)
    runStaticLoadXorPairs<kLoadPairsPerPass, kLoadPairsPerPass>(
        checksum, threadRegionBase);

  // No normal-path store is issued. The trap dependency keeps every load live.
  if (checksum == 0xffffffffu)
    asm volatile("trap;");
}

void launch(const float* data, int blocks, int rounds, cudaStream_t stream) {
  if (rounds < 1 || rounds > kMaxRounds) fail("invalid round count");
  if (rounds == 1)
    l2LoadXorKernel<kLoadsPerPass><<<blocks, kThreads, 0, stream>>>(
        data, kBlockRun);
  else
    l2LoadXorKernel<2 * kLoadsPerPass><<<blocks, kThreads, 0, stream>>>(
        data, kBlockRun);
  CUDA_CHECK(cudaGetLastError());
}

void run(const Options& options) {
  if (options.iterations != 1)
    fail("v1-style L2 kernel fixes --iterations to 1; use CUDA Graph nodes/replays to extend duration");
  CUDA_CHECK(cudaSetDevice(options.device));
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, options.device));
  const int blocks = kLaunchBlocks;
  const int rounds = options.activeLoads / kLoadsPerPass;
  float* data = nullptr;
  CUDA_CHECK(cudaMalloc(&data, kDataBytes));
  // Use the same 52 x 256 initialization geometry as v1.
  initBits<<<kInitBlocks, 256>>>(data, kDataElements, inputMode(options.input));
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  // v1 reports the minimum of repeated in-process launches, therefore it
  // measures after the first launch has filled the 16,896 KiB working set in
  // L2. Run the same kernel outside the timing event to establish that state.
  for (int warmup = 0; warmup < kWarmupLaunches; ++warmup)
    launch(data, blocks, rounds, 0);
  CUDA_CHECK(cudaDeviceSynchronize());

  const float milliseconds = timedLaunchSequence(
      options,
      [&](cudaStream_t stream) {
        launch(data, blocks, rounds, stream);
      },
      "l2v2_load_xor_graph_replay");
  const double issuedLoads = static_cast<double>(options.activeLoads) *
      options.iterations * blocks * kThreads * launchMultiplier(options);
  const double logicalBytes = issuedLoads * sizeof(float);
  const double bandwidth = logicalBytes / (milliseconds / 1000.0) / 1.0e9;
  std::cout << std::fixed << std::setprecision(3)
            << "kernel=l2LoadXorKernel rounds=" << rounds
            << " iterations=" << options.iterations
            << " launch_mode=" << options.launchMode
            << " graph_nodes=" << options.graphNodes
            << " graph_replays=" << options.graphReplays
            << " sm_count=" << prop.multiProcessorCount
            << " blocks=" << blocks
            << " data_kib=" << (kDataBytes / 1024)
            << " kernel_ms=" << milliseconds
            << " logical_bytes=" << std::setprecision(0) << logicalBytes
            << " logical_bw_gbs=" << std::setprecision(3) << bandwidth << '\n';

  CUDA_CHECK(cudaFree(data));
}

}  // namespace

int main(int argc, char** argv) {
  try {
    run(l2v2::parseOptions(argc, argv));
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    if (std::string(error.what()) == "help") {
      l2v2::printCommonUsage(argv[0],
          "v1-style L2 path: ordinary global LDG; each pass adds 64 LDG + 32 LOP3.");
      return EXIT_SUCCESS;
    }
    std::cerr << "Error: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
