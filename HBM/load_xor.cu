// HBM v2 LDG-dominant load-xor differential kernel.
// Each pair of scalar loads feeds one LOP3, so a pass has 64 LDG.E.32 and
// 32 LOP3. NCU verifies these counts at the executed warp-opcode level.
#include "common.cuh"

#ifndef HBM_LOAD_XOR_DEVICE_ONLY
#include <vector>
#endif

namespace hbmv2 {

__device__ __forceinline__ std::uint32_t loadHbmWord(const std::uint32_t* address) {
  std::uint32_t value;
  asm volatile("ld.global.u32 %0, [%1];" : "=r"(value) : "l"(address) : "memory");
  return value;
}

__device__ __forceinline__ std::uint32_t xorThreeWords(
    std::uint32_t state, std::uint32_t first, std::uint32_t second) {
  std::uint32_t result;
  asm volatile("lop3.b32 %0, %1, %2, %3, 0x96;"
               : "=r"(result) : "r"(state), "r"(first), "r"(second));
  return result;
}

// This is the L2-style static offset body. The largest offset is below the
// 128 KiB CTA region, so ptxas does not need a new large-plane base for R2.
template <int Pair, int Remaining>
__device__ __forceinline__ void runStaticLoadXorPairs(
    std::uint32_t& state, const std::uint32_t* regionBase) {
  constexpr size_t offset = static_cast<size_t>(Pair) * kPairStrideWords;
  const auto first = loadHbmWord(regionBase + offset);
  const auto second = loadHbmWord(regionBase + offset + kThreads);
  state = xorThreeWords(state, first, second);
  if constexpr (Remaining > 1)
    runStaticLoadXorPairs<Pair + 1, Remaining - 1>(state, regionBase);
}

template <int ActiveLoads>
__global__ void hbmLoadDominantKernel(const packed_fp32* __restrict__ data,
                                      size_t groups, int innerRepeats) {
  const auto* words = reinterpret_cast<const std::uint32_t*>(data);
  std::uint32_t state = 0x6d2b79f5u ^ threadIdx.x;

#pragma unroll 1
  for (int repeat = 0; repeat < innerRepeats; ++repeat) {
#pragma unroll 1
    for (size_t group = 0; group < groups; ++group) {
      const size_t region = group * gridDim.x + blockIdx.x;
      const auto* regionBase = words + region * kWordsPerRegion + threadIdx.x;
      runStaticLoadXorPairs<0, kLoadPairsPerRound>(state, regionBase);
      if constexpr (ActiveLoads == 2 * kScalarLoadsPerRound)
        runStaticLoadXorPairs<kLoadPairsPerRound, kLoadPairsPerRound>(state, regionBase);
    }
  }
  if (state == 0xffffffffu) asm volatile("trap;");
}

inline void launchLoadDominant(const packed_fp32* data, size_t groups, int blocks,
                               int rounds, int innerRepeats, cudaStream_t stream) {
  if (rounds == 1)
    hbmLoadDominantKernel<kScalarLoadsPerRound><<<blocks, kThreads, 0, stream>>>(
        data, groups, innerRepeats);
  else
    hbmLoadDominantKernel<2 * kScalarLoadsPerRound><<<blocks, kThreads, 0, stream>>>(
        data, groups, innerRepeats);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace hbmv2

#ifndef HBM_LOAD_XOR_DEVICE_ONLY
namespace {
using namespace hbmv2;

void run(const Options& options) {
  CUDA_CHECK(cudaSetDevice(options.device));
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, options.device));
  const size_t bytes = options.mib * 1024ull * 1024ull;
  if (bytes <= static_cast<size_t>(prop.l2CacheSize)) fail("--mib must exceed L2 size");
  const size_t count = bytes / sizeof(packed_fp32);
  const int blocks = prop.multiProcessorCount * options.blocksPerSm;
  if (blocks != kDominantFixedGridBlocks)
    fail("load-xor requires 108 SM x 8 blocks/SM (864 blocks) on this A100");
  const size_t groupBytes = static_cast<size_t>(blocks) * kWordsPerRegion * sizeof(std::uint32_t);
  const size_t groups = bytes / groupBytes;
  if (groups == 0) fail("buffer too small for HBM stream geometry");
  packed_fp32* data = nullptr;
  CUDA_CHECK(cudaMalloc(&data, bytes));
  const int initBlocks = std::min<int>(65535, (count + kThreads - 1) / kThreads);
  initBits<<<initBlocks, kThreads>>>(data, count, inputMode(options.input));
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  const int rounds = options.activeLoads / kScalarLoadsPerRound;
  // Establish page mappings and the steady-state stream before timing.
  launchLoadDominant(data, groups, blocks, rounds, 1, nullptr);
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<double> milliseconds;
  milliseconds.reserve(options.trials);
  cudaEvent_t start{}, stop{};
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  for (int trial = 0; trial < options.trials; ++trial) {
    CUDA_CHECK(cudaEventRecord(start));
    launchLoadDominant(data, groups, blocks, rounds, 1, nullptr);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    milliseconds.push_back(ms);
  }
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  const double best = *std::min_element(milliseconds.begin(), milliseconds.end());
  // A scalar load returns 4 B per lane, i.e. 128 B per warp LDG.E.32.
  const double logicalBytes = static_cast<double>(options.activeLoads) *
      sizeof(std::uint32_t) * static_cast<double>(groups) * blocks * kThreads;
  std::cout << std::fixed << std::setprecision(3)
            << "kernel=hbmLoadDominantKernel rounds=" << rounds
            << " groups=" << groups << " blocks=" << blocks << " kernel_ms=" << best
            << " logical_bytes=" << std::setprecision(0) << logicalBytes
            << " logical_bw_gbs=" << std::setprecision(3)
            << logicalBytes / (best / 1000.0) / 1e9 << '\n';
  CUDA_CHECK(cudaFree(data));
}
}  // namespace

int main(int argc, char** argv) {
  try { run(hbmv2::parseOptions(argc, argv)); return EXIT_SUCCESS; }
  catch (const std::exception& error) {
    if (std::string(error.what()) == "help") {
      hbmv2::printUsage(argv[0],
                         "HBM LDG-dominant: r2-r1 adds 64 scalar LDG + 32 LOP3; "
                         "all other executed SASS opcodes are balanced.");
      return EXIT_SUCCESS;
    }
    std::cerr << "Error: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
#endif  // HBM_LOAD_XOR_DEVICE_ONLY
