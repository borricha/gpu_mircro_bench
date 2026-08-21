// HBM v2 LOP3-only control. R1/R2 differ by exactly one 32-LOP3 pass;
// it is the control counterpart of load_xor's extra 64 scalar-LDG pass.
#include "common.cuh"

#ifndef HBM_XOR_ONLY_DEVICE_ONLY
#include <vector>
#endif

namespace hbmv2 {

#define HBM_XOR4 \
  "  lop3.b32 %0, %0, %1, %2, 0x96;\n" \
  "  lop3.b32 %1, %1, %2, %3, 0x96;\n" \
  "  lop3.b32 %2, %2, %3, %0, 0x96;\n" \
  "  lop3.b32 %3, %3, %0, %1, 0x96;\n"
#define HBM_XOR8 HBM_XOR4 HBM_XOR4
#define HBM_XOR16 HBM_XOR8 HBM_XOR8
#define HBM_XOR32 HBM_XOR16 HBM_XOR16
#define HBM_ISSUE_XOR_PASS(x, y, z, w) \
  asm volatile("{\n" HBM_XOR32 "}" : "+r"(x), "+r"(y), "+r"(z), "+r"(w))
// LOP3-only control for the L2-style HBM mapping. It has the same repeat and
// group loops as hbmLoadDominantKernel; R2 only appends one static LOP3 pass.
template <int ActiveLoads>
__global__ void hbmXorOnlyKernel(size_t groups, int innerRepeats) {
  const std::uint32_t lane = threadIdx.x;
  std::uint32_t x = lane ^ 0x12345678u;
  std::uint32_t y = lane ^ 0x9abcdef0u;
  std::uint32_t z = lane ^ 0x31415926u;
  std::uint32_t w = lane ^ 0x27182818u;

#pragma unroll 1
  for (int repeat = 0; repeat < innerRepeats; ++repeat) {
#pragma unroll 1
    for (size_t group = 0; group < groups; ++group) {
      HBM_ISSUE_XOR_PASS(x, y, z, w);
      if constexpr (ActiveLoads == 2 * kScalarLoadsPerRound) {
        HBM_ISSUE_XOR_PASS(x, y, z, w);
      }
    }
  }
  if ((x & y & z & w) == 0xffffffffu) asm volatile("trap;");
}

inline void launchXorOnly(size_t groups, int blocks, int rounds,
                          int innerRepeats, cudaStream_t stream) {
  if (rounds == 1)
    hbmXorOnlyKernel<kScalarLoadsPerRound><<<blocks, kThreads, 0, stream>>>(
        groups, innerRepeats);
  else
    hbmXorOnlyKernel<2 * kScalarLoadsPerRound><<<blocks, kThreads, 0, stream>>>(
        groups, innerRepeats);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace hbmv2

#undef HBM_XOR4
#undef HBM_XOR8
#undef HBM_XOR16
#undef HBM_XOR32
#undef HBM_ISSUE_XOR_PASS

#ifndef HBM_XOR_ONLY_DEVICE_ONLY
namespace {
using namespace hbmv2;

void run(const Options& options) {
  CUDA_CHECK(cudaSetDevice(options.device));
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, options.device));
  const size_t bytes = options.mib * 1024ull * 1024ull;
  if (bytes <= static_cast<size_t>(prop.l2CacheSize)) fail("--mib must exceed L2 size");
  const int blocks = prop.multiProcessorCount * options.blocksPerSm;
  const size_t groupBytes = static_cast<size_t>(blocks) * kWordsPerRegion * sizeof(std::uint32_t);
  const size_t groups = bytes / groupBytes;
  if (groups == 0) fail("buffer too small for HBM stream geometry");
  const int rounds = options.activeLoads / kScalarLoadsPerRound;

  // Keep the allocation/context footprint equal to load_xor. This buffer is
  // deliberately untouched by the timed XOR-only kernel.
  packed_fp32* footprint = nullptr;
  CUDA_CHECK(cudaMalloc(&footprint, bytes));
  launchXorOnly(groups, blocks, rounds, 1, nullptr);
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<double> milliseconds;
  milliseconds.reserve(options.trials);
  cudaEvent_t start{}, stop{};
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  for (int trial = 0; trial < options.trials; ++trial) {
    CUDA_CHECK(cudaEventRecord(start));
    launchXorOnly(groups, blocks, rounds, 1, nullptr);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    milliseconds.push_back(ms);
  }
  CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop));
  const double best = *std::min_element(milliseconds.begin(), milliseconds.end());
  const double logicalLop3 = static_cast<double>(options.activeLoads) / 2.0 *
      static_cast<double>(groups) * blocks * kThreads;
  std::cout << std::fixed << std::setprecision(3)
            << "kernel=hbmXorOnlyKernel rounds=" << rounds
            << " groups=" << groups << " blocks=" << blocks << " kernel_ms=" << best
            << " logical_lop3=" << std::setprecision(0) << logicalLop3 << '\n';
  CUDA_CHECK(cudaFree(footprint));
}
}  // namespace

int main(int argc, char** argv) {
  try { run(hbmv2::parseOptions(argc, argv)); return EXIT_SUCCESS; }
  catch (const std::exception& error) {
    if (std::string(error.what()) == "help") {
      hbmv2::printUsage(argv[0], "HBM xor-only: r2-r1 adds exactly 32 LOP3 per thread/group.");
      return EXIT_SUCCESS;
    }
    std::cerr << "Error: " << error.what() << '\n'; return EXIT_FAILURE;
  }
}
#endif  // HBM_XOR_ONLY_DEVICE_ONLY
