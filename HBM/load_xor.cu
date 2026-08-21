// HBM v2 LDG-dominant load-xor differential kernel.
// Each pair of scalar loads feeds one LOP3, so a pass has 64 LDG.E.32 and
// 32 LOP3. NCU verifies these counts at the executed warp-opcode level.
#include "common.cuh"

#ifndef HBM_LOAD_XOR_DEVICE_ONLY
#include <vector>
#endif

namespace hbmv2 {

// The static PTX body avoids an address-calculation instruction between
// individual loads. Each pair is two explicit scalar global loads and one
// LOP3, exactly matching the L1/L2 differential ratio. Do not use v4.u32
// here: ptxas may scalarize a partially consumed vector load.
#define HBM_LOAD_PAIR(first, second) \
  "  ld.global.u32 %0, [%2+" #first "];\n" \
  "  ld.global.u32 %1, [%2+" #second "];\n" \
  "  lop3.b32 %3, %3, %0, %1, 0x96;\n"
#define HBM_LOAD_PASS \
  HBM_LOAD_PAIR(0x0, 0x360000) HBM_LOAD_PAIR(0x6c0000, 0xa20000) \
  HBM_LOAD_PAIR(0xd80000, 0x10e0000) HBM_LOAD_PAIR(0x1440000, 0x17a0000) \
  HBM_LOAD_PAIR(0x1b00000, 0x1e60000) HBM_LOAD_PAIR(0x21c0000, 0x2520000) \
  HBM_LOAD_PAIR(0x2880000, 0x2be0000) HBM_LOAD_PAIR(0x2f40000, 0x32a0000) \
  HBM_LOAD_PAIR(0x3600000, 0x3960000) HBM_LOAD_PAIR(0x3cc0000, 0x4020000) \
  HBM_LOAD_PAIR(0x4380000, 0x46e0000) HBM_LOAD_PAIR(0x4a40000, 0x4da0000) \
  HBM_LOAD_PAIR(0x5100000, 0x5460000) HBM_LOAD_PAIR(0x57c0000, 0x5b20000) \
  HBM_LOAD_PAIR(0x5e80000, 0x61e0000) HBM_LOAD_PAIR(0x6540000, 0x68a0000) \
  HBM_LOAD_PAIR(0x6c00000, 0x6f60000) HBM_LOAD_PAIR(0x72c0000, 0x7620000) \
  HBM_LOAD_PAIR(0x7980000, 0x7ce0000) HBM_LOAD_PAIR(0x8040000, 0x83a0000) \
  HBM_LOAD_PAIR(0x8700000, 0x8a60000) HBM_LOAD_PAIR(0x8dc0000, 0x9120000) \
  HBM_LOAD_PAIR(0x9480000, 0x97e0000) HBM_LOAD_PAIR(0x9b40000, 0x9ea0000) \
  HBM_LOAD_PAIR(0xa200000, 0xa560000) HBM_LOAD_PAIR(0xa8c0000, 0xac20000) \
  HBM_LOAD_PAIR(0xaf80000, 0xb2e0000) HBM_LOAD_PAIR(0xb640000, 0xb9a0000) \
  HBM_LOAD_PAIR(0xbd00000, 0xc060000) HBM_LOAD_PAIR(0xc3c0000, 0xc720000) \
  HBM_LOAD_PAIR(0xca80000, 0xcde0000) HBM_LOAD_PAIR(0xd140000, 0xd4a0000)

#define HBM_ISSUE_LOAD_PASS(state, cursor) do { \
  [[maybe_unused]] std::uint32_t a, b; \
  asm volatile("{\n" HBM_LOAD_PASS "}" \
      : "=&r"(a), "=&r"(b), "+l"(cursor), "+r"(state) \
      : : "memory"); \
  asm volatile("add.u64 %0, %0, 0x0d800000;" : "+l"(cursor)); \
} while (false)

#define HBM_R1_ADD4 \
  "  add.u32 %0, %0, %1;\n" "  add.u32 %0, %0, %1;\n" \
  "  add.u32 %0, %0, %1;\n" "  add.u32 %0, %0, %1;\n"
#define HBM_R1_MAD4 \
  "  mad.lo.u32 %0, %0, 0x5bd1e995, %1;\n" \
  "  mad.lo.u32 %0, %0, 0x5bd1e995, %1;\n" \
  "  mad.lo.u32 %0, %0, 0x5bd1e995, %1;\n" \
  "  mad.lo.u32 %0, %0, 0x5bd1e995, %1;\n"
#define HBM_R1_PADDING(state, seed, rounds) do { \
  asm volatile("{\n" HBM_R1_ADD4 HBM_R1_ADD4 HBM_R1_ADD4 HBM_R1_ADD4 \
               HBM_R1_ADD4 HBM_R1_ADD4 HBM_R1_ADD4 HBM_R1_MAD4 \
               HBM_R1_MAD4 HBM_R1_MAD4 HBM_R1_MAD4 \
               "  .reg .pred r1_pad_pred;\n" \
               "  setp.ge.s32 r1_pad_pred, %2, 0;\n" \
               "  @r1_pad_pred add.u32 %0, %0, %1;\n" \
               "  @r1_pad_pred bra r1_pad_done;\n" \
               "  trap;\n" "r1_pad_done:\n" "}" \
               : "+r"(state) : "r"(seed), "r"(rounds)); \
} while (false)
#define HBM_R2_PADDING(state, seed, rounds) do { \
  asm volatile("{\n" "  .reg .pred r2_pad_pred;\n" \
               "  setp.ge.s32 r2_pad_pred, %1, 0;\n" \
               "  @r2_pad_pred bra r2_pad_done;\n" "  trap;\n" \
               "r2_pad_done:\n" \
               "  mad.lo.u32 %0, %0, 0x5bd1e995, %2;\n" "}" \
               : "+r"(state) : "r"(rounds), "r"(seed)); \
} while (false)

// Core HBM differential kernel. The fixed 108-SM x 8-CTA/SM geometry, loop,
// cursor progression, predicate, and integer padding are shared by R1/R2.
// The only intended variable body is the second 64-LDG.E.32 + 32-LOP3 pass.
__global__ void hbmLoadDominantKernel(const packed_fp32* __restrict__ data,
                                      size_t groups, int rounds,
                                      int innerRepeats) {
  const size_t initial = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  std::uint32_t state = 0x6d2b79f5u ^ threadIdx.x;
  const std::uint32_t seed = 0x9e3779b9u ^ blockIdx.x;

#pragma unroll 1
  for (int repeat = 0; repeat < innerRepeats; ++repeat) {
    const std::uint32_t* cursor =
        reinterpret_cast<const std::uint32_t*>(data) + initial;
#pragma unroll 1
    for (size_t group = 0; group < groups; ++group) {
      HBM_ISSUE_LOAD_PASS(state, cursor);
      if (rounds > 1) {
        HBM_ISSUE_LOAD_PASS(state, cursor);
        HBM_R2_PADDING(state, seed, rounds);
      } else {
        asm volatile("add.u64 %0, %0, 0x0d800000;" : "+l"(cursor));
        HBM_R1_PADDING(state, seed, rounds);
      }
    }
  }
  if (state == 0xffffffffu) asm volatile("trap;");
}

inline void launchLoadDominant(const packed_fp32* data, size_t groups, int blocks,
                               int rounds, int innerRepeats, cudaStream_t stream) {
  hbmLoadDominantKernel<<<blocks, kThreads, 0, stream>>>(
      data, groups, rounds, innerRepeats);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace hbmv2

#undef HBM_LOAD_PAIR
#undef HBM_LOAD_PASS
#undef HBM_ISSUE_LOAD_PASS
#undef HBM_R1_ADD4
#undef HBM_R1_MAD4
#undef HBM_R1_PADDING
#undef HBM_R2_PADDING

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
  const size_t gridStride = static_cast<size_t>(blocks) * kThreads;
  // Reserve space for the complete r2 two-pass stream. Any tail smaller than
  // a complete group is deliberately left untouched.
  const size_t groups = count / (gridStride * kScalarLoadsPerGroup);
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
      sizeof(std::uint32_t) * static_cast<double>(groups) * gridStride;
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
