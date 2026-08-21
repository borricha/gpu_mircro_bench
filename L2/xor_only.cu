// L2 v2 differential control: keep v1 launch/address arithmetic, omit LDG.
#include "common.cuh"

#include <iomanip>

namespace {

using namespace l2v2;

// Emit one 3-input XOR as one LOP3 instruction.
__device__ __forceinline__ std::uint32_t xorThreeWords(
    std::uint32_t a, std::uint32_t b, std::uint32_t c) {
  std::uint32_t result;
  asm volatile("lop3.b32 %0, %1, %2, %3, 0x96;"
               : "=r"(result) : "r"(a), "r"(b), "r"(c));
  return result;
}

// Compile-time recursion grows only the LOP3 control body. r1 and r2 share
// the same launch and address prologue; only this body's LOP3 count grows
// from 32 to 64.
template <int Slot, int Remaining>
__device__ __forceinline__ void runStaticXorSlots(std::uint32_t& checksum,
                                                   std::uint32_t addressLo,
                                                   std::uint32_t addressHi) {
  // These are the low/high halves of a real 64-bit global pointer. Using both
  // keeps the load-xor pointer-address prologue, including LEA.HI.X, live.
  checksum = xorThreeWords(checksum, addressLo, addressHi);
  if constexpr (Remaining > 1)
    runStaticXorSlots<Slot + 1, Remaining - 1>(checksum, addressLo, addressHi);
}

// As in load-xor, specialize r1 and r2 as separate SASS kernels. Selecting a
// runtime `if (rounds == 2)` inside the kernel could change BRA predicates
// between conditions, so the host selects the specialized kernel instead.
template <int Rounds>
__global__ void l2XorOnlyKernel(const float* __restrict__ data, int regionCount) {
  static_assert(Rounds == 1 || Rounds == 2, "invalid XOR-only round count");
  std::uint32_t checksum = 0;
  // Reproduce the load-xor CTA-to-512 KiB-region address calculation exactly.
  // The pointer-derived XOR inputs keep blockIdx %, IMAD/LEA, and the full
  // pointer prologue live without issuing any global-memory load.
  const size_t regionOffset = static_cast<size_t>(blockIdx.x % regionCount) *
      kElementsPerRegion + threadIdx.x;
  const float* threadRegionBase = data + regionOffset;
  // Feed the real pointer halves to each LOP3. Unlike an empty volatile asm,
  // this checksum dependency prevents NVCC from removing the v1-style region
  // mapping and 64-bit pointer-address arithmetic.
  const std::uintptr_t baseWord = reinterpret_cast<std::uintptr_t>(threadRegionBase);
  const std::uint32_t addressLo = static_cast<std::uint32_t>(baseWord);
  const std::uint32_t addressHi = static_cast<std::uint32_t>(baseWord >> 32);
  runStaticXorSlots<0, kLoadPairsPerPass>(checksum, addressLo, addressHi);
  if constexpr (Rounds == 2)
    runStaticXorSlots<kLoadPairsPerPass, kLoadPairsPerPass>(
        checksum, addressLo, addressHi);
  // No normal-path store is issued, but the trap dependency keeps every XOR live.
  if (checksum == 0xffffffffu)
    asm volatile("trap;");
}

void launch(const float* data, int blocks, int rounds, cudaStream_t stream) {
  if (rounds < 1 || rounds > kMaxRounds) fail("invalid round count");
  if (rounds == 1)
    l2XorOnlyKernel<1><<<blocks, kThreads, 0, stream>>>(data, kBlockRun);
  else
    l2XorOnlyKernel<2><<<blocks, kThreads, 0, stream>>>(data, kBlockRun);
  CUDA_CHECK(cudaGetLastError());
}

void run(const Options& options) {
  if (options.iterations != 1)
    fail("v1-style L2 kernels fix --iterations to 1; use CUDA Graph nodes/replays to extend duration");
  CUDA_CHECK(cudaSetDevice(options.device));
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, options.device));
  const int blocks = kLaunchBlocks;
  const int rounds = options.activeLoads / kLoadsPerPass;
  float* data = nullptr;
  CUDA_CHECK(cudaMalloc(&data, kDataBytes));
  initBits<<<kInitBlocks, 256>>>(data, kDataElements, inputMode(options.input));
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  // Match load-xor's launch and clock warm-up count. The allocation is still
  // retained for an identical pointer prologue, although this kernel issues no
  // global-memory data load.
  for (int warmup = 0; warmup < kWarmupLaunches; ++warmup)
    launch(data, blocks, rounds, 0);
  CUDA_CHECK(cudaDeviceSynchronize());

  const float milliseconds = timedLaunchSequence(
      options,
      [&](cudaStream_t stream) { launch(data, blocks, rounds, stream); },
      "l2v2_xor_only_graph_replay");
  // SASS folds this operation into one 3-input LOP3. The NCU full report should
  // show only a 32-LOP3 r1-to-r2 increment, with BRA/IMAD/ISETP unchanged.
  const double issuedLop3 = static_cast<double>(rounds * kLoadPairsPerPass) *
      blocks * kThreads * launchMultiplier(options);
  std::cout << std::fixed << std::setprecision(3)
            << "kernel=l2XorOnlyKernel rounds=" << rounds
            << " iterations=" << options.iterations
            << " launch_mode=" << options.launchMode
            << " graph_nodes=" << options.graphNodes
            << " graph_replays=" << options.graphReplays
            << " sm_count=" << prop.multiProcessorCount
            << " blocks=" << blocks
            << " kernel_ms=" << milliseconds
            << " logical_lop3=" << std::setprecision(0) << issuedLop3 << '\n';

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
          "L2 XOR-only control: no LDG; r2-r1 adds exactly 32 LOP3 per thread.");
      return EXIT_SUCCESS;
    }
    std::cerr << "Error: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
