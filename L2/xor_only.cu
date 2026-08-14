// L2 v2 differential control: keep v1 launch/address arithmetic, omit LDG.
#include "common.cuh"

#include <iomanip>

namespace {

using namespace l2v2;

__device__ __forceinline__ std::uint32_t xor3Bits(
    std::uint32_t a, std::uint32_t b, std::uint32_t c) {
  std::uint32_t result;
  asm volatile("lop3.b32 %0, %1, %2, %3, 0x96;"
               : "=r"(result) : "r"(a), "r"(b), "r"(c));
  return result;
}

// Compile-time recursion으로 LOP3만 늘리는 control body. r1/r2는 같은 launch와
// 같은 prologue를 공유하며, template body의 LOP3 수만 32 → 64로 바뀐다.
template <int Slot, int Remaining>
__device__ __forceinline__ void staticXorSlots(std::uint32_t& checksum,
                                               std::uint32_t addressLo,
                                               std::uint32_t addressHi) {
  // 둘 모두 실제 64-bit global pointer의 half다. 이렇게 해야 LEA.HI.X를 포함한
  // load-xor의 전체 pointer-address prologue도 live 상태로 유지된다.
  checksum = xor3Bits(checksum, addressLo, addressHi);
  if constexpr (Remaining > 1)
    staticXorSlots<Slot + 1, Remaining - 1>(checksum, addressLo, addressHi);
}

// load-xor와 마찬가지로 r1/r2를 별도 SASS kernel로 특수화한다. runtime
// "if (rounds == 2)"를 kernel 안에 두면 BRA의 predicate 상태가 두 조건에서
// 달라질 수 있으므로, host에서만 kernel을 선택한다.
template <int Rounds>
__global__ void l2XorOnlyKernel(const float* __restrict__ data, int blockRun) {
  static_assert(Rounds == 1 || Rounds == 2, "invalid XOR-only round count");
  std::uint32_t checksum = 0;
  // load-xor와 byte-for-byte 같은 CTA→512 KiB-region 주소 계산을 수행한다.
  // empty volatile asm의 pointer input은 이 계산을 dead-code 제거하지 못하게 한다.
  // 따라서 두 kernel의 blockIdx %, IMAD/LEA 및 pointer prologue를 공통으로 만든다.
  const size_t blockOffset = static_cast<size_t>(blockIdx.x % blockRun) *
      kElementsPerRegion + threadIdx.x;
  const float* base = data + blockOffset;
  // 각 LOP3의 입력으로 실제 base pointer의 하위 word를 쓴다. 단순한 empty asm과
  // 달리 이 값은 checksum에 data-dependence가 있으므로, NVCC가 v1과 동일한
  // blockIdx % / 64-bit pointer-address 계산을 지울 수 없다.
  const std::uintptr_t baseWord = reinterpret_cast<std::uintptr_t>(base);
  const std::uint32_t addressLo = static_cast<std::uint32_t>(baseWord);
  const std::uint32_t addressHi = static_cast<std::uint32_t>(baseWord >> 32);
  staticXorSlots<0, kLoadsPerPass / 2>(checksum, addressLo, addressHi);
  if constexpr (Rounds == 2)
    staticXorSlots<kLoadsPerPass / 2, kLoadsPerPass / 2>(checksum, addressLo, addressHi);
  // normal path에는 store가 없지만, trap dependency가 모든 XOR를 live로 유지한다.
  if (checksum == 0xffffffffu)
    asm volatile("trap;");
}

void launch(const float* data, int blocks, unsigned int /* iterations */, int rounds,
            cudaStream_t stream) {
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
  initBits<<<52, 256>>>(data, kDataElements, inputMode(options.input));
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  // load-xor와 같은 launch/clock warm-up 횟수를 사용한다. L2 data allocation은
  // control kernel에 불필요하며, 이 control에는 global-memory traffic이 없다.
  for (int warmup = 0; warmup < 15; ++warmup)
    launch(data, blocks, options.iterations, rounds, 0);
  CUDA_CHECK(cudaDeviceSynchronize());

  const float milliseconds = timedLaunchSequence(
      options,
      [&](cudaStream_t stream) { launch(data, blocks, options.iterations, rounds, stream); },
      "l2v2_xor_only_graph_replay");
  // SASS는 두 XOR source operation을 한 3-input LOP3로 fold한다. NCU full report
  // 에서 r1/r2는 LOP3만 32개 증가하고, BRA/IMAD/ISETP는 동일해야 한다.
  const double issuedLop3 = static_cast<double>(rounds * (kLoadsPerPass / 2)) *
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
