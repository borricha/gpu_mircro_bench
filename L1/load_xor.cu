// L1 v2 load slope kernel: one extra pass is exactly 64 LDG + 32 LOP3.
#include "common.cuh"

#include <iomanip>

namespace {

using namespace l1v2;

// `ca`(cache all)는 global load의 반환값을 L1/TEX에 캐시하도록 요청한다.
// float 연산으로 해석하지 않고 32-bit bit pattern 그대로 받아 data bit pattern의
// switching 효과도 포함해 측정한다.
__device__ __forceinline__ std::uint32_t loadCacheAllBits(const float* address) {
  std::uint32_t value;
  asm volatile("ld.global.ca.u32 %0, [%1];"
               : "=r"(value) : "l"(address) : "memory");
  return value;
}

// checksum의 의존성을 만드는 3-input XOR이다. LOP3 한 개가 발행되며, load 결과가
// 이 연산의 입력이므로 컴파일러가 앞선 load를 dead-code로 지울 수 없다.
__device__ __forceinline__ std::uint32_t xor3Bits(
    std::uint32_t a, std::uint32_t b, std::uint32_t c) {
  std::uint32_t result;
  asm volatile("lop3.b32 %0, %1, %2, %3, 0x96;"
               : "=r"(result) : "r"(a), "r"(b), "r"(c));
  return result;
}

// 한 pass = thread당 load 64개 + LOP3 32개.
// Pair 하나가 서로 64 KiB 떨어진 두 FP32 word를 읽고 LOP3 하나로 소비한다.
// template 재귀는 컴파일 시 완전히 펼쳐져 loop index 계산이 hot path에 남지 않는다.
template <int Pair, int Remaining>
__device__ __forceinline__ void loadXorPass(std::uint32_t& checksum,
                                            const float* B) {
  constexpr int kOffset = Pair * kThreads;
  const std::uint32_t a = loadCacheAllBits(B + kOffset);
  const std::uint32_t b = loadCacheAllBits(B + kElementsPerSm / 2 + kOffset);
  checksum = xor3Bits(checksum, a, b);
  if constexpr (Remaining > 1)
    loadXorPass<Pair + 1, Remaining - 1>(checksum, B);
}

// L1 v2의 핵심 kernel.
//
// - grid: SM당 block 하나, block: 512 threads (= 16 warps)
// - data: 128 KiB working set. 각 block은 같은 data를 읽으므로 warm-up 뒤 L1 hit를
//   반복적으로 발생시킨다.
// - rounds=1(r1)이면 pass 1개, rounds=2(r2)이면 pass 2개를 실행한다.
//   따라서 r2-r1의 active body 증분은 thread당 iteration마다 LDG 64개와 LOP3 32개다.
//
// pass loop는 항상 kMaxRounds번의 uniform branch를 실행한다. xor-only control
// kernel도 동일한 branch 구조를 가지므로, 두 slope를 빼 LDG를 분리할 때 branch
// 관련 에너지는 상쇄된다.
__global__ void l1LoadXorKernel(std::uint32_t* __restrict__ checksums,
                                const float* __restrict__ data, unsigned int iterations,
                                int zero, int rounds) {
  // 같은 warp의 lane은 연속된 word를 읽어 coalesced 32-B sector 요청을 만든다.
  const float* B0 = data + threadIdx.x;
  std::uint32_t checksum = 0;
#pragma unroll 1
  for (unsigned int iteration = 0; iteration < iterations; ++iteration) {
    // runtime 값 zero(항상 0)를 더해 주소가 컴파일 타임 상수가 되지 않게 만든다.
    // 실제 주소/traffic은 변하지 않는다.
    B0 += zero;
#pragma unroll 1
    for (int pass = 0; pass < kMaxRounds; ++pass) {
      if (rounds > pass) loadXorPass<0, kPairsPerPass>(checksum, B0);
    }
  }
  // 정상 입력에서는 false라 store traffic은 없다. 하지만 checksum이 observable이어서
  // LDG와 LOP3 체인이 제거되지 않는다.
  if (checksum == 0xffffffffu)
    checksums[blockIdx.x * blockDim.x + threadIdx.x] = checksum;
}

void launch(std::uint32_t* checksums, const float* data, int blocks,
            unsigned int iterations, int rounds, cudaStream_t stream) {
  if (rounds < 1 || rounds > kMaxRounds) fail("invalid round count");
  l1LoadXorKernel<<<blocks, kThreads, 0, stream>>>(checksums, data, iterations, 0, rounds);
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
  initBits<<<52, 256>>>(data, kElementsPerSm, inputMode(options.input));
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
