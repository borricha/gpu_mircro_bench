// L1 v2 control slope kernel: one extra pass is exactly 32 LOP3.
#include "common.cuh"

#include <iomanip>

namespace {

using namespace l1v2;

struct XorWords {
  std::uint32_t a;
};

// load 없이 LOP3 자체의 증분 에너지를 재기 위한 동일한 XOR primitive.
__device__ __forceinline__ std::uint32_t xor3Bits(
    std::uint32_t a, std::uint32_t b, std::uint32_t c) {
  std::uint32_t result;
  asm volatile("lop3.b32 %0, %1, %2, %3, 0x96;"
               : "=r"(result) : "r"(a), "r"(b), "r"(c));
  return result;
}

// 한 pass = thread당 LOP3 32개. load-xor와 같은 pass/round branch 구조를 유지해
// r2-r1 차이에서 control-flow 성분이 짝을 이루도록 한다.
template <int Pair, int Remaining>
__device__ __forceinline__ void xorOnlyPass(std::uint32_t& checksum,
                                            const XorWords& words,
                                            std::uint32_t threadWord) {
  checksum = xor3Bits(checksum, words.a, threadWord);
  if constexpr (Remaining > 1)
    xorOnlyPass<Pair + 1, Remaining - 1>(checksum, words, threadWord);
}

// L1 load-xor의 control benchmark.
// SM당 512-thread block 하나와 동일한 iteration/pass 구조를 쓰되 data load는 전혀
// 하지 않는다. r1→r2의 에너지 기울기는 LOP3 32개/thread/iteration의 기울기다.
// 이 기울기를 load-xor의 같은 r1→r2 차분에서 빼면 LDG 성분을 분리할 수 있다.
__global__ void l1XorOnlyKernel(std::uint32_t* __restrict__ checksums,
                                unsigned int iterations, XorWords words,
                                int rounds) {
  const std::uint32_t threadWord = static_cast<std::uint32_t>(threadIdx.x);
  std::uint32_t checksum = 0;
#pragma unroll 1
  for (unsigned int iteration = 0; iteration < iterations; ++iteration) {
#pragma unroll 1
    for (int pass = 0; pass < kMaxRounds; ++pass) {
      if (rounds > pass)
        xorOnlyPass<0, kPairsPerPass>(checksum, words, threadWord);
    }
  }
  // 정상 경로 store는 없지만, checksum chain은 compiler dead-code 제거를 막는다.
  if (checksum == 0xffffffffu)
    checksums[blockIdx.x * blockDim.x + threadIdx.x] = checksum;
}

XorWords makeWords() {
  std::uint32_t seed = 0x6a09e667u;
  auto next = [&] {
    seed ^= seed << 13;
    seed ^= seed >> 17;
    seed ^= seed << 5;
    return seed;
  };
  return {next()};
}

void launch(std::uint32_t* checksums, int blocks, unsigned int iterations,
            int rounds, cudaStream_t stream) {
  const XorWords words = makeWords();
  if (rounds < 1 || rounds > kMaxRounds) fail("invalid round count");
  l1XorOnlyKernel<<<blocks, kThreads, 0, stream>>>(checksums, iterations, words, rounds);
  CUDA_CHECK(cudaGetLastError());
}

void run(const Options& options) {
  CUDA_CHECK(cudaSetDevice(options.device));
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, options.device));
  const int blocks = prop.multiProcessorCount;
  const int rounds = options.activeLoads / kLoadsPerPass;
  std::uint32_t* checksums = nullptr;
  CUDA_CHECK(cudaMalloc(&checksums, static_cast<size_t>(blocks) * kThreads * sizeof(std::uint32_t)));

  const float milliseconds = timedLaunchSequence(
      options,
      [&](cudaStream_t stream) {
        launch(checksums, blocks, options.iterations, rounds, stream);
      },
      "l1v2_xor_only_graph_replay");
  const double issuedLop3 = static_cast<double>(options.activeLoads / 2) *
      options.iterations * blocks * kThreads * launchMultiplier(options);
  std::cout << std::fixed << std::setprecision(3)
            << "kernel=l1XorOnlyKernel rounds=" << rounds
            << " iterations=" << options.iterations
            << " launch_mode=" << options.launchMode
            << " graph_nodes=" << options.graphNodes
            << " graph_replays=" << options.graphReplays
            << " sm_count=" << blocks
            << " kernel_ms=" << milliseconds
            << " logical_lop3=" << std::setprecision(0) << issuedLop3 << '\n';

  CUDA_CHECK(cudaFree(checksums));
}

}  // namespace

int main(int argc, char** argv) {
  try {
    run(l1v2::parseOptions(argc, argv));
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    if (std::string(error.what()) == "help") {
      l1v2::printCommonUsage(argv[0], "xor-only: each pass adds 32 LOP3 and no data load.");
      return EXIT_SUCCESS;
    }
    std::cerr << "Error: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
