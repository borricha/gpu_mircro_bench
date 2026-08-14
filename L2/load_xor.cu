// L2 v2 differential kernel: v1의 일반 global-load mapping 위에서,
// 추가 pass마다 LDG 64개 + LOP3 64개만 늘린다.
#include "common.cuh"

#include <iomanip>

namespace {

using namespace l2v2;

// v1 loadOnlyKernel과 동일하게 평범한 global load를 쓴다. 즉 .cg/__ldcg로
// L1 allocation을 금지하지 않는다.  v1에서 약 5 TB/s가 나오던 경로를 보존한다.
__device__ __forceinline__ std::uint32_t loadGlobal(const float* address) {
  return __float_as_uint(*address);
}

// i/offset을 template 값으로 둬서 모든 load address를 compile-time에 고정한다.
// r1/r2 모두 동일한 base-address prologue를 쓰며, r2에는 이 template body의 뒤
// 32 pair만 추가된다. 따라서 active-load 수를 바꿔도 dynamic IMAD는 늘지 않는다.
template <int Pair, int Remaining>
__device__ __forceinline__ void staticLoadXorPairs(
    std::uint32_t& checksum, const float* pairBase) {
  constexpr size_t kOffset = static_cast<size_t>(Pair) * kPairStride;
  checksum ^= loadGlobal(pairBase + kOffset);
  checksum ^= loadGlobal(pairBase + kOffset + kBlockSize);
  if constexpr (Remaining > 1)
    staticLoadXorPairs<Pair + 1, Remaining - 1>(checksum, pairBase);
}

// L2 v2의 핵심 kernel.
//
// - data: v1과 동일한 16,896 KiB FP32. A100 40 MiB L2 안에 들어간다.
// - grid/block: v1과 동일한 200,000 CTA × 1024 threads. scheduler가 많은 CTA를
//   지속 공급하여 L2 latency를 숨긴다.
// - blockIdx.x % 33 region과 i별 stride도 v1과 동일하다. 따라서 r2(128 loads)는
//   기존 loadOnlyKernel의 128 load address sequence와 같다.
// - r1/r2 차이는 active pass 수뿐이다. r2-r1의 active body 증분은 thread당
//   iteration마다 일반 LDG 64개 + LOP3 64개다.
template <int ActiveLoads, int BlockSize>
__global__ void l2LoadXorKernel(const float* __restrict__ data,
                                int blockRun) {
  std::uint32_t checksum = 0;
  // v1의 region mapping은 유지하되, i별 offset은 위 template에 static으로 넣는다.
  // v1과 같은 33 × 512 KiB = 16,896 KiB working set. CTA는 512 KiB region 하나를
  // 맡고, compile-time offset 64쌍을 순회한다. region은 L1보다 커서 반복 CTA가
  // 만든 data는 L2에 머무르지만 r1/r2 address arithmetic은 동일하게 유지된다.
  const size_t blockOffset = static_cast<size_t>(blockIdx.x % blockRun) *
      kElementsPerRegion + threadIdx.x;
  const float* base = data + blockOffset;
  staticLoadXorPairs<0, kLoadsPerPass / 2>(checksum, base);
  if constexpr (ActiveLoads == 2 * kLoadsPerPass)
    staticLoadXorPairs<kLoadsPerPass / 2, kLoadsPerPass / 2>(checksum, base);

  // v1과 동일하게 normal-path store는 없고, trap 의존성이 모든 load를 live로 만든다.
  if (checksum == 0xffffffffu)
    asm volatile("trap;");
}

void launch(const float* data, int blocks, unsigned int /* iterations */, int rounds,
            cudaStream_t stream) {
  if (rounds < 1 || rounds > kMaxRounds) fail("invalid round count");
  if (rounds == 1)
    l2LoadXorKernel<kLoadsPerPass, kBlockSize><<<blocks, kThreads, 0, stream>>>(
        data, kBlockRun);
  else
    l2LoadXorKernel<2 * kLoadsPerPass, kBlockSize><<<blocks, kThreads, 0, stream>>>(
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
  // v1 initKernel과 같은 52 × 256 launch geometry를 사용한다.
  initBits<<<52, 256>>>(data, kDataElements, inputMode(options.input));
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  // v1은 같은 process에서 11회 실행한 뒤 minimum을 보고한다. 따라서 첫 실행이
  // 16,896 KiB working set을 L2로 채운 뒤의 warm-cache 상태를 측정한다. v2도
  // 측정 event 바깥에서 동일 kernel을 15번 실행하여 이 조건을 맞춘다.
  for (int warmup = 0; warmup < 15; ++warmup)
    launch(data, blocks, options.iterations, rounds, 0);
  CUDA_CHECK(cudaDeviceSynchronize());

  const float milliseconds = timedLaunchSequence(
      options,
      [&](cudaStream_t stream) {
        launch(data, blocks, options.iterations, rounds, stream);
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
          "v1-style L2 path: ordinary global LDG; each pass adds 64 LDG + 64 LOP3.");
      return EXIT_SUCCESS;
    }
    std::cerr << "Error: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
