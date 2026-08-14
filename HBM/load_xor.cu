// HBM v2 LDG-dominant load-xor differential kernel.
// The 64 or 128 LDG.E.128 instructions are not paired
// with a proportional LOP3 data reduction: it uses one LOP3 per two vector
// loads. NCU confirms the retained LDG.E.128 instructions at the executed
// warp-opcode level.
#include "common.cuh"

#include <vector>

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
  const size_t groups = count / (gridStride * kVectorsPerGroup);
  if (groups == 0) fail("buffer too small for HBM stream geometry");
  packed_fp32* data = nullptr;
  CUDA_CHECK(cudaMalloc(&data, bytes));
  const int initBlocks = std::min<int>(65535, (count + kThreads - 1) / kThreads);
  initBits<<<initBlocks, kThreads>>>(data, count, inputMode(options.input));
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  const int rounds = options.activeLoads / kVectorLoadsPerRound;
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
  const double logicalBytes = static_cast<double>(options.activeLoads) * sizeof(packed_fp32) *
      static_cast<double>(groups) * gridStride;
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
                         "HBM LDG-dominant: r2-r1 adds 64 LDG.v4 + 32 LOP3; "
                         "all other executed SASS opcodes are balanced.");
      return EXIT_SUCCESS;
    }
    std::cerr << "Error: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
