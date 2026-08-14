// HBM v2 LOP3-only control slope.  r1/r2 differ by exactly one 128-LOP3
// pass; it is the control counterpart of load_xor's extra 64-LDG pass.
#include "common.cuh"

#include <vector>

namespace {
using namespace hbmv2;

void run(const Options& options) {
  CUDA_CHECK(cudaSetDevice(options.device));
  cudaDeviceProp prop{}; CUDA_CHECK(cudaGetDeviceProperties(&prop, options.device));
  const size_t bytes = options.mib * 1024ull * 1024ull;
  if (bytes <= static_cast<size_t>(prop.l2CacheSize)) fail("--mib must exceed L2 size");
  const size_t count = bytes / sizeof(packed_fp32);
  const int blocks = prop.multiProcessorCount * options.blocksPerSm;
  const size_t gridStride = static_cast<size_t>(blocks) * kThreads;
  const size_t groups = count / (gridStride * kVectorsPerGroup);
  if (groups == 0) fail("buffer too small for HBM stream geometry");
  const int rounds = options.activeLoads / kVectorLoadsPerRound;

  // Allocate the same-sized buffer as load_xor.cu to keep the process/context
  // footprint comparable; it is deliberately untouched by the timed kernel.
  packed_fp32* footprint = nullptr; CUDA_CHECK(cudaMalloc(&footprint, bytes));
  launchXorOnly(groups, blocks, rounds, 1, nullptr); CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<double> milliseconds; milliseconds.reserve(options.trials);
  cudaEvent_t start{}, stop{}; CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop));
  for (int trial = 0; trial < options.trials; ++trial) {
    CUDA_CHECK(cudaEventRecord(start)); launchXorOnly(groups, blocks, rounds, 1, nullptr);
    CUDA_CHECK(cudaEventRecord(stop)); CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f; CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop)); milliseconds.push_back(ms);
  }
  CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop));
  const double best = *std::min_element(milliseconds.begin(), milliseconds.end());
  const double logicalLop3 = static_cast<double>(options.activeLoads) * 2.0 *
      static_cast<double>(groups) * gridStride;
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
      hbmv2::printUsage(argv[0], "HBM xor-only: r2-r1 adds exactly 128 LOP3 per thread/group.");
      return EXIT_SUCCESS;
    }
    std::cerr << "Error: " << error.what() << '\n'; return EXIT_FAILURE;
  }
}
