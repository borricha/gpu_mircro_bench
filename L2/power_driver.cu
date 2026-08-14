// v1 power-trial framing for the L2 v2 SASS-differential kernels.
// One process owns the held allocation, L2 buffer, warmup graph, measurement
// graph, and NVML sampler so precondition and measurement share the same cache.
#include "common.cuh"
#include "../common/power_telemetry.hpp"

#include <nvml.h>
#include <nvtx3/nvToolsExt.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

namespace {
using namespace l2v2;

struct PowerOptions {
  int device = 0;
  std::string kind = "load-xor";
  int activeLoads = 64;
  double preconditionSeconds = 20.0;
  unsigned int sampleMs = 10;
  double pConstantW = 33.154;
  double pStaticW = 34.103;
  size_t allocatedMiB = 1024;
  std::string input = "random";
  int launchBatch = 32;
  int graphNodes = 5000;
  std::string telemetryOutput;
  std::string output;
};

[[noreturn]] void failPower(const std::string& message) { throw std::runtime_error(message); }

long long parseNumber(const char* text, const char* name) {
  char* end = nullptr;
  const long long value = std::strtoll(text, &end, 10);
  if (!end || *end != '\0') failPower(std::string("invalid ") + name);
  return value;
}

double parseDouble(const char* text, const char* name) {
  char* end = nullptr;
  const double value = std::strtod(text, &end);
  if (!end || *end != '\0' || !std::isfinite(value))
    failPower(std::string("invalid ") + name);
  return value;
}

PowerOptions parsePower(int argc, char** argv) {
  PowerOptions options;
  for (int i = 2; i < argc; ++i) {
    const std::string arg = argv[i];
    auto value = [&]() -> const char* {
      if (++i >= argc) failPower("missing value for " + arg);
      return argv[i];
    };
    if (arg == "--device") options.device = static_cast<int>(parseNumber(value(), "device"));
    else if (arg == "--kind") options.kind = value();
    else if (arg == "--active-loads") options.activeLoads = static_cast<int>(parseNumber(value(), "active-loads"));
    else if (arg == "--precondition-seconds") options.preconditionSeconds = parseDouble(value(), "precondition-seconds");
    else if (arg == "--sample-ms") options.sampleMs = static_cast<unsigned int>(parseNumber(value(), "sample-ms"));
    else if (arg == "--p-constant-w") options.pConstantW = parseDouble(value(), "p-constant-w");
    else if (arg == "--p-static-w") options.pStaticW = parseDouble(value(), "p-static-w");
    else if (arg == "--allocated-mib") options.allocatedMiB = static_cast<size_t>(parseNumber(value(), "allocated-mib"));
    else if (arg == "--input") options.input = value();
    else if (arg == "--launch-batch") options.launchBatch = static_cast<int>(parseNumber(value(), "launch-batch"));
    else if (arg == "--graph-nodes") options.graphNodes = static_cast<int>(parseNumber(value(), "graph-nodes"));
    else if (arg == "--telemetry-output") options.telemetryOutput = value();
    else if (arg == "--output") options.output = value();
    else if (arg == "--help" || arg == "-h") {
      std::cout << "Usage: " << argv[0] << " --power-trial --kind xor-only|load-xor "
                   "--active-loads 64|128 [v1 power-trial options]\n";
      std::exit(EXIT_SUCCESS);
    } else failPower("unknown option: " + arg);
  }
  if (options.device < 0 || options.kind != "xor-only" && options.kind != "load-xor" ||
      (options.activeLoads != 64 && options.activeLoads != 128) ||
      options.preconditionSeconds < 0 || options.sampleMs == 0 ||
      options.pConstantW < 0 || options.pStaticW < 0 || options.allocatedMiB == 0 ||
      options.launchBatch < 1 || options.graphNodes < 1)
    failPower("invalid power-trial option");
  inputMode(options.input);
  return options;
}

void nvmlCheck(nvmlReturn_t status, const char* context) {
  if (status != NVML_SUCCESS)
    failPower(std::string(context) + ": " + nvmlErrorString(status));
}

nvmlDevice_t nvmlForCudaDevice(int device) {
  char pciBusId[32]{};
  CUDA_CHECK(cudaDeviceGetPCIBusId(pciBusId, sizeof(pciBusId), device));
  nvmlDevice_t nvmlDevice{};
  nvmlCheck(nvmlDeviceGetHandleByPciBusId_v2(pciBusId, &nvmlDevice),
            "nvmlDeviceGetHandleByPciBusId_v2");
  return nvmlDevice;
}

__device__ __forceinline__ std::uint32_t loadGlobal(const float* address) {
  return __float_as_uint(*address);
}

template <int Pair, int Remaining>
__device__ __forceinline__ void loadPairs(std::uint32_t& checksum, const float* base) {
  constexpr size_t offset = static_cast<size_t>(Pair) * kPairStride;
  checksum ^= loadGlobal(base + offset);
  checksum ^= loadGlobal(base + offset + kBlockSize);
  if constexpr (Remaining > 1) loadPairs<Pair + 1, Remaining - 1>(checksum, base);
}

template <int ActiveLoads>
__global__ void powerLoadXorKernel(const float* __restrict__ data, int blockRun) {
  std::uint32_t checksum = 0;
  const size_t blockOffset = static_cast<size_t>(blockIdx.x % blockRun) *
      kElementsPerRegion + threadIdx.x;
  const float* base = data + blockOffset;
  loadPairs<0, kLoadsPerPass / 2>(checksum, base);
  if constexpr (ActiveLoads == 128) loadPairs<kLoadsPerPass / 2, kLoadsPerPass / 2>(checksum, base);
  if (checksum == 0xffffffffu) asm volatile("trap;");
}

__device__ __forceinline__ std::uint32_t xor3Bits(std::uint32_t a, std::uint32_t b,
                                                   std::uint32_t c) {
  std::uint32_t result;
  asm volatile("lop3.b32 %0, %1, %2, %3, 0x96;" : "=r"(result) : "r"(a), "r"(b), "r"(c));
  return result;
}

template <int Slot, int Remaining>
__device__ __forceinline__ void xorSlots(std::uint32_t& checksum,
                                          std::uint32_t lo, std::uint32_t hi) {
  checksum = xor3Bits(checksum, lo, hi);
  if constexpr (Remaining > 1) xorSlots<Slot + 1, Remaining - 1>(checksum, lo, hi);
}

template <int Rounds>
__global__ void powerXorOnlyKernel(const float* __restrict__ data, int blockRun) {
  std::uint32_t checksum = 0;
  const size_t blockOffset = static_cast<size_t>(blockIdx.x % blockRun) *
      kElementsPerRegion + threadIdx.x;
  const auto address = reinterpret_cast<std::uintptr_t>(data + blockOffset);
  const auto lo = static_cast<std::uint32_t>(address);
  const auto hi = static_cast<std::uint32_t>(address >> 32);
  xorSlots<0, kLoadsPerPass / 2>(checksum, lo, hi);
  if constexpr (Rounds == 2) xorSlots<kLoadsPerPass / 2, kLoadsPerPass / 2>(checksum, lo, hi);
  if (checksum == 0xffffffffu) asm volatile("trap;");
}

void launch(const PowerOptions& options, const float* data, cudaStream_t stream) {
  if (options.kind == "load-xor") {
    if (options.activeLoads == 64)
      powerLoadXorKernel<64><<<kLaunchBlocks, kThreads, 0, stream>>>(data, kBlockRun);
    else
      powerLoadXorKernel<128><<<kLaunchBlocks, kThreads, 0, stream>>>(data, kBlockRun);
  } else {
    if (options.activeLoads == 64)
      powerXorOnlyKernel<1><<<kLaunchBlocks, kThreads, 0, stream>>>(data, kBlockRun);
    else
      powerXorOnlyKernel<2><<<kLaunchBlocks, kThreads, 0, stream>>>(data, kBlockRun);
  }
  CUDA_CHECK(cudaGetLastError());
}

bool hasContent(const std::string& path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  return input && input.tellg() > 0;
}

void ensureParentDirectory(const std::string& path) {
  if (path.empty()) return;
  const auto parent = std::filesystem::path(path).parent_path();
  if (parent.empty()) return;
  std::error_code error;
  std::filesystem::create_directories(parent, error);
  if (error) failPower("cannot create output directory: " + parent.string());
}

void runPowerTrial(const PowerOptions& options) {
  CUDA_CHECK(cudaSetDevice(options.device));
  nvmlDevice_t nvmlDevice = nvmlForCudaDevice(options.device);
  void* held = nullptr;
  float* data = nullptr;
  CUDA_CHECK(cudaMalloc(&held, options.allocatedMiB * 1024ull * 1024ull));
  CUDA_CHECK(cudaMemset(held, 0, options.allocatedMiB * 1024ull * 1024ull));
  CUDA_CHECK(cudaMalloc(&data, kDataBytes));
  initBits<<<52, 256>>>(data, kDataElements, inputMode(options.input));
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaStream_t stream{};
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  cudaGraph_t warmGraph{}, measureGraph{};
  cudaGraphExec_t warmExec{}, measureExec{};
  auto capture = [&](int nodes, cudaGraph_t* graph, cudaGraphExec_t* exec) {
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    for (int i = 0; i < nodes; ++i) launch(options, data, stream);
    CUDA_CHECK(cudaStreamEndCapture(stream, graph));
    CUDA_CHECK(cudaGraphInstantiate(exec, *graph, nullptr, nullptr, 0));
  };
  capture(options.launchBatch, &warmGraph, &warmExec);
  capture(options.graphNodes, &measureGraph, &measureExec);

  long long preconditionLaunches = 0;
  const auto until = std::chrono::steady_clock::now() +
      std::chrono::duration_cast<std::chrono::steady_clock::duration>(
          std::chrono::duration<double>(options.preconditionSeconds));
  do {
    CUDA_CHECK(cudaGraphLaunch(warmExec, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    preconditionLaunches += options.launchBatch;
  } while (std::chrono::steady_clock::now() < until);

  cudaEvent_t start{}, stop{};
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  const auto wallStart = powertelemetry::Clock::now();
  powertelemetry::Sampler sampler(nvmlDevice, options.sampleMs);
  sampler.start(wallStart);
  nvtxRangePushA(options.kind == "load-xor" ? "l2_load_xor_power_graph" : "l2_xor_only_power_graph");
  CUDA_CHECK(cudaEventRecord(start, stream));
  CUDA_CHECK(cudaGraphLaunch(measureExec, stream));
  CUDA_CHECK(cudaEventRecord(stop, stream));
  CUDA_CHECK(cudaEventSynchronize(stop));
  nvtxRangePop();
  const auto wallStop = powertelemetry::Clock::now();
  sampler.stop();
  float milliseconds = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
  const double activeSeconds = milliseconds / 1000.0;
  const auto telemetry = sampler.summarize(
      std::chrono::duration<double>(wallStop - wallStart).count());
  if (!options.telemetryOutput.empty()) {
    ensureParentDirectory(options.telemetryOutput);
    sampler.write_csv(options.telemetryOutput);
  }

  const double logicalLdg = options.kind == "load-xor"
      ? static_cast<double>(options.activeLoads) * kLaunchBlocks * kThreads * options.graphNodes : 0.0;
  const double logicalLop3 = static_cast<double>(options.activeLoads / 2) *
      kLaunchBlocks * kThreads * options.graphNodes;
  const double eConst = (telemetry.time_weighted_avg_w - options.pConstantW) * activeSeconds;
  const double eStatic = (telemetry.time_weighted_avg_w - options.pConstantW - options.pStaticW) * activeSeconds;
  const char* header = "kind,active_loads,graph_nodes,precondition_launches,kernel_active_s,logical_ldg,logical_lop3,board_avg_w,p_const_w,p_static_w,energy_after_const_j,energy_after_const_static_j,util_avg_pct,sm_clock_avg_mhz,sm_clock_min_mhz,sm_clock_max_mhz,mem_clock_avg_mhz,throttle_any_pct,throttle_sw_power_pct,throttle_hw_slowdown_pct,throttle_hw_thermal_slowdown_pct,sample_count,telemetry_csv";
  std::ostringstream row;
  row << options.kind << ',' << options.activeLoads << ',' << options.graphNodes << ','
      << preconditionLaunches << ',' << std::fixed << std::setprecision(9) << activeSeconds << ','
      << logicalLdg << ',' << logicalLop3 << ',' << telemetry.time_weighted_avg_w << ','
      << options.pConstantW << ',' << options.pStaticW << ',' << eConst << ',' << eStatic << ','
      << telemetry.util_avg_pct << ',' << telemetry.sm_clock_avg_mhz << ','
      << telemetry.sm_clock_min_mhz << ',' << telemetry.sm_clock_max_mhz << ','
      << telemetry.mem_clock_avg_mhz << ',' << telemetry.throttle_any_pct << ','
      << telemetry.throttle_sw_power_pct << ',' << telemetry.throttle_hw_slowdown_pct << ','
      << telemetry.throttle_hw_thermal_pct << ',' << telemetry.sample_count << ','
      << options.telemetryOutput;
  if (!options.output.empty()) {
    ensureParentDirectory(options.output);
    const bool writeHeader = !hasContent(options.output);
    std::ofstream output(options.output, std::ios::app);
    if (!output) failPower("cannot open output: " + options.output);
    if (writeHeader) output << header << '\n';
    output << row.str() << '\n';
  }
  std::cout << std::fixed << std::setprecision(3)
            << "Power trial (v1 framing): kind=" << options.kind
            << " active_loads=" << options.activeLoads
            << " graph_nodes=" << options.graphNodes
            << " precondition_launches=" << preconditionLaunches
            << " active_s=" << activeSeconds
            << " board=" << telemetry.time_weighted_avg_w
            << "W e_const=" << eConst
            << "J e_const_static=" << eStatic << "J\n";

  CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaGraphExecDestroy(measureExec)); CUDA_CHECK(cudaGraphDestroy(measureGraph));
  CUDA_CHECK(cudaGraphExecDestroy(warmExec)); CUDA_CHECK(cudaGraphDestroy(warmGraph));
  CUDA_CHECK(cudaStreamDestroy(stream)); CUDA_CHECK(cudaFree(data)); CUDA_CHECK(cudaFree(held));
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 2 || std::string(argv[1]) != "--power-trial")
      failPower("use --power-trial");
    const auto options = parsePower(argc, argv);
    nvmlCheck(nvmlInit_v2(), "nvmlInit_v2");
    runPowerTrial(options);
    nvmlShutdown();
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
