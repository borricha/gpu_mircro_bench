// L1 v2 in-process CUDA-Graph/NVML power harness.
// This uses the same measurement framing as the L2 harness: one process owns
// allocation, warm-up graph, measurement graph, CUDA events, and NVML samples.
#include "common.cuh"
#include "../../measure/power_telemetry.hpp"

#include <nvml.h>

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace {
using namespace l1v2;

struct PowerOptions {
  int device = 0;
  std::string kind = "load-xor";
  int activeLoads = kLoadsPerPass;
  unsigned int iterations = kDefaultIterations;
  double preconditionSeconds = 20.0;
  unsigned int sampleMs = 10;
  double pConstantW = 33.154;
  double pStaticW = 34.103;
  std::string input = "random";
  int launchBatch = 32;
  int graphNodes = 1322;
  int repetition = 1;
  std::string telemetryOutput;
  std::string output;
};

struct XorWords { std::uint32_t a; };

[[noreturn]] void failPower(const std::string& message) {
  throw std::runtime_error(message);
}

long long parseNumber(const char* text, const char* name) {
  char* end = nullptr;
  const long long value = std::strtoll(text, &end, 10);
  if (end == nullptr || *end != '\0') failPower(std::string("invalid ") + name);
  return value;
}

double parseDouble(const char* text, const char* name) {
  char* end = nullptr;
  const double value = std::strtod(text, &end);
  if (end == nullptr || *end != '\0' || !std::isfinite(value))
    failPower(std::string("invalid ") + name);
  return value;
}

PowerOptions parsePowerOptions(int argc, char** argv) {
  if (argc < 2 || std::string(argv[1]) != "--power-trial")
    failPower("use --power-trial");
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
    else if (arg == "--iterations") options.iterations = static_cast<unsigned int>(parseNumber(value(), "iterations"));
    else if (arg == "--precondition-seconds") options.preconditionSeconds = parseDouble(value(), "precondition-seconds");
    else if (arg == "--sample-ms") options.sampleMs = static_cast<unsigned int>(parseNumber(value(), "sample-ms"));
    else if (arg == "--p-constant-w") options.pConstantW = parseDouble(value(), "p-constant-w");
    else if (arg == "--p-static-w") options.pStaticW = parseDouble(value(), "p-static-w");
    else if (arg == "--input") options.input = value();
    else if (arg == "--launch-batch") options.launchBatch = static_cast<int>(parseNumber(value(), "launch-batch"));
    else if (arg == "--graph-nodes") options.graphNodes = static_cast<int>(parseNumber(value(), "graph-nodes"));
    else if (arg == "--repetition") options.repetition = static_cast<int>(parseNumber(value(), "repetition"));
    else if (arg == "--telemetry-output") options.telemetryOutput = value();
    else if (arg == "--output") options.output = value();
    else if (arg == "--help" || arg == "-h") {
      std::cout << "Usage: " << argv[0]
                << " --power-trial --kind load-xor|xor-only --active-loads 64|128|192|256"
                << " [--iterations 61037 --graph-nodes 1322 --launch-batch 32]"
                << " [--precondition-seconds 20 --sample-ms 10]"
                << " [--p-constant-w W --p-static-w W]"
                << " [--telemetry-output FILE --output FILE]\n";
      std::exit(EXIT_SUCCESS);
    } else {
      failPower("unknown option: " + arg);
    }
  }
  if (options.device < 0 || (options.kind != "load-xor" && options.kind != "xor-only") ||
      options.activeLoads < kLoadsPerPass || options.activeLoads > kMaxRounds * kLoadsPerPass ||
      options.activeLoads % kLoadsPerPass != 0 || options.iterations == 0 ||
      options.preconditionSeconds < 0 || options.sampleMs == 0 || options.launchBatch < 1 ||
      options.graphNodes < 1 || options.repetition < 1 || options.pConstantW < 0 ||
      options.pStaticW < 0)
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
  nvmlDevice_t result{};
  nvmlCheck(nvmlDeviceGetHandleByPciBusId_v2(pciBusId, &result),
            "nvmlDeviceGetHandleByPciBusId_v2");
  return result;
}

bool hasContent(const std::string& path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  return file && file.tellg() > 0;
}

void ensureParentDirectory(const std::string& path) {
  const auto parent = std::filesystem::path(path).parent_path();
  if (parent.empty()) return;
  std::error_code error;
  std::filesystem::create_directories(parent, error);
  if (error) failPower("cannot create output directory: " + parent.string());
}

__device__ __forceinline__ std::uint32_t loadCacheAllBits(const float* address) {
  std::uint32_t value;
  asm volatile("ld.global.ca.u32 %0, [%1];" : "=r"(value) : "l"(address) : "memory");
  return value;
}

__device__ __forceinline__ std::uint32_t xor3Bits(
    std::uint32_t a, std::uint32_t b, std::uint32_t c) {
  std::uint32_t result;
  asm volatile("lop3.b32 %0, %1, %2, %3, 0x96;"
               : "=r"(result) : "r"(a), "r"(b), "r"(c));
  return result;
}

template <int Pair, int Remaining>
__device__ __forceinline__ void loadXorPass(std::uint32_t& checksum, const float* base) {
  constexpr int kOffset = Pair * kThreads;
  const auto a = loadCacheAllBits(base + kOffset);
  const auto b = loadCacheAllBits(base + kElementsPerSm / 2 + kOffset);
  checksum = xor3Bits(checksum, a, b);
  if constexpr (Remaining > 1) loadXorPass<Pair + 1, Remaining - 1>(checksum, base);
}

template <int Pair, int Remaining>
__device__ __forceinline__ void xorOnlyPass(std::uint32_t& checksum,
                                             const XorWords& words,
                                             std::uint32_t threadWord) {
  checksum = xor3Bits(checksum, words.a, threadWord);
  if constexpr (Remaining > 1) xorOnlyPass<Pair + 1, Remaining - 1>(checksum, words, threadWord);
}

__global__ void l1PowerLoadXorKernel(std::uint32_t* __restrict__ checksums,
                                     const float* __restrict__ data,
                                     unsigned int iterations, int zero, int rounds) {
  const float* base = data + threadIdx.x;
  std::uint32_t checksum = 0;
#pragma unroll 1
  for (unsigned int iteration = 0; iteration < iterations; ++iteration) {
    base += zero;
#pragma unroll 1
    for (int pass = 0; pass < kMaxRounds; ++pass)
      if (rounds > pass) loadXorPass<0, kPairsPerPass>(checksum, base);
  }
  if (checksum == 0xffffffffu)
    checksums[blockIdx.x * blockDim.x + threadIdx.x] = checksum;
}

__global__ void l1PowerXorOnlyKernel(std::uint32_t* __restrict__ checksums,
                                     unsigned int iterations, XorWords words, int rounds) {
  const auto threadWord = static_cast<std::uint32_t>(threadIdx.x);
  std::uint32_t checksum = 0;
#pragma unroll 1
  for (unsigned int iteration = 0; iteration < iterations; ++iteration) {
#pragma unroll 1
    for (int pass = 0; pass < kMaxRounds; ++pass)
      if (rounds > pass) xorOnlyPass<0, kPairsPerPass>(checksum, words, threadWord);
  }
  if (checksum == 0xffffffffu)
    checksums[blockIdx.x * blockDim.x + threadIdx.x] = checksum;
}

XorWords makeWords() {
  std::uint32_t seed = 0x6a09e667u;
  seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
  return {seed};
}

void runPowerTrial(const PowerOptions& options) {
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

  cudaStream_t stream{};
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  const XorWords words = makeWords();
  auto launch = [&](cudaStream_t target) {
    if (options.kind == "load-xor") {
      l1PowerLoadXorKernel<<<blocks, kThreads, 0, target>>>(
          checksums, data, options.iterations, 0, rounds);
    } else {
      l1PowerXorOnlyKernel<<<blocks, kThreads, 0, target>>>(
          checksums, options.iterations, words, rounds);
    }
    CUDA_CHECK(cudaGetLastError());
  };
  auto capture = [&](int nodes, cudaGraph_t* graph, cudaGraphExec_t* exec) {
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    for (int node = 0; node < nodes; ++node) launch(stream);
    CUDA_CHECK(cudaStreamEndCapture(stream, graph));
    CUDA_CHECK(cudaGraphInstantiate(exec, *graph, nullptr, nullptr, 0));
  };
  cudaGraph_t warmGraph{}, measureGraph{};
  cudaGraphExec_t warmExec{}, measureExec{};
  capture(options.launchBatch, &warmGraph, &warmExec);
  capture(options.graphNodes, &measureGraph, &measureExec);

  long long preconditionLaunches = 0;
  const auto preconditionEnd = std::chrono::steady_clock::now() +
      std::chrono::duration_cast<std::chrono::steady_clock::duration>(
          std::chrono::duration<double>(options.preconditionSeconds));
  do {
    CUDA_CHECK(cudaGraphLaunch(warmExec, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    preconditionLaunches += options.launchBatch;
  } while (std::chrono::steady_clock::now() < preconditionEnd);

  cudaEvent_t start{}, stop{};
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  const nvmlDevice_t nvmlDevice = nvmlForCudaDevice(options.device);
  const auto wallStart = powertelemetry::Clock::now();
  powertelemetry::Sampler sampler(nvmlDevice, options.sampleMs);
  sampler.start(wallStart);
  nvtxRangePushA(options.kind == "load-xor" ? "l1_load_xor_power_graph" : "l1_xor_only_power_graph");
  CUDA_CHECK(cudaEventRecord(start, stream));
  CUDA_CHECK(cudaGraphLaunch(measureExec, stream));
  CUDA_CHECK(cudaEventRecord(stop, stream));
  CUDA_CHECK(cudaEventSynchronize(stop));
  nvtxRangePop();
  const auto wallStop = powertelemetry::Clock::now();
  sampler.stop();
  float milliseconds = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  const double activeSeconds = milliseconds / 1000.0;
  const auto telemetry = sampler.summarize(
      std::chrono::duration<double>(wallStop - wallStart).count());
  if (!options.telemetryOutput.empty()) {
    ensureParentDirectory(options.telemetryOutput);
    sampler.write_csv(options.telemetryOutput);
  }

  const double threadWork = static_cast<double>(options.graphNodes) * options.iterations *
      blocks * kThreads;
  const double logicalLdg = options.kind == "load-xor"
      ? static_cast<double>(options.activeLoads) * threadWork : 0.0;
  const double logicalLop3 = static_cast<double>(options.activeLoads / 2) * threadWork;
  const double eConst = (telemetry.time_weighted_avg_w - options.pConstantW) * activeSeconds;
  const double eStatic =
      (telemetry.time_weighted_avg_w - options.pConstantW - options.pStaticW) * activeSeconds;
  const char* header =
      "kind,active_loads,graph_nodes,precondition_launches,kernel_active_s,logical_ldg,logical_lop3,board_avg_w,p_const_w,p_static_w,energy_after_const_j,energy_after_const_static_j,util_avg_pct,sm_clock_avg_mhz,sm_clock_min_mhz,sm_clock_max_mhz,mem_clock_avg_mhz,throttle_any_pct,throttle_sw_power_pct,throttle_hw_slowdown_pct,throttle_hw_thermal_slowdown_pct,sample_count,telemetry_csv";
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
            << "Power trial (in-process graph): kind=" << options.kind
            << " active_loads=" << options.activeLoads
            << " iterations=" << options.iterations
            << " graph_nodes=" << options.graphNodes
            << " active_s=" << activeSeconds
            << " board=" << telemetry.time_weighted_avg_w
            << "W e_const=" << eConst << "J e_const_static=" << eStatic << "J\n";

  CUDA_CHECK(cudaGraphExecDestroy(measureExec)); CUDA_CHECK(cudaGraphDestroy(measureGraph));
  CUDA_CHECK(cudaGraphExecDestroy(warmExec)); CUDA_CHECK(cudaGraphDestroy(warmGraph));
  CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaFree(checksums)); CUDA_CHECK(cudaFree(data));
}
}  // namespace

int main(int argc, char** argv) {
  try {
    const auto options = parsePowerOptions(argc, argv);
    nvmlCheck(nvmlInit_v2(), "nvmlInit_v2");
    try {
      runPowerTrial(options);
      nvmlCheck(nvmlShutdown(), "nvmlShutdown");
    } catch (...) {
      nvmlShutdown();
      throw;
    }
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
