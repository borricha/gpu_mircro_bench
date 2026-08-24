// Long-running CUDA-Graph/NVML harness for the HBM v2 SASS-differential
// kernels. It reuses common.cuh, so NCU and power execute the same
// load-xor/xor-only device code.
#include "common.cuh"
#include "../common/power_telemetry.hpp"

#include <nvml.h>
#include <nvtx3/nvToolsExt.h>

#include <chrono>
#include <cmath>
#include <fstream>
#include <filesystem>
#include <limits>
#include <sstream>
#include <vector>

// Reuse the exact device kernels from the two experiment sources. The guards
// omit their standalone timing harnesses and mains in this translation unit.
#define HBM_LOAD_XOR_DEVICE_ONLY
#include "load_xor.cu"
#undef HBM_LOAD_XOR_DEVICE_ONLY
#define HBM_XOR_ONLY_DEVICE_ONLY
#include "xor_only.cu"
#undef HBM_XOR_ONLY_DEVICE_ONLY

namespace {
using namespace hbmv2;

struct PowerOptions {
  int device = 0;
  // The retained load kernel is the final LDG-dominant variant.
  std::string kind = "load-xor";
  int activeLoads = 64;
  size_t mib = 1024;
  int blocksPerSm = kDefaultBlocksPerSm;
  int gridBlocks = 0;
  int threads = kDefaultThreads;
  std::string input = "random";
  double preconditionSeconds = 20.0;
  double measureSeconds = 60.0;
  unsigned int sampleMs = 10;
  int innerRepeats = 64;
  int launchBatch = 8;
  int graphNodes = 0;  // derive from an event-timed graph node
  double pConstantW = 0.0;
  double pStaticW = 0.0;
  int repetition = 1;
  std::string telemetryOutput;
  std::string output;
};

[[noreturn]] void failPower(const std::string& message) {
  throw std::runtime_error(message);
}

long long parseIntegerPower(const char* text, const char* name) {
  char* end = nullptr;
  const long long value = std::strtoll(text, &end, 10);
  if (!end || *end != '\0') failPower(std::string("invalid ") + name);
  return value;
}

double parseDoublePower(const char* text, const char* name) {
  char* end = nullptr;
  const double value = std::strtod(text, &end);
  if (!end || *end != '\0' || !std::isfinite(value))
    failPower(std::string("invalid ") + name);
  return value;
}

PowerOptions parsePowerOptions(int argc, char** argv) {
  PowerOptions options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto value = [&]() -> const char* {
      if (++i >= argc) failPower("missing value for " + arg);
      return argv[i];
    };
    if (arg == "--device") options.device = static_cast<int>(parseIntegerPower(value(), "device"));
    else if (arg == "--kind") options.kind = value();
    else if (arg == "--active-loads") options.activeLoads = static_cast<int>(parseIntegerPower(value(), "active-loads"));
    else if (arg == "--mib") options.mib = static_cast<size_t>(parseIntegerPower(value(), "mib"));
    else if (arg == "--blocks-per-sm") options.blocksPerSm = static_cast<int>(parseIntegerPower(value(), "blocks-per-sm"));
    else if (arg == "--grid-blocks") options.gridBlocks = static_cast<int>(parseIntegerPower(value(), "grid-blocks"));
    else if (arg == "--threads") options.threads = static_cast<int>(parseIntegerPower(value(), "threads"));
    else if (arg == "--input") options.input = value();
    else if (arg == "--precondition-seconds") options.preconditionSeconds = parseDoublePower(value(), "precondition-seconds");
    else if (arg == "--measure-seconds") options.measureSeconds = parseDoublePower(value(), "measure-seconds");
    else if (arg == "--sample-ms") options.sampleMs = static_cast<unsigned int>(parseIntegerPower(value(), "sample-ms"));
    else if (arg == "--inner-repeats") options.innerRepeats = static_cast<int>(parseIntegerPower(value(), "inner-repeats"));
    else if (arg == "--launch-batch") options.launchBatch = static_cast<int>(parseIntegerPower(value(), "launch-batch"));
    else if (arg == "--graph-nodes") options.graphNodes = static_cast<int>(parseIntegerPower(value(), "graph-nodes"));
    else if (arg == "--p-constant-w") options.pConstantW = parseDoublePower(value(), "p-constant-w");
    else if (arg == "--p-static-w") options.pStaticW = parseDoublePower(value(), "p-static-w");
    else if (arg == "--repetition") options.repetition = static_cast<int>(parseIntegerPower(value(), "repetition"));
    else if (arg == "--telemetry-output") options.telemetryOutput = value();
    else if (arg == "--output") options.output = value();
    else if (arg == "--help" || arg == "-h") {
      std::cout << "Usage: " << argv[0]
                << " --kind load-xor|xor-only --active-loads 64|128"
                << " [--device N --mib 1024 --blocks-per-sm 8 | --grid-blocks N]"
                << " [--threads 128|256]"
                << " [--precondition-seconds 20 --measure-seconds 60]"
                << " [--inner-repeats 64 --graph-nodes 0 --sample-ms 10]"
                << " [--p-constant-w W --p-static-w W]"
                << " [--telemetry-output FILE --output FILE]\n";
      std::exit(EXIT_SUCCESS);
    } else failPower("unknown option: " + arg);
  }
  if (options.device < 0 ||
      (options.kind != "load-xor" && options.kind != "xor-only") ||
      (options.activeLoads != 64 && options.activeLoads != 128) || options.mib == 0 ||
      options.blocksPerSm < 1 || options.gridBlocks < 0 ||
      (options.threads != 128 && options.threads != 256) || options.preconditionSeconds < 0 ||
      options.measureSeconds <= 0 || options.sampleMs == 0 || options.innerRepeats < 1 ||
      options.launchBatch < 1 || options.graphNodes < 0 || options.repetition < 1 ||
      options.pConstantW < 0 || options.pStaticW < 0)
    failPower("invalid power option");
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
  if (path.empty()) return;
  const auto parent = std::filesystem::path(path).parent_path();
  if (parent.empty()) return;
  std::error_code error;
  std::filesystem::create_directories(parent, error);
  if (error) failPower("cannot create output directory: " + parent.string());
}

void runPowerTrial(const PowerOptions& options) {
  CUDA_CHECK(cudaSetDevice(options.device));
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, options.device));
  const size_t bytes = options.mib * 1024ull * 1024ull;
  if (bytes <= static_cast<size_t>(prop.l2CacheSize))
    failPower("--mib must exceed L2 size for an HBM power trial");
  const size_t count = bytes / sizeof(packed_fp32);
  const int blocks = options.gridBlocks > 0
      ? options.gridBlocks
      : prop.multiProcessorCount * options.blocksPerSm;
  const size_t groupBytes = static_cast<size_t>(blocks) * wordsPerRegion(options.threads) * sizeof(std::uint32_t);
  const size_t groups = bytes / groupBytes;
  if (groups == 0) failPower("buffer too small for HBM stream geometry");
  const int rounds = options.activeLoads / kScalarLoadsPerRound;

  packed_fp32* data = nullptr;
  CUDA_CHECK(cudaMalloc(&data, bytes));
  // Allocating/initializing a matching footprint keeps the CUDA context and
  // memory footprint stable between xor-only and both load kernels. Xor-only never
  // reads it during the measurement graph.
  const int initBlocks = std::min<int>(65535, (count + kDefaultThreads - 1) / kDefaultThreads);
  initBits<<<initBlocks, kDefaultThreads>>>(data, count, inputMode(options.input));
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaStream_t stream{};
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  // The XOR-only control kernel is much shorter than the HBM load kernel.
  // Grow the number of full-stream repeats until one graph node is long
  // enough that a ~60 s experiment needs only O(10^3), rather than millions,
  // of graph nodes.  This does not change the r2-r1 opcode relationship.
  int powerInnerRepeats = options.innerRepeats;
  auto launch = [&](cudaStream_t target) {
    if (options.kind == "load-xor")
      launchLoadDominant(data, groups, blocks, options.threads, rounds, powerInnerRepeats, target);
    else
      launchXorOnly(groups, blocks, rounds, options.threads, powerInnerRepeats, target);
  };

  // Event-time one node. This calibration is outside the telemetry interval.
  auto timeOneNode = [&]() {
    launch(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    cudaEvent_t calibrationStart{}, calibrationStop{};
    CUDA_CHECK(cudaEventCreate(&calibrationStart));
    CUDA_CHECK(cudaEventCreate(&calibrationStop));
    CUDA_CHECK(cudaEventRecord(calibrationStart, stream));
    launch(stream);
    CUDA_CHECK(cudaEventRecord(calibrationStop, stream));
    CUDA_CHECK(cudaEventSynchronize(calibrationStop));
    float milliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, calibrationStart, calibrationStop));
    CUDA_CHECK(cudaEventDestroy(calibrationStart));
    CUDA_CHECK(cudaEventDestroy(calibrationStop));
    return milliseconds;
  };
  float nodeMilliseconds = timeOneNode();
  constexpr float kTargetNodeMilliseconds = 50.0f;
  if (options.graphNodes == 0 && nodeMilliseconds > 0.0f &&
      nodeMilliseconds < kTargetNodeMilliseconds) {
    const int multiplier = std::max(2, static_cast<int>(std::ceil(
        kTargetNodeMilliseconds / nodeMilliseconds)));
    if (powerInnerRepeats <= std::numeric_limits<int>::max() / multiplier) {
      powerInnerRepeats *= multiplier;
      nodeMilliseconds = timeOneNode();
    }
  }
  if (nodeMilliseconds <= 0.0f) failPower("unable to time graph node");

  // A captured graph can execute contiguous nodes faster than individually
  // launched kernels.  Time a small captured graph as well, so `--measure-
  // seconds` describes the actual graph execution interval rather than a
  // host-launch calibration interval.
  auto timeGraphNode = [&]() {
    cudaGraph_t calibrationGraph{};
    cudaGraphExec_t calibrationExec{};
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    for (int node = 0; node < options.launchBatch; ++node) launch(stream);
    CUDA_CHECK(cudaStreamEndCapture(stream, &calibrationGraph));
    CUDA_CHECK(cudaGraphInstantiate(&calibrationExec, calibrationGraph, nullptr, nullptr, 0));
    cudaEvent_t start{}, stop{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, stream));
    CUDA_CHECK(cudaGraphLaunch(calibrationExec, stream));
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float milliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaGraphExecDestroy(calibrationExec));
    CUDA_CHECK(cudaGraphDestroy(calibrationGraph));
    return milliseconds / options.launchBatch;
  };
  nodeMilliseconds = timeGraphNode();
  if (options.graphNodes == 0 && nodeMilliseconds > 0.0f &&
      nodeMilliseconds < kTargetNodeMilliseconds) {
    const int multiplier = std::max(2, static_cast<int>(std::ceil(
        kTargetNodeMilliseconds / nodeMilliseconds)));
    if (powerInnerRepeats <= std::numeric_limits<int>::max() / multiplier) {
      powerInnerRepeats *= multiplier;
      nodeMilliseconds = timeGraphNode();
    }
  }
  if (nodeMilliseconds <= 0.0f) failPower("unable to time CUDA-Graph node");
  const int graphNodes = options.graphNodes > 0 ? options.graphNodes
      : std::max(1, static_cast<int>(std::llround(
          options.measureSeconds / (nodeMilliseconds / 1000.0))));

  cudaGraph_t warmGraph{}, measureGraph{};
  cudaGraphExec_t warmExec{}, measureExec{};
  auto capture = [&](int nodes, cudaGraph_t* graph, cudaGraphExec_t* exec) {
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    for (int node = 0; node < nodes; ++node) launch(stream);
    CUDA_CHECK(cudaStreamEndCapture(stream, graph));
    CUDA_CHECK(cudaGraphInstantiate(exec, *graph, nullptr, nullptr, 0));
  };
  capture(options.launchBatch, &warmGraph, &warmExec);
  capture(graphNodes, &measureGraph, &measureExec);

  const auto cleanup = [&]() {
    CUDA_CHECK(cudaGraphExecDestroy(measureExec)); CUDA_CHECK(cudaGraphDestroy(measureGraph));
    CUDA_CHECK(cudaGraphExecDestroy(warmExec)); CUDA_CHECK(cudaGraphDestroy(warmGraph));
    CUDA_CHECK(cudaStreamDestroy(stream)); CUDA_CHECK(cudaFree(data));
  };

  long long preconditionNodes = 0;
  const auto preconditionEnd = std::chrono::steady_clock::now() +
      std::chrono::duration_cast<std::chrono::steady_clock::duration>(
          std::chrono::duration<double>(options.preconditionSeconds));
  do {
    CUDA_CHECK(cudaGraphLaunch(warmExec, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    preconditionNodes += options.launchBatch;
  } while (std::chrono::steady_clock::now() < preconditionEnd);

  cudaEvent_t start{}, stop{};
  CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop));
  const nvmlDevice_t nvmlDevice = nvmlForCudaDevice(options.device);
  const auto wallStart = powertelemetry::Clock::now();
  powertelemetry::Sampler sampler(nvmlDevice, options.sampleMs);
  sampler.start(wallStart);
  nvtxRangePushA(options.kind == "load-xor" ? "hbm_load_xor_power_graph" : "hbm_xor_only_power_graph");
  CUDA_CHECK(cudaEventRecord(start, stream));
  CUDA_CHECK(cudaGraphLaunch(measureExec, stream));
  CUDA_CHECK(cudaEventRecord(stop, stream));
  CUDA_CHECK(cudaEventSynchronize(stop));
  nvtxRangePop();
  const auto wallStop = powertelemetry::Clock::now();
  sampler.stop();
  float activeMilliseconds = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&activeMilliseconds, start, stop));
  CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop));
  const double activeSeconds = activeMilliseconds / 1000.0;
  const auto telemetry = sampler.summarize(
      std::chrono::duration<double>(wallStop - wallStart).count());
  if (!options.telemetryOutput.empty()) {
    ensureParentDirectory(options.telemetryOutput);
    sampler.write_csv(options.telemetryOutput);
  }

  const double workGroups = static_cast<double>(graphNodes) * powerInnerRepeats *
      groups * blocks * (options.threads / 32);  // warp × group instances
  // `activeLoads` is already expressed in the executed warp-level SASS LDG
  // count. NCU's Executed Instruction Mix reports 64/128 LDG for r1/r2,
  // respectively, per warp/group. Keep this telemetry accounting in that
  // exact same unit.
  constexpr double kExecutedSassPerPtxSlot = 1.0;
  const double variableLdg =
      options.kind == "load-xor"
      ? static_cast<double>(options.activeLoads) * workGroups * kExecutedSassPerPtxSlot
      : 0.0;
  // The fixed LOP3 prologue in xor-only is deliberately excluded: this is the
  // r1/r2-varying instruction count used for the regression slope.
  const double lop3PerLdg = 0.5;
  const double variableLop3 =
      static_cast<double>(options.activeLoads) * lop3PerLdg * workGroups *
      kExecutedSassPerPtxSlot;
  const double ldgRate = variableLdg / activeSeconds;
  const double lop3Rate = variableLop3 / activeSeconds;
  const double boardEnergy = telemetry.time_weighted_avg_w * activeSeconds;
  const double afterConstEnergy = (telemetry.time_weighted_avg_w - options.pConstantW) * activeSeconds;
  const double afterConstStaticEnergy =
      (telemetry.time_weighted_avg_w - options.pConstantW - options.pStaticW) * activeSeconds;

  const char* header =
      "kind,active_loads,sass_count_unit,repetition,graph_nodes,inner_repeats,precondition_nodes,"
      "kernel_active_s,variable_ldg_warp_instructions,variable_lop3_warp_instructions,"
      "variable_ldg_warp_instr_per_s,variable_lop3_warp_instr_per_s,board_avg_w,"
      "p_constant_w,p_static_w,board_energy_j,energy_after_const_j,"
      "energy_after_const_static_j,util_avg_pct,sm_clock_avg_mhz,sm_clock_min_mhz,"
      "sm_clock_max_mhz,mem_clock_avg_mhz,temp_max_c,sample_count,throttle_any_pct,"
      "throttle_sw_power_pct,throttle_hw_slowdown_pct,throttle_hw_thermal_pct,telemetry_csv,"
      "grid_blocks,threads_per_block,groups";
  std::ostringstream row;
  row << options.kind << ',' << options.activeLoads << ",executed_warp_sass," << options.repetition << ','
      << graphNodes << ',' << powerInnerRepeats << ',' << preconditionNodes << ','
      << std::fixed << std::setprecision(9) << activeSeconds << ',' << variableLdg << ','
      << variableLop3 << ',' << ldgRate << ',' << lop3Rate << ','
      << telemetry.time_weighted_avg_w << ',' << options.pConstantW << ','
      << options.pStaticW << ',' << boardEnergy << ',' << afterConstEnergy << ','
      << afterConstStaticEnergy << ',' << telemetry.util_avg_pct << ','
      << telemetry.sm_clock_avg_mhz << ',' << telemetry.sm_clock_min_mhz << ','
      << telemetry.sm_clock_max_mhz << ',' << telemetry.mem_clock_avg_mhz << ','
      << telemetry.temp_max_c << ',' << telemetry.sample_count << ','
      << telemetry.throttle_any_pct << ',' << telemetry.throttle_sw_power_pct << ','
      << telemetry.throttle_hw_slowdown_pct << ',' << telemetry.throttle_hw_thermal_pct << ','
      << options.telemetryOutput << ',' << blocks << ',' << options.threads << ',' << groups;
  if (!options.output.empty()) {
    ensureParentDirectory(options.output);
    const bool writeHeader = !hasContent(options.output);
    std::ofstream output(options.output, std::ios::app);
    if (!output) failPower("cannot open output: " + options.output);
    if (writeHeader) output << header << '\n';
    output << row.str() << '\n';
  }

  std::cout << std::fixed << std::setprecision(3)
            << "SASS power trial: kind=" << options.kind
            << " active_loads=" << options.activeLoads
            << " grid_blocks=" << blocks << " threads=" << options.threads
            << " graph_nodes=" << graphNodes << " inner_repeats=" << powerInnerRepeats
            << " active_s=" << activeSeconds << " board=" << telemetry.time_weighted_avg_w
            << "W ldg_rate=" << ldgRate / 1.0e9 << " Gwarp-inst/s"
            << " lop3_rate=" << lop3Rate / 1.0e9 << " Gwarp-inst/s\n";
  cleanup();
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
