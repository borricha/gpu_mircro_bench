// Static-power calibration using the PTX nanosleep instruction.
//
// This follows the Wattchmen methodology: first measure P_const while the
// CUDA context is idle, then keep every SM populated by sleeping warps.  The
// latter consumes no global memory bandwidth and has no target ALU operation;
// its board power minus P_const is the shared-resource/static-power estimate.

#include <cuda_runtime.h>
#include <nvml.h>
#include <nvtx3/nvtx3.hpp>

#include "../common/power_telemetry.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    const cudaError_t error__ = (call);                                      \
    if (error__ != cudaSuccess)                                              \
      throw std::runtime_error(std::string(#call) + ": " +                  \
                               cudaGetErrorString(error__));                 \
  } while (0)

namespace {

struct Options {
  int device = 0;
  // Physical NVML index used only for the no-CUDA-context P_const interval.
  // It must be explicit because CUDA_VISIBLE_DEVICES remaps CUDA indices.
  int nvmlDevice = -1;
  double idleSeconds = 60.0;
  double preconditionSeconds = 0.0;
  double measureSeconds = 180.0;
  unsigned int sampleMs = 10;
  // A negative value requests an in-process CUDA-context-idle measurement.
  double pConstantW = -1.0;
  unsigned int sleepNs = 1'000'000;
  int threads = 256;
  int blocksPerSm = 8;
  // Exact grid override for matching a low-concurrency benchmark.  A grid of
  // 57 CTAs on a 114-SM H100 can occupy at most half of the SMs at once.
  int gridBlocks = 0;
  int repetition = 1;
  std::string idleTelemetryOutput;
  std::string telemetryOutput;
  std::string output;
};

struct StaticNvtxDomain {
  static constexpr char const* name{"StaticNanosleepBench"};
};
using StaticNvtxRange = nvtx3::scoped_range_in<StaticNvtxDomain>;

[[noreturn]] void fail(const std::string& message) {
  throw std::runtime_error(message);
}

void nvmlCheck(nvmlReturn_t status, const char* what) {
  if (status != NVML_SUCCESS)
    fail(std::string(what) + ": " + nvmlErrorString(status));
}

long long parseInteger(const char* text, const char* name) {
  char* end = nullptr;
  const long long value = std::strtoll(text, &end, 10);
  if (!end || *end != '\0') fail(std::string("invalid ") + name + ": " + text);
  return value;
}

double parseDouble(const char* text, const char* name) {
  char* end = nullptr;
  const double value = std::strtod(text, &end);
  if (!end || *end != '\0' || !std::isfinite(value))
    fail(std::string("invalid ") + name + ": " + text);
  return value;
}

void usage(const char* program) {
  std::cout
      << "Usage: " << program << " [options]\n"
      << "  --device N                  CUDA logical device (default: 0)\n"
      << "  --nvml-device N             physical NVML GPU for auto P_const\n"
      << "  --idle-seconds S            auto P_const sampling duration (default: 60)\n"
      << "  --precondition-seconds S    nanosleep settling before sampling (default: 0)\n"
      << "  --measure-seconds S         nanosleep sample target (default: 180)\n"
      << "  --sample-ms N               requested NVML sample period (default: 10)\n"
      << "  --p-constant-w W            use known P_const; omit for auto-idle P_const\n"
      << "  --sleep-ns N                nanosleep request per loop iteration (default: 1000000)\n"
      << "  --threads N                 threads/block, multiple of 32 (default: 256)\n"
      << "  --blocks-per-sm N           blocks launched per SM (default: 8)\n"
      << "  --grid-blocks N             exact CTA grid override (default: derived)\n"
      << "  --repetition N              CSV trial label (default: 1)\n"
      << "  --idle-telemetry-output F   auto P_const raw NVML samples CSV\n"
      << "  --telemetry-output F        nanosleep raw NVML samples CSV\n"
      << "  --output F                  append one static-power summary CSV row\n";
}

Options parseOptions(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto value = [&]() -> const char* {
      if (++i >= argc) fail("missing value for " + arg);
      return argv[i];
    };
    if (arg == "--device")
      options.device = static_cast<int>(parseInteger(value(), "device"));
    else if (arg == "--nvml-device")
      options.nvmlDevice = static_cast<int>(parseInteger(value(), "nvml-device"));
    else if (arg == "--idle-seconds")
      options.idleSeconds = parseDouble(value(), "idle-seconds");
    else if (arg == "--precondition-seconds")
      options.preconditionSeconds = parseDouble(value(), "precondition-seconds");
    else if (arg == "--measure-seconds")
      options.measureSeconds = parseDouble(value(), "measure-seconds");
    else if (arg == "--sample-ms")
      options.sampleMs = static_cast<unsigned int>(parseInteger(value(), "sample-ms"));
    else if (arg == "--p-constant-w")
      options.pConstantW = parseDouble(value(), "p-constant-w");
    else if (arg == "--sleep-ns")
      options.sleepNs = static_cast<unsigned int>(parseInteger(value(), "sleep-ns"));
    else if (arg == "--threads")
      options.threads = static_cast<int>(parseInteger(value(), "threads"));
    else if (arg == "--blocks-per-sm")
      options.blocksPerSm = static_cast<int>(parseInteger(value(), "blocks-per-sm"));
    else if (arg == "--grid-blocks")
      options.gridBlocks = static_cast<int>(parseInteger(value(), "grid-blocks"));
    else if (arg == "--repetition")
      options.repetition = static_cast<int>(parseInteger(value(), "repetition"));
    else if (arg == "--idle-telemetry-output")
      options.idleTelemetryOutput = value();
    else if (arg == "--telemetry-output")
      options.telemetryOutput = value();
    else if (arg == "--output")
      options.output = value();
    else if (arg == "--help" || arg == "-h") {
      usage(argv[0]);
      std::exit(EXIT_SUCCESS);
    } else {
      fail("unknown option: " + arg);
    }
  }
  if (options.device < 0 || options.nvmlDevice < -1 || options.idleSeconds <= 0.0 ||
      options.preconditionSeconds < 0.0 || options.measureSeconds <= 0.0 ||
      options.sampleMs == 0 || options.pConstantW < -1.0 || options.sleepNs == 0 ||
      options.threads < 32 || options.threads > 1024 || options.threads % 32 != 0 ||
      options.blocksPerSm < 1 || options.gridBlocks < 0 || options.repetition < 1)
    fail("invalid option value");
  return options;
}

__device__ __forceinline__ void nanosleepPtx(unsigned int nanoseconds) {
  // PTX nanosleep.u32 is supported on sm_70+.  The volatile inline assembly
  // prevents the compiler from deleting or hoisting this deliberate stall.
  asm volatile("nanosleep.u32 %0;" : : "r"(nanoseconds));
}

__global__ void nanosleepKernel(std::uint64_t iterations, unsigned int sleepNs) {
  // Every thread executes exactly the same sleep-only loop.  There are no
  // loads, stores, FP operations, or Tensor Core operations in this kernel.
  for (std::uint64_t iteration = 0; iteration < iterations; ++iteration)
    nanosleepPtx(sleepNs);
}

std::uint64_t iterationsFor(double seconds, unsigned int sleepNs) {
  const long double requested = std::ceil(
      static_cast<long double>(seconds) * 1.0e9L / static_cast<long double>(sleepNs));
  if (requested > static_cast<long double>(std::numeric_limits<std::uint64_t>::max()))
    fail("requested duration is too large");
  return std::max<std::uint64_t>(1, static_cast<std::uint64_t>(requested));
}

nvmlDevice_t nvmlDeviceForCudaDevice(int cudaDevice) {
  char pciBusId[32]{};
  CUDA_CHECK(cudaDeviceGetPCIBusId(pciBusId, sizeof(pciBusId), cudaDevice));
  nvmlDevice_t device{};
  nvmlCheck(nvmlDeviceGetHandleByPciBusId_v2(pciBusId, &device),
            "nvmlDeviceGetHandleByPciBusId_v2");
  return device;
}

std::string nvmlUuid(nvmlDevice_t device) {
  char uuid[NVML_DEVICE_UUID_BUFFER_SIZE]{};
  nvmlCheck(nvmlDeviceGetUUID(device, uuid, sizeof(uuid)), "nvmlDeviceGetUUID");
  return uuid;
}

struct MeasuredInterval {
  powertelemetry::Summary telemetry;
  double wallSeconds = 0.0;
  double gpuSeconds = std::numeric_limits<double>::quiet_NaN();
};

MeasuredInterval measureIdle(nvmlDevice_t device, const Options& options) {
  const auto start = std::chrono::steady_clock::now();
  powertelemetry::Sampler sampler(device, options.sampleMs);
  sampler.start(start);
  std::this_thread::sleep_for(std::chrono::duration<double>(options.idleSeconds));
  const auto stop = std::chrono::steady_clock::now();
  sampler.stop();
  const double seconds = std::chrono::duration<double>(stop - start).count();
  sampler.write_csv(options.idleTelemetryOutput);
  return {sampler.summarize(seconds), seconds};
}

void launchNanosleep(std::uint64_t iterations, unsigned int sleepNs, int blocks,
                     int threads, cudaStream_t stream) {
  nanosleepKernel<<<blocks, threads, 0, stream>>>(iterations, sleepNs);
  CUDA_CHECK(cudaPeekAtLastError());
}

MeasuredInterval measureNanosleep(nvmlDevice_t device, const Options& options,
                                  std::uint64_t iterations, int blocks) {
  cudaStream_t stream{};
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  cudaEvent_t startEvent{}, stopEvent{};
  CUDA_CHECK(cudaEventCreate(&startEvent));
  CUDA_CHECK(cudaEventCreate(&stopEvent));

  const auto wallStart = std::chrono::steady_clock::now();
  powertelemetry::Sampler sampler(device, options.sampleMs);
  sampler.start(wallStart);
  {
    StaticNvtxRange range{"NANOSLEEP_STATIC_MEASURE"};
    CUDA_CHECK(cudaEventRecord(startEvent, stream));
    launchNanosleep(iterations, options.sleepNs, blocks, options.threads, stream);
    CUDA_CHECK(cudaEventRecord(stopEvent, stream));
    CUDA_CHECK(cudaEventSynchronize(stopEvent));
  }
  const auto wallStop = std::chrono::steady_clock::now();
  sampler.stop();
  float milliseconds = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&milliseconds, startEvent, stopEvent));
  const double wallSeconds =
      std::chrono::duration<double>(wallStop - wallStart).count();
  sampler.write_csv(options.telemetryOutput);
  CUDA_CHECK(cudaEventDestroy(startEvent));
  CUDA_CHECK(cudaEventDestroy(stopEvent));
  CUDA_CHECK(cudaStreamDestroy(stream));
  return {sampler.summarize(wallSeconds), wallSeconds, milliseconds / 1000.0};
}

void preconditionNanosleep(const Options& options, std::uint64_t iterations,
                           int blocks) {
  if (options.preconditionSeconds <= 0.0) return;
  const std::uint64_t preconditionIterations =
      iterationsFor(options.preconditionSeconds, options.sleepNs);
  cudaStream_t stream{};
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  StaticNvtxRange range{"NANOSLEEP_STATIC_PRECONDITION"};
  launchNanosleep(preconditionIterations, options.sleepNs, blocks, options.threads, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaStreamDestroy(stream));
}

bool hasContent(const std::string& path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  return input && input.tellg() > 0;
}

std::string timestamp() {
  const auto now = std::chrono::system_clock::now();
  const std::time_t raw = std::chrono::system_clock::to_time_t(now);
  std::tm local{};
  localtime_r(&raw, &local);
  char text[32];
  std::strftime(text, sizeof(text), "%Y-%m-%dT%H:%M:%S%z", &local);
  return text;
}

void appendSummary(const Options& options, const cudaDeviceProp& prop,
                   int residentBlocksPerSm, std::uint64_t iterations,
                   const MeasuredInterval& idle, double pConstantW,
                   const MeasuredInterval& active) {
  const double staticPowerW = active.telemetry.time_weighted_avg_w - pConstantW;
  const double staticEnergyJ = staticPowerW * active.gpuSeconds;
  const char* header =
      "timestamp,device_index,gpu_name,repetition,sm_count,threads_per_block,"
      "requested_blocks_per_sm,requested_grid_blocks,launched_blocks,resident_blocks_per_sm,"
      "sleep_ns,iterations,"
      "p_constant_source,p_constant_w,idle_elapsed_s,idle_power_avg_w,"
      "nanosleep_kernel_active_s,nanosleep_wall_elapsed_s,nanosleep_power_avg_w,"
      "static_power_w,static_energy_j,util_avg_pct,sm_clock_avg_mhz,"
      "sm_clock_min_mhz,sm_clock_max_mhz,mem_clock_avg_mhz,temp_max_c,"
      "sample_count,throttle_any_pct,throttle_sw_power_pct,"
      "throttle_hw_slowdown_pct,throttle_sw_thermal_pct,"
      "throttle_hw_thermal_pct,throttle_app_clock_pct,throttle_reasons_or";
  std::ostringstream row;
  row << timestamp() << ',' << options.device << ',' << prop.name << ','
      << options.repetition << ',' << prop.multiProcessorCount << ','
      << options.threads << ',' << options.blocksPerSm << ',' << options.gridBlocks << ','
      << (options.gridBlocks > 0 ? options.gridBlocks : prop.multiProcessorCount * options.blocksPerSm)
      << ',' << residentBlocksPerSm << ',' << options.sleepNs << ',' << iterations << ','
      << (options.pConstantW >= 0.0 ? "provided" : "auto_cuda_idle") << ','
      << std::fixed << std::setprecision(6) << pConstantW << ','
      << idle.wallSeconds << ',' << idle.telemetry.time_weighted_avg_w << ','
      << active.gpuSeconds << ',' << active.wallSeconds << ','
      << active.telemetry.time_weighted_avg_w << ',' << staticPowerW << ','
      << staticEnergyJ << ',' << active.telemetry.util_avg_pct << ','
      << active.telemetry.sm_clock_avg_mhz << ','
      << active.telemetry.sm_clock_min_mhz << ','
      << active.telemetry.sm_clock_max_mhz << ','
      << active.telemetry.mem_clock_avg_mhz << ',' << active.telemetry.temp_max_c
      << ',' << active.telemetry.sample_count << ','
      << active.telemetry.throttle_any_pct << ','
      << active.telemetry.throttle_sw_power_pct << ','
      << active.telemetry.throttle_hw_slowdown_pct << ','
      << active.telemetry.throttle_sw_thermal_pct << ','
      << active.telemetry.throttle_hw_thermal_pct << ','
      << active.telemetry.throttle_app_clock_pct << ','
      << active.telemetry.throttle_reasons_or;
  if (!options.output.empty()) {
    const bool writeHeader = !hasContent(options.output);
    std::ofstream output(options.output, std::ios::app);
    if (!output) fail("cannot open output: " + options.output);
    if (writeHeader) output << header << '\n';
    output << row.str() << '\n';
  }
  std::cout << std::fixed << std::setprecision(3)
            << "NanoSleep static trial: " << active.telemetry.time_weighted_avg_w
            << " W board - " << pConstantW << " W P_const = " << staticPowerW
            << " W static (" << active.gpuSeconds << " s kernel, "
            << options.threads * residentBlocksPerSm << " resident threads/SM, "
            << "throttle=" << active.telemetry.throttle_any_pct << "%)\n";
}

}  // namespace

int main(int argc, char** argv) {
  bool nvmlInitialized = false;
  try {
    const Options options = parseOptions(argc, argv);
    // Do this before *any* CUDA runtime call. CUDA_VISIBLE_DEVICES remaps
    // --device, so the physical NVML index is intentionally a separate,
    // explicit option. This is the only way the auto P_const sample can be a
    // genuine lowest-power-state/no-CUDA-context measurement.
    nvmlCheck(nvmlInit_v2(), "nvmlInit_v2");
    nvmlInitialized = true;

    MeasuredInterval idle;
    double pConstantW = options.pConstantW;
    nvmlDevice_t idleNvmlDevice{};
    std::string idleUuid;
    if (pConstantW < 0.0) {
      if (options.nvmlDevice < 0) {
        fail("auto P_const requires --nvml-device PHYSICAL_INDEX; "
             "this prevents CUDA_VISIBLE_DEVICES from selecting the wrong GPU");
      }
      nvmlCheck(nvmlDeviceGetHandleByIndex_v2(options.nvmlDevice, &idleNvmlDevice),
                "nvmlDeviceGetHandleByIndex_v2(auto P_const)");
      idleUuid = nvmlUuid(idleNvmlDevice);
      std::cout << "Measuring no-CUDA-context P_const on physical NVML GPU "
                << options.nvmlDevice << " for " << options.idleSeconds << " s\n";
      idle = measureIdle(idleNvmlDevice, options);
      pConstantW = idle.telemetry.time_weighted_avg_w;
    }

    CUDA_CHECK(cudaSetDevice(options.device));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, options.device));
    if (prop.major < 7) fail("nanosleep.u32 requires sm_70 or newer");
    const nvmlDevice_t nvmlDevice = nvmlDeviceForCudaDevice(options.device);
    if (options.pConstantW < 0.0 && nvmlUuid(nvmlDevice) != idleUuid) {
      fail("--nvml-device does not identify the CUDA logical --device after "
           "CUDA_VISIBLE_DEVICES remapping");
    }

    int maxBlocksPerSm = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxBlocksPerSm, nanosleepKernel, options.threads, 0));
    const int residentBlocksPerSm = std::min(options.blocksPerSm, maxBlocksPerSm);
    if (residentBlocksPerSm < options.blocksPerSm) {
      std::cerr << "Warning: requested " << options.blocksPerSm << " blocks/SM, but "
                << "this launch can residently hold " << maxBlocksPerSm << " blocks/SM.\n";
    }
    const int blocks = options.gridBlocks > 0
        ? options.gridBlocks
        : prop.multiProcessorCount * options.blocksPerSm;
    const std::uint64_t iterations =
        iterationsFor(options.measureSeconds, options.sleepNs);

    preconditionNanosleep(options, iterations, blocks);
    const MeasuredInterval active =
        measureNanosleep(nvmlDevice, options, iterations, blocks);
    appendSummary(options, prop, residentBlocksPerSm, iterations, idle, pConstantW, active);

    nvmlCheck(nvmlShutdown(), "nvmlShutdown");
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    if (nvmlInitialized) nvmlShutdown();
    std::cerr << "Error: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
