#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    const cudaError_t error__ = (call);                                      \
    if (error__ != cudaSuccess)                                              \
      throw std::runtime_error(std::string(#call) + ": " +                 \
                               cudaGetErrorString(error__));                 \
  } while (0)

namespace hbmv2 {

using packed_fp32 = uint4;

// HBM v1와 동일한 256-thread CTA와 기본 8 CTA/SM을 사용한다.
constexpr int kThreads = 256;
constexpr int kDefaultBlocksPerSm = 8;
// L1 v2와 동일한 differential-pass 형태다. 한 pass는 thread당 64개의
// 128-bit HBM load이며, r1/r2는 pass 1/2개(64/128 loads)를 수행한다.
// body가 충분히 커서 ptxas가 runtime round 선택을 predicate가 아니라 실제
// branch로 유지하도록 한다.
constexpr int kVectorLoadsPerRound = 64;
constexpr int kMaxRounds = 2;
constexpr int kVectorsPerGroup = kVectorLoadsPerRound * kMaxRounds;

struct Options {
  int device = 0;
  size_t mib = 1024;
  int blocksPerSm = kDefaultBlocksPerSm;
  int activeLoads = kVectorLoadsPerRound;
  std::string input = "random";
  int trials = 1;
};

[[noreturn]] inline void fail(const std::string& message) {
  throw std::runtime_error(message);
}

inline long long parseInteger(const char* text, const char* name) {
  char* end = nullptr;
  const long long value = std::strtoll(text, &end, 10);
  if (end == nullptr || *end != '\0') fail(std::string("invalid ") + name);
  return value;
}

inline int inputMode(const std::string& input) {
  if (input == "random") return 0;
  if (input == "zero") return 1;
  if (input == "one-point-one") return 2;
  fail("invalid --input (expected random, zero, or one-point-one)");
}

inline Options parseOptions(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto value = [&]() -> const char* {
      if (++i >= argc) fail("missing value for " + arg);
      return argv[i];
    };
    if (arg == "--device") options.device = static_cast<int>(parseInteger(value(), "device"));
    else if (arg == "--mib") options.mib = static_cast<size_t>(parseInteger(value(), "mib"));
    else if (arg == "--blocks-per-sm") options.blocksPerSm = static_cast<int>(parseInteger(value(), "blocks-per-sm"));
    else if (arg == "--active-loads") options.activeLoads = static_cast<int>(parseInteger(value(), "active-loads"));
    else if (arg == "--rounds") options.activeLoads = static_cast<int>(parseInteger(value(), "rounds")) * kVectorLoadsPerRound;
    else if (arg == "--trials") options.trials = static_cast<int>(parseInteger(value(), "trials"));
    else if (arg == "--input") options.input = value();
    else if (arg == "--help" || arg == "-h") fail("help");
    else fail("unknown option: " + arg);
  }
  if (options.device < 0 || options.mib == 0 || options.blocksPerSm < 1 ||
      options.trials < 1 || (options.activeLoads != 64 && options.activeLoads != 128))
    fail("invalid option value");
  inputMode(options.input);
  return options;
}

__device__ __forceinline__ std::uint32_t mixBits(std::uint32_t x) {
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  return x ^ (x >> 16);
}

__global__ void initBits(packed_fp32* data, size_t count, int mode) {
  constexpr std::uint32_t kOnePointOne = 0x3f8ccccdu;
  for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count; i += static_cast<size_t>(gridDim.x) * blockDim.x) {
    const auto base = static_cast<std::uint32_t>(i * 4);
    data[i] = mode == 1 ? make_uint4(0, 0, 0, 0)
        : mode == 2 ? make_uint4(kOnePointOne, kOnePointOne, kOnePointOne, kOnePointOne)
                    : make_uint4(mixBits(base), mixBits(base + 1),
                                 mixBits(base + 2), mixBits(base + 3));
  }
}

inline void printUsage(const char* program, const char* description) {
  std::cout << "Usage: " << program
            << " [--device N] [--mib N] [--blocks-per-sm N]"
            << " [--active-loads 64|128 | --rounds 1|2]"
            << " [--trials N] [--input random|zero|one-point-one]\n"
            << description << '\n';
}

// HBM's inline PTX helpers and the two kernels live in this one shared header.
// Keeping them here makes HBM match L1/L2: two experiment CUDA files plus
// one common.cuh used by both the regular and power executables.

// The LDG-dominant kernel is intentionally specialized for this A100's
// normal 108 SM x 8 CTA/SM geometry. Every load address is a compile-time
// immediate, so one pass issues 64 LDG.E.128 and only 32 LOP3, without an
// integer pointer update between individual loads. One add advances the
// cursor by the complete 64-vector pass.
constexpr int kDominantFixedGridBlocks = 864;
constexpr std::uint64_t kDominantGridStrideBytes = 0x00360000ull;
constexpr std::uint64_t kDominantPassBytes = 0x0d800000ull;
#define HBM_LOAD_DOMINANT_STATIC \
  "  ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%8+0x0];\n" \
  "  ld.global.cg.v4.u32 {%4, %5, %6, %7}, [%8+0x360000];\n" \
  "  lop3.b32 %9, %9, %0, %4, 0x96;\n" \
  "  ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%8+0x6c0000];\n" \
  "  ld.global.cg.v4.u32 {%4, %5, %6, %7}, [%8+0xa20000];\n" \
  "  lop3.b32 %9, %9, %0, %4, 0x96;\n" \
  "  ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%8+0xd80000];\n" \
  "  ld.global.cg.v4.u32 {%4, %5, %6, %7}, [%8+0x10e0000];\n" \
  "  lop3.b32 %9, %9, %0, %4, 0x96;\n" \
  "  ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%8+0x1440000];\n" \
  "  ld.global.cg.v4.u32 {%4, %5, %6, %7}, [%8+0x17a0000];\n" \
  "  lop3.b32 %9, %9, %0, %4, 0x96;\n" \
  "  ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%8+0x1b00000];\n" \
  "  ld.global.cg.v4.u32 {%4, %5, %6, %7}, [%8+0x1e60000];\n" \
  "  lop3.b32 %9, %9, %0, %4, 0x96;\n" \
  "  ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%8+0x21c0000];\n" \
  "  ld.global.cg.v4.u32 {%4, %5, %6, %7}, [%8+0x2520000];\n" \
  "  lop3.b32 %9, %9, %0, %4, 0x96;\n" \
  "  ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%8+0x2880000];\n" \
  "  ld.global.cg.v4.u32 {%4, %5, %6, %7}, [%8+0x2be0000];\n" \
  "  lop3.b32 %9, %9, %0, %4, 0x96;\n" \
  "  ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%8+0x2f40000];\n" \
  "  ld.global.cg.v4.u32 {%4, %5, %6, %7}, [%8+0x32a0000];\n" \
  "  lop3.b32 %9, %9, %0, %4, 0x96;\n" \
  "  ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%8+0x3600000];\n" \
  "  ld.global.cg.v4.u32 {%4, %5, %6, %7}, [%8+0x3960000];\n" \
  "  lop3.b32 %9, %9, %0, %4, 0x96;\n" \
  "  ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%8+0x3cc0000];\n" \
  "  ld.global.cg.v4.u32 {%4, %5, %6, %7}, [%8+0x4020000];\n" \
  "  lop3.b32 %9, %9, %0, %4, 0x96;\n" \
  "  ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%8+0x4380000];\n" \
  "  ld.global.cg.v4.u32 {%4, %5, %6, %7}, [%8+0x46e0000];\n" \
  "  lop3.b32 %9, %9, %0, %4, 0x96;\n" \
  "  ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%8+0x4a40000];\n" \
  "  ld.global.cg.v4.u32 {%4, %5, %6, %7}, [%8+0x4da0000];\n" \
  "  lop3.b32 %9, %9, %0, %4, 0x96;\n" \
  "  ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%8+0x5100000];\n" \
  "  ld.global.cg.v4.u32 {%4, %5, %6, %7}, [%8+0x5460000];\n" \
  "  lop3.b32 %9, %9, %0, %4, 0x96;\n" \
  "  ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%8+0x57c0000];\n" \
  "  ld.global.cg.v4.u32 {%4, %5, %6, %7}, [%8+0x5b20000];\n" \
  "  lop3.b32 %9, %9, %0, %4, 0x96;\n" \
  "  ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%8+0x5e80000];\n" \
  "  ld.global.cg.v4.u32 {%4, %5, %6, %7}, [%8+0x61e0000];\n" \
  "  lop3.b32 %9, %9, %0, %4, 0x96;\n" \
  "  ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%8+0x6540000];\n" \
  "  ld.global.cg.v4.u32 {%4, %5, %6, %7}, [%8+0x68a0000];\n" \
  "  lop3.b32 %9, %9, %0, %4, 0x96;\n" \
  "  ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%8+0x6c00000];\n" \
  "  ld.global.cg.v4.u32 {%4, %5, %6, %7}, [%8+0x6f60000];\n" \
  "  lop3.b32 %9, %9, %0, %4, 0x96;\n" \
  "  ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%8+0x72c0000];\n" \
  "  ld.global.cg.v4.u32 {%4, %5, %6, %7}, [%8+0x7620000];\n" \
  "  lop3.b32 %9, %9, %0, %4, 0x96;\n" \
  "  ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%8+0x7980000];\n" \
  "  ld.global.cg.v4.u32 {%4, %5, %6, %7}, [%8+0x7ce0000];\n" \
  "  lop3.b32 %9, %9, %0, %4, 0x96;\n" \
  "  ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%8+0x8040000];\n" \
  "  ld.global.cg.v4.u32 {%4, %5, %6, %7}, [%8+0x83a0000];\n" \
  "  lop3.b32 %9, %9, %0, %4, 0x96;\n" \
  "  ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%8+0x8700000];\n" \
  "  ld.global.cg.v4.u32 {%4, %5, %6, %7}, [%8+0x8a60000];\n" \
  "  lop3.b32 %9, %9, %0, %4, 0x96;\n" \
  "  ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%8+0x8dc0000];\n" \
  "  ld.global.cg.v4.u32 {%4, %5, %6, %7}, [%8+0x9120000];\n" \
  "  lop3.b32 %9, %9, %0, %4, 0x96;\n" \
  "  ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%8+0x9480000];\n" \
  "  ld.global.cg.v4.u32 {%4, %5, %6, %7}, [%8+0x97e0000];\n" \
  "  lop3.b32 %9, %9, %0, %4, 0x96;\n" \
  "  ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%8+0x9b40000];\n" \
  "  ld.global.cg.v4.u32 {%4, %5, %6, %7}, [%8+0x9ea0000];\n" \
  "  lop3.b32 %9, %9, %0, %4, 0x96;\n" \
  "  ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%8+0xa200000];\n" \
  "  ld.global.cg.v4.u32 {%4, %5, %6, %7}, [%8+0xa560000];\n" \
  "  lop3.b32 %9, %9, %0, %4, 0x96;\n" \
  "  ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%8+0xa8c0000];\n" \
  "  ld.global.cg.v4.u32 {%4, %5, %6, %7}, [%8+0xac20000];\n" \
  "  lop3.b32 %9, %9, %0, %4, 0x96;\n" \
  "  ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%8+0xaf80000];\n" \
  "  ld.global.cg.v4.u32 {%4, %5, %6, %7}, [%8+0xb2e0000];\n" \
  "  lop3.b32 %9, %9, %0, %4, 0x96;\n" \
  "  ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%8+0xb640000];\n" \
  "  ld.global.cg.v4.u32 {%4, %5, %6, %7}, [%8+0xb9a0000];\n" \
  "  lop3.b32 %9, %9, %0, %4, 0x96;\n" \
  "  ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%8+0xbd00000];\n" \
  "  ld.global.cg.v4.u32 {%4, %5, %6, %7}, [%8+0xc060000];\n" \
  "  lop3.b32 %9, %9, %0, %4, 0x96;\n" \
  "  ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%8+0xc3c0000];\n" \
  "  ld.global.cg.v4.u32 {%4, %5, %6, %7}, [%8+0xc720000];\n" \
  "  lop3.b32 %9, %9, %0, %4, 0x96;\n" \
  "  ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%8+0xca80000];\n" \
  "  ld.global.cg.v4.u32 {%4, %5, %6, %7}, [%8+0xcde0000];\n" \
  "  lop3.b32 %9, %9, %0, %4, 0x96;\n" \
  "  ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%8+0xd140000];\n" \
  "  ld.global.cg.v4.u32 {%4, %5, %6, %7}, [%8+0xd4a0000];\n" \
  "  lop3.b32 %9, %9, %0, %4, 0x96;\n"
// Four dependent bitwise consumers.  Repeating this 32 times makes one
// control pass exactly 128 LOP3, matching the LOP3 increment of one 64-LDG
// load pass (four FP32 lanes consumed per pair).
#define HBM_XOR_4 \
  "  lop3.b32 %0, %0, %1, %2, 0x96;\n" \
  "  lop3.b32 %1, %1, %2, %3, 0x96;\n" \
  "  lop3.b32 %2, %2, %3, %0, 0x96;\n" \
  "  lop3.b32 %3, %3, %0, %1, 0x96;\n"
#define HBM_XOR_8 HBM_XOR_4 HBM_XOR_4
#define HBM_XOR_16 HBM_XOR_8 HBM_XOR_8
#define HBM_XOR_32 HBM_XOR_16 HBM_XOR_16
#define HBM_XOR_64 HBM_XOR_32 HBM_XOR_32
#define HBM_XOR_128 HBM_XOR_64 HBM_XOR_64

// 64 LDG.E.128 + 32 LOP3 + one cursor advance. Each pair consumes only the
// first lane of two explicit v4 loads, forcing both 128-bit LDG instructions
// to remain while keeping LDG the dominant executed opcode.
__device__ __forceinline__ void loadDominantPass(std::uint32_t& state,
                                                 const packed_fp32*& cursor) {
  [[maybe_unused]] std::uint32_t a, b, c, d, e, f, g, h;
  asm volatile("{\n" HBM_LOAD_DOMINANT_STATIC "}"
      : "=&r"(a), "=&r"(b), "=&r"(c), "=&r"(d),
        "=&r"(e), "=&r"(f), "=&r"(g), "=&r"(h), "+l"(cursor), "+r"(state)
      : : "memory");
  asm volatile("add.u64 %0, %0, 0x0d800000;" : "+l"(cursor));
}

// r1's no-load second slot advances exactly the same 64-vector distance as
// a loadDominantPass, but emits no memory or data-consumer instruction.
__device__ __forceinline__ void advanceDominantCursor(const packed_fp32*& cursor) {
  asm volatile("add.u64 %0, %0, 0x0d800000;" : "+l"(cursor));
}

// The immediate-address second pass in r2 still requires address/control
// instructions. r1 executes this no-load counterpart so those non-memory
// opcodes are identical in r1/r2 and disappear from the energy difference.
__device__ __forceinline__ void r1DominantStaticPadding(
    int rounds, std::uint32_t& state, std::uint32_t seed) {
  asm volatile("{\n"
               "  add.u32 %0, %0, %1;\n" "  add.u32 %0, %0, %1;\n"
               "  add.u32 %0, %0, %1;\n" "  add.u32 %0, %0, %1;\n"
               "  add.u32 %0, %0, %1;\n" "  add.u32 %0, %0, %1;\n"
               "  add.u32 %0, %0, %1;\n" "  add.u32 %0, %0, %1;\n"
               "  add.u32 %0, %0, %1;\n" "  add.u32 %0, %0, %1;\n"
               "  add.u32 %0, %0, %1;\n" "  add.u32 %0, %0, %1;\n"
               "  add.u32 %0, %0, %1;\n" "  add.u32 %0, %0, %1;\n"
               "  add.u32 %0, %0, %1;\n" "  add.u32 %0, %0, %1;\n"
               "  add.u32 %0, %0, %1;\n" "  add.u32 %0, %0, %1;\n"
               "  add.u32 %0, %0, %1;\n" "  add.u32 %0, %0, %1;\n"
               "  add.u32 %0, %0, %1;\n" "  add.u32 %0, %0, %1;\n"
               "  add.u32 %0, %0, %1;\n" "  add.u32 %0, %0, %1;\n"
               "  add.u32 %0, %0, %1;\n" "  add.u32 %0, %0, %1;\n"
               "  add.u32 %0, %0, %1;\n" "  add.u32 %0, %0, %1;\n"
               "  mad.lo.u32 %0, %0, 0x5bd1e995, %1;\n"
               "  mad.lo.u32 %0, %0, 0x5bd1e995, %1;\n"
               "  mad.lo.u32 %0, %0, 0x5bd1e995, %1;\n"
               "  mad.lo.u32 %0, %0, 0x5bd1e995, %1;\n"
               "  mad.lo.u32 %0, %0, 0x5bd1e995, %1;\n"
               "  mad.lo.u32 %0, %0, 0x5bd1e995, %1;\n"
               "  mad.lo.u32 %0, %0, 0x5bd1e995, %1;\n"
               "  mad.lo.u32 %0, %0, 0x5bd1e995, %1;\n"
               "  mad.lo.u32 %0, %0, 0x5bd1e995, %1;\n"
               "  mad.lo.u32 %0, %0, 0x5bd1e995, %1;\n"
               "  mad.lo.u32 %0, %0, 0x5bd1e995, %1;\n"
               "  mad.lo.u32 %0, %0, 0x5bd1e995, %1;\n"
               "  mad.lo.u32 %0, %0, 0x5bd1e995, %1;\n"
               "  mad.lo.u32 %0, %0, 0x5bd1e995, %1;\n"
               "  mad.lo.u32 %0, %0, 0x5bd1e995, %1;\n"
               "  mad.lo.u32 %0, %0, 0x5bd1e995, %1;\n"
               "  .reg .pred r1_dominant_pad_pred;\n"
               "  setp.ge.s32 r1_dominant_pad_pred, %2, 0;\n"
               "  @r1_dominant_pad_pred add.u32 %0, %0, %1;\n"
               "  @r1_dominant_pad_pred bra r1_dominant_pad_done;\n"
               "  trap;\n"
               "r1_dominant_pad_done:\n"
               "}" : "+r"(state) : "r"(seed), "r"(rounds));
}

// r1 and r2 both need one predicate + integer MAD per group for their branch
// padding. Keeping the MAD here (rather than 64 address-MADs) makes it small
// and exactly balanced between the two variants.
__device__ __forceinline__ void r2DominantBranchPadding(
    int rounds, std::uint32_t& state, std::uint32_t seed) {
  asm volatile("{\n"
               "  .reg .pred r2_dominant_pad_pred;\n"
               "  setp.ge.s32 r2_dominant_pad_pred, %1, 0;\n"
               "  @r2_dominant_pad_pred bra r2_dominant_pad_done;\n"
               "  trap;\n"
               "r2_dominant_pad_done:\n"
               "  mad.lo.u32 %0, %0, 0x5bd1e995, %2;\n"
               "}" : "+r"(state) : "r"(rounds), "r"(seed));
}

__device__ __forceinline__ void xorPass(
    std::uint32_t& x, std::uint32_t& y, std::uint32_t& z, std::uint32_t& w) {
  asm volatile("{\n" HBM_XOR_128 "}"
               : "+r"(x), "+r"(y), "+r"(z), "+r"(w));
}

__device__ __forceinline__ void r2BranchPadding(int rounds) {
  asm volatile("{\n"
               "  .reg .pred r2_branch_pad_pred;\n"
               "  setp.ge.s32 r2_branch_pad_pred, %0, 0;\n"
               "  @r2_branch_pad_pred bra r2_branch_pad_done;\n"
               "  trap;\n"
               "r2_branch_pad_done:\n"
               "}" :: "r"(rounds));
}

#undef HBM_LOAD_DOMINANT_STATIC
#undef HBM_XOR_4
#undef HBM_XOR_8
#undef HBM_XOR_16
#undef HBM_XOR_32
#undef HBM_XOR_64
#undef HBM_XOR_128

// This is the only retained HBM load kernel. It is specialized for the
// normal A100 grid defined in slots.cuh. r1/r2 have equal loop,
// pointer, predicate and MAD padding; r2 adds 64 LDG.E.128 and 32 LOP3 per
// thread/group, so LDG is the dominant varying SASS opcode.
__global__ void hbmLoadDominantKernel(const packed_fp32* __restrict__ data,
                                      size_t groups, int rounds, int innerRepeats) {
  const size_t initial = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  std::uint32_t state = 0x6d2b79f5u ^ threadIdx.x;
  const std::uint32_t seed = 0x9e3779b9u ^ blockIdx.x;

#pragma unroll 1
  for (int repeat = 0; repeat < innerRepeats; ++repeat) {
    const packed_fp32* cursor = data + initial;
#pragma unroll 1
    for (size_t group = 0; group < groups; ++group) {
      loadDominantPass(state, cursor);
      if (rounds > 1) {
        loadDominantPass(state, cursor);
        r2DominantBranchPadding(rounds, state, seed);
      } else {
        advanceDominantCursor(cursor);
        r1DominantStaticPadding(rounds, state, seed);
      }
    }
  }
  if (state == 0xffffffffu) asm volatile("trap;");
}

inline void launchLoadDominant(const packed_fp32* data, size_t groups, int blocks,
                               int rounds, int innerRepeats, cudaStream_t stream) {
  hbmLoadDominantKernel<<<blocks, kThreads, 0, stream>>>(
      data, groups, rounds, innerRepeats);
  CUDA_CHECK(cudaGetLastError());
}

__global__ void hbmXorOnlyKernel(size_t groups, int rounds, int innerRepeats) {
  const std::uint32_t lane = threadIdx.x;
  std::uint32_t x = lane ^ 0x12345678u;
  std::uint32_t y = lane ^ 0x9abcdef0u;
  std::uint32_t z = lane ^ 0x31415926u;
  std::uint32_t w = lane ^ 0x27182818u;

#pragma unroll 1
  for (int repeat = 0; repeat < innerRepeats; ++repeat) {
#pragma unroll 1
    for (size_t group = 0; group < groups; ++group) {
      xorPass(x, y, z, w);
      if (rounds > 1) {
        xorPass(x, y, z, w);
        r2BranchPadding(rounds);
      } else {
        r2BranchPadding(rounds);
      }
    }
  }
  if ((x & y & z & w) == 0xffffffffu) asm volatile("trap;");
}

inline void launchXorOnly(size_t groups, int blocks, int rounds,
                          int innerRepeats, cudaStream_t stream) {
  hbmXorOnlyKernel<<<blocks, kThreads, 0, stream>>>(groups, rounds, innerRepeats);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace hbmv2
