# HBM differential LDG microbenchmark

This benchmark attributes the energy of an HBM-served load by comparing two
otherwise balanced SASS instruction streams. It targets an A100-40 PCIe GPU
(108 SMs) and uses a 1 GiB FP32 allocation, well above its 40 MiB L2
capacity.

## Load path and launch geometry

`load_xor.cu` launches the core `hbmLoadDominantKernel`. It uses explicit
`ld.global.u32` inline PTX, which explicitly issues a normal cacheable
`LDG.E.32` load. This avoids ptxas scalarizing a partially-consumed vector
load. The 1 GiB stream is much larger than L1/L2; each lane loads one FP32
word (4 B), so one warp-level LDG.E.32 represents 32 lanes × 4 B = **128 B**
of logical data.

- Grid: 108 SM × 8 CTA/SM = **864 CTAs**.
- CTA: 256 threads = **8 warps**.
- Input: deterministic random `uint4` bit patterns by default. Use random for
  HBM attribution because all-zero data can change memory-compression behavior.
- The buffer is split into complete stream groups of
  `864 CTAs × 256 threads × 128 scalar-load slots`. For a 1 GiB allocation this is
  two complete groups; a partial tail is intentionally unused.

The compile-time inline-PTX body removes per-load index arithmetic. One pass
issues exactly **64 LDG.E.32 + 32 LOP3 per thread**. R1 uses one pass
(`--active-loads 64`); R2 uses two passes (`--active-loads 128`). The R1
no-load slot executes matched cursor, predicate, branch, and integer padding
so NCU can isolate the intended R2−R1 increment:

```text
R2 − R1 = +64 warp LDG.E.32 +32 warp LOP3 per warp/group
```

`xor_only.cu` launches `hbmXorOnlyKernel`, the LOP3-only control. It retains
the same grid, group loop, branch structure, and allocation footprint but
does not issue a data load. Its R2−R1 increment is exactly **+32 LOP3 per
warp/group**.

## Documentation formulas

For one `load-xor` kernel launch, where `G` is the number of complete stream
groups and `I` is `--inner-repeats` (one for `single`):

```text
warps = 864 CTAs × (256 threads / 32) = 6,912 warps

logical bytes(R1) = G × I × 6,912 warps × 64 LDG × 128 B
logical bytes(R2) = G × I × 6,912 warps × 128 LDG × 128 B
```

For `--mib 1024`, `G = 2`; therefore one R1 launch issues 113,246,208 B
(0.10546875 GiB) and one R2 launch issues 226,492,416 B (0.2109375 GiB).

Let `ΔE_xor = E(xor R2) − E(xor R1)` and
`ΔE_load = E(load R2) − E(load R1)`, measured with equal graph work. The
per-warp-instruction attribution is:

```text
e_LOP3 = ΔE_xor / ΔN_LOP3
e_LDG  = (ΔE_load − 0.5 × ΔN_LDG × e_LOP3) / ΔN_LDG
pJ/bit = e_LDG / 1,024 bits
```

The `0.5` term comes from one LOP3 per two varying LDG.E.32 instructions.

## Run

```bash
cd /scale/cal/home/kupsy/gpu_micro_bench/HBM
CUDA_VISIBLE_DEVICES=0 ./bench.sh build
CUDA_VISIBLE_DEVICES=0 ./bench.sh single load-xor --device 0 --mib 1024 \
  --blocks-per-sm 8 --active-loads 64 --input random
CUDA_VISIBLE_DEVICES=0 ./bench.sh single xor-only --device 0 --mib 1024 \
  --blocks-per-sm 8 --active-loads 64 --input random
```

## NCU instruction-mix verification

```bash
CUDA_VISIBLE_DEVICES=0 NCU_OUTPUT=build/ncu/load_r1_full \
  ./bench.sh ncu full load-xor --device 0 --mib 1024 --blocks-per-sm 8 \
  --active-loads 64 --input random
CUDA_VISIBLE_DEVICES=0 NCU_OUTPUT=build/ncu/load_r2_full \
  ./bench.sh ncu full load-xor --device 0 --mib 1024 --blocks-per-sm 8 \
  --active-loads 128 --input random
CUDA_VISIBLE_DEVICES=0 NCU_OUTPUT=build/ncu/xor_r1_full \
  ./bench.sh ncu full xor-only --device 0 --mib 1024 --blocks-per-sm 8 \
  --active-loads 64 --input random
CUDA_VISIBLE_DEVICES=0 NCU_OUTPUT=build/ncu/xor_r2_full \
  ./bench.sh ncu full xor-only --device 0 --mib 1024 --blocks-per-sm 8 \
  --active-loads 128 --input random

./bench.sh instruction-mix
```

## Power attribution

```bash
CUDA_VISIBLE_DEVICES=0 ./bench.sh differential \
  --device 0 --mib 1024 --blocks-per-sm 8 \
  --p-const-w 33.154 --p-static-w 34.103 \
  --input random --repetitions 2
```

The summary reports both `P_board - P_const` and
`P_board - P_const - P_static` LDG energy per logical bit.

## Nsight Systems

```bash
CUDA_VISIBLE_DEVICES=0 NSYS_OUTPUT=build/nsys/hbm_load_r1 \
  ./bench.sh nsys power load-xor \
  --device 0 --mib 1024 --blocks-per-sm 8 --active-loads 64 --input random \
  --precondition-seconds 20 --measure-seconds 60 --sample-ms 10 \
  --p-constant-w 33.154 --p-static-w 34.103
```
