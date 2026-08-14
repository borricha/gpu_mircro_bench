# HBM v2 differential LDG benchmark

`load_xor.cu` is the final HBM load kernel. It uses explicit
`ld.global.cg.v4.u32` instructions, so the data path is `LDG.E.128` and
bypasses L1. The 1 GiB footprint exceeds A100's L2 capacity. Its r1/r2 pair
uses the fixed 108 SM × 8 CTA/SM geometry: r2 adds 64 vector loads and 32
`LOP3` per thread/group, while r1 padding balances the remaining executed
SASS opcodes.

`xor_only.cu` is the corresponding control pair. r2 adds 128 `LOP3` per
thread/group; the energy slope of that pair is scaled by the NCU-verified
0.5 varying `LOP3` per extra HBM LDG before attributing the remaining
`load-xor` slope to `LDG.E.128`.

## Run

```bash
cd /scale/cal/home/kupsy/LLMInfra/micro_bench/HBM
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
