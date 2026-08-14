# L1 v2 differential LDG energy benchmark

Two fixed-round kernels retain L1 v1's 512 threads/block, one block/SM,
128 KiB FP32 footprint, `ld.global.ca.u32`, and 61,037 outer iterations.
The CUDA files contain only initialization, one kernel launch, and CUDA-event
timing. NCU, Nsight Systems, and power telemetry are dispatched by `bench.sh`.

| Kernel | One added pass |
| --- | --- |
| `load-xor` | 64 `LDG.E.STRONG.SM` and 32 `LOP3.LUT` |
| `xor-only` | 32 `LOP3.LUT`, with no global/L1 data load |

`--active-loads 64|128|192|256` selects one through four passes.  The pass
loop always runs four times, so both binaries have the same round-dependent
uniform branch behavior. NCU confirms that this control term is identical in
the two binaries at every point; subtracting their slopes removes it.

For fitted energy slopes in J/pass:

```text
e_LDG = (slope(load-xor) - slope(xor-only)) / 64
```

Here `slope(xor-only)` is the matched **XOR + pass-control** slope, not a
claim of pure LOP3 energy in isolation. It is exactly the term to subtract
from load-xor for LDG attribution. Before fitting, use
`E_dynamic(r) = E_total(r) - P_baseline * T(r)` with a matched baseline (or
jointly model that term); never subtract watts directly from a J/pass slope.

## Run

```bash
cd /scale/cal/home/kupsy/LLMInfra/micro_bench/L1
CUDA_VISIBLE_DEVICES=0 ./bench.sh build
CUDA_VISIBLE_DEVICES=0 ./bench.sh single load-xor --device 0 --active-loads 64 --input random
CUDA_VISIBLE_DEVICES=0 ./bench.sh single xor-only --device 0 --active-loads 64 --input random
```

## NCU verification

```bash
CUDA_VISIBLE_DEVICES=0 NCU_OUTPUT=build/ncu/load_xor_r1 ./bench.sh ncu verify load-xor --device 0 --active-loads 64 --input random
CUDA_VISIBLE_DEVICES=0 NCU_OUTPUT=build/ncu/load_xor_r2 ./bench.sh ncu verify load-xor --device 0 --active-loads 128 --input random
CUDA_VISIBLE_DEVICES=0 NCU_OUTPUT=build/ncu/xor_only_r1 ./bench.sh ncu verify xor-only --device 0 --active-loads 64 --input random
CUDA_VISIBLE_DEVICES=0 NCU_OUTPUT=build/ncu/xor_only_r2 ./bench.sh ncu verify xor-only --device 0 --active-loads 128 --input random
```

Make a CSV and paper-ready Markdown table from those four reports:

```bash
./bench.sh instruction-mix \
  build/ncu/load_xor_r1.ncu-rep \
  build/ncu/load_xor_r2.ncu-rep \
  build/ncu/xor_only_r1.ncu-rep \
  build/ncu/xor_only_r2.ncu-rep
```

Expected r1 -> r2: load-xor has 2x global-load and memory instruction counts,
and +32 LOP3/thread/iteration; xor-only has no global loads and the same
+32 LOP3 increment. The uniform-branch control count changes with r, but its
value is exactly equal between load-xor and xor-only for each r. Static SASS
has one compact body: load-xor `(LDG, LOP3)=(64,32)`, xor-only `(0,32)`.
The table labels dynamic LDG/LOP3 counts as **expected SASS counts**: NCU
does not expose an opcode-specific LOP3 counter, but it verifies the matching
integer-instruction delta directly. Tiny L2/DRAM values can remain from
non-LDG activity such as instruction/parameter path traffic; the decisive
checks are `l1tex__...global_op_ld = 0` and memory instructions = 0 for
`xor-only`, versus 100% L1 hit for `load-xor`.

## In-process CUDA-Graph/NVML power trial

```bash
CUDA_VISIBLE_DEVICES=0 ./bench.sh power load-xor \
  --device 0 --active-loads 64 --iterations 61037 --input random \
  --precondition-seconds 20 --graph-nodes 1322 --launch-batch 32 --sample-ms 10 \
  --p-constant-w 33.154 --p-static-w 34.103 \
  --telemetry-output build/power/telemetry/l1_load_r1.csv \
  --output build/power/l1_power.csv
```

`power_driver.cu` captures the graph, preconditions it, measures graph replay
with CUDA events, and samples the corresponding physical GPU through NVML in
the same process. The `differential` command runs matched XOR-only and
load-XOR r1/r2 pairs, then writes attributed LOP3 and LDG energy to CSV.

## Nsight Systems

```bash
CUDA_VISIBLE_DEVICES=0 NSYS_OUTPUT=build/nsys/l1_load_r1 \
  ./bench.sh nsys power load-xor \
  --device 0 --active-loads 64 --iterations 61037 --input random \
  --precondition-seconds 20 --graph-nodes 1322 --launch-batch 32 \
  --sample-ms 10 --p-constant-w 33.154 --p-static-w 34.103
```
