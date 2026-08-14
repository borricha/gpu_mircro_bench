# NanoSleep static-power calibration

This benchmark implements the `NANOSLEEP` method used by Wattchmen to isolate
GPU shared-resource/static power. It first obtains `P_const` before creating a
CUDA context (unless `--p-constant-w` is supplied). It then launches enough
blocks to cover every SM; every resident thread repeatedly executes only
inline-PTX `nanosleep.u32`, with no global-memory load/store, FP instruction,
or Tensor Core instruction.

The reported estimate is:

```text
P_static = P_board(nanosleep) - P_const
```

`nanosleep` is approximate, so `nanosleep_kernel_active_s` in the summary is
the CUDA-event duration actually measured, rather than the requested duration.
The default configuration is 256 threads and eight blocks per SM, i.e. up to
2048 resident threads (64 warps) per A100 SM. The summary records the actual
occupancy limit returned by CUDA.

Build and run on physical GPU 1:

```bash
CUDA_VISIBLE_DEVICES=1 ./bench.sh build

CUDA_VISIBLE_DEVICES=1 ./bench.sh run \
  --device 0 \
  --nvml-device 1 \
  --idle-seconds 60 \
  --precondition-seconds 20 \
  --measure-seconds 180 \
  --sample-ms 10 \
  --sleep-ns 1000000 \
  --threads 256 \
  --blocks-per-sm 8 \
  --repetition 1 \
  --idle-telemetry-output build/power/telemetry/static_idle_rep1.csv \
  --telemetry-output build/power/telemetry/static_nanosleep_rep1.csv \
  --output build/power/static_summary.csv
```

`--nvml-device` is the physical `nvidia-smi -i` index. It is deliberately
separate from logical CUDA `--device`: with `CUDA_VISIBLE_DEVICES=1`, CUDA
device 0 maps to physical NVML device 1.

To use the already chosen 46 W constant-power baseline instead of measuring
one in-process, add `--p-constant-w 46` (then `--nvml-device` is unnecessary).
For Wattchmen-style robustness, run
multiple repetitions and let the GPU cool between runs; the paper uses long
steady-state runs and takes a median across trials.

The `ncu` mode is intended only for a short instruction check, for example:

```bash
CUDA_VISIBLE_DEVICES=1 NCU_OUTPUT=build/ncu/static_short \
  ./bench.sh ncu --device 0 --measure-seconds 1 \
  --p-constant-w 46 --threads 256 --blocks-per-sm 8
```

For a 60-second Nsight Systems trace with its NVML metrics plugin enabled:

```bash
CUDA_VISIBLE_DEVICES=1 NSYS_OUTPUT=build/nsys/static_rep1_60s \
  ./bench.sh nsys --device 0 --p-constant-w 46 \
  --precondition-seconds 20 --measure-seconds 60 --sample-ms 10 \
  --threads 256 --blocks-per-sm 8 \
  --telemetry-output build/power/telemetry/static_rep1_nsys.csv \
  --output build/power/static_summary_nsys.csv
```
