# Microbenchmarks

This directory is the cleaned, runnable home for the final microbenchmarks.
The original `custom_ubench/*` directories remain untouched as experimental
history. Each cache-level directory contains the two SASS-differential source
files used in the final methodology:

- `load_xor.cu`: r1/r2 vary the load path and its required XOR consumers.
- `xor_only.cu`: the matched XOR/control reference used to remove that term.

`bench.sh` owns builds, NCU, Nsight Systems, and regular single-kernel runs.
Each cache level has a small `power_driver.cu` because CUDA Graph replay,
CUDA events, and NVML sampling must execute in the same CUDA process as the
kernel. Generated binaries, reports, and telemetry go under a local `build/`
directory and are not tracked.

```text
micro_bench/
├── L1/      # L1-resident LDG.E.32 and XOR differential benchmark
├── L2/      # L2-resident LDG.E.32 and XOR differential benchmark
├── HBM/     # HBM-resident LDG.E.128 / LDG-dominant differential benchmark
└── static/  # Wattchmen-style nanosleep P_static calibration
```

Run commands from an individual benchmark directory so its scripts place
outputs in that benchmark's local `build/` directory.
