# L2 cache v2 — v1-style differential LDG benchmark

`load_xor.cu` preserves the original v1 L2 benchmark path:

- ordinary global load (no `.cg` / `__ldcg`),
- `200,000` CTAs × `1024` threads,
- v1's `blockIdx.x % 33` region mapping,
- `16,896 KiB` FP32 working set, inside A100's 40 MiB L2.

`--active-loads 64` is r1 and `--active-loads 128` is r2.  r2 has the same
128 LDG address sequence as v1 `loadOnlyKernel`. The benchmark executes 15
warm-up launches outside the timed CUDA event, matching v1's repeated-run
warm-cache condition.

`xor_only.cu` is the differential control. It has no global-memory load,
but preserves the same launch geometry and r1→r2 address-IMAD/LOP3 increment.
Subtracting its energy slope from `load-xor` attributes the remaining slope to
`LDG.E.32`.

## NCU instruction-mix verification

```bash
CUDA_VISIBLE_DEVICES=0 NCU_OUTPUT=build/ncu/l2v2_r1_full \
./bench.sh ncu full load-xor --device 0 --active-loads 64 --input random

CUDA_VISIBLE_DEVICES=0 NCU_OUTPUT=build/ncu/l2v2_r2_full \
./bench.sh ncu full load-xor --device 0 --active-loads 128 --input random

./bench.sh instruction-mix \
  build/ncu/l2v2_r1_full.ncu-rep build/ncu/l2v2_r2_full.ncu-rep
```

## Power attribution

```bash
CUDA_VISIBLE_DEVICES=0 ./bench.sh differential \
  --device 0 \
  --p-const-w 33 --p-static-w 34 \
  --input random --repetitions 2
```

The CSV summary reports both `(P_board - P_const)` and
`(P_board - P_const - P_static)` LDG pJ/bit results.

## Nsight Systems

```bash
CUDA_VISIBLE_DEVICES=0 NSYS_OUTPUT=build/nsys/l2_load_r1 \
  ./bench.sh nsys power load-xor \
  --device 0 --active-loads 64 --input random \
  --precondition-seconds 20 --graph-nodes 5000 --launch-batch 32 \
  --p-constant-w 33.154 --p-static-w 34.103
```
