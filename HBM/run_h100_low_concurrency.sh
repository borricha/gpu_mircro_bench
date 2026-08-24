#!/usr/bin/env bash
# Reproduce the throttle-avoidance geometry on an H100 without changing its
# power limit or application clocks.  Defaults target the 114-SM PCIe H100.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATIC_DIR="${ROOT_DIR}/../static"

DEVICE="${DEVICE:-0}"
NVML_DEVICE="${NVML_DEVICE:-0}"
GRID_BLOCKS="${GRID_BLOCKS:-57}"
THREADS="${THREADS:-128}"
MIB="${MIB:-1024}"
REPETITIONS="${REPETITIONS:-2}"
MEASURE_SECONDS="${MEASURE_SECONDS:-60}"
IDLE_SECONDS="${IDLE_SECONDS:-60}"
STATIC_SECONDS="${STATIC_SECONDS:-180}"
PRECONDITION_SECONDS="${PRECONDITION_SECONDS:-20}"
COOLDOWN_SECONDS="${COOLDOWN_SECONDS:-30}"
SAMPLE_MS="${SAMPLE_MS:-10}"
RUN_TAG="${RUN_TAG:-low_concurrency_${GRID_BLOCKS}cta_${THREADS}t_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/result/H100/${RUN_TAG}}"

if [[ "${THREADS}" != 128 && "${THREADS}" != 256 ]]; then
  echo "THREADS must be 128 or 256" >&2
  exit 2
fi
if (( GRID_BLOCKS < 1 )); then
  echo "GRID_BLOCKS must be positive" >&2
  exit 2
fi

mkdir -p "${OUT_DIR}/ncu" "${OUT_DIR}/power"

export CUDA_ARCH="${CUDA_ARCH:-sm_90}"

echo "H100 low-concurrency geometry: ${GRID_BLOCKS} CTAs × ${THREADS} threads = $((GRID_BLOCKS * THREADS / 32)) target warps"
echo "Output: ${OUT_DIR}"

"${ROOT_DIR}/bench.sh" build
"${STATIC_DIR}/bench.sh" build

static_csv="${OUT_DIR}/static_summary.csv"
"${STATIC_DIR}/bench.sh" run \
  --device "${DEVICE}" --nvml-device "${NVML_DEVICE}" \
  --idle-seconds "${IDLE_SECONDS}" --measure-seconds "${STATIC_SECONDS}" \
  --precondition-seconds "${PRECONDITION_SECONDS}" --sample-ms "${SAMPLE_MS}" \
  --threads "${THREADS}" --grid-blocks "${GRID_BLOCKS}" --blocks-per-sm 1 \
  --idle-telemetry-output "${OUT_DIR}/static_idle_telemetry.csv" \
  --telemetry-output "${OUT_DIR}/static_nanosleep_telemetry.csv" \
  --output "${static_csv}"

IFS=, read -r p_const p_static < <(
  python3 - "${static_csv}" <<'PY'
import csv
import sys

with open(sys.argv[1], newline="") as source:
    rows = list(csv.DictReader(source))
if not rows:
    raise SystemExit("static calibration produced no CSV row")
row = rows[-1]
print(f'{row["p_constant_w"]},{row["static_power_w"]}')
PY
)
echo "Measured baselines: P_const=${p_const} W, P_static=${p_static} W"

for input in zero random; do
  for kind in load-xor xor-only; do
    for rounds in 1 2; do
      prefix="${OUT_DIR}/ncu/${input}_${kind}_r${rounds}"
      NCU_OUTPUT="${prefix}" "${ROOT_DIR}/bench.sh" ncu verify "${kind}" \
        --device "${DEVICE}" --mib "${MIB}" --grid-blocks "${GRID_BLOCKS}" \
        --threads "${THREADS}" --input "${input}" --rounds "${rounds}"
    done
  done
done

for input in zero random; do
  power_dir="${OUT_DIR}/power/differential_${input}"
  "${ROOT_DIR}/bench.sh" differential \
    --device "${DEVICE}" --mib "${MIB}" --grid-blocks "${GRID_BLOCKS}" \
    --threads "${THREADS}" --input "${input}" --repetitions "${REPETITIONS}" \
    --measure-seconds "${MEASURE_SECONDS}" --precondition-seconds "${PRECONDITION_SECONDS}" \
    --cooldown-seconds "${COOLDOWN_SECONDS}" --sample-ms "${SAMPLE_MS}" \
    --p-const-w "${p_const}" --p-static-w "${p_static}" --output-dir "${power_dir}"
done

python3 - "${OUT_DIR}" <<'PY'
import csv
import glob
import os
import sys

root = sys.argv[1]
failed = []
for path in sorted(glob.glob(os.path.join(root, "power", "differential_*", "raw.csv"))):
    with open(path, newline="") as source:
        for row in csv.DictReader(source):
            checks = (
                "throttle_any_pct",
                "throttle_sw_power_pct",
                "throttle_hw_slowdown_pct",
                "throttle_hw_thermal_pct",
            )
            active = {name: float(row[name]) for name in checks if float(row[name]) != 0.0}
            if active:
                failed.append((path, row["kind"], row["active_loads"], row["repetition"], active))

if failed:
    print("Throttle detected:")
    for path, kind, loads, rep, active in failed:
        print(f"  {os.path.relpath(path, root)} {kind} r{int(loads) // 64} rep{rep}: {active}")
    raise SystemExit(1)
print("PASS: all r1/r2, load/control, zero/random samples report zero throttle.")
PY
