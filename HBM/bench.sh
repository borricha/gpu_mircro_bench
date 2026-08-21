#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${ROOT_DIR}/build"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.1}"
NVCC_BIN="${NVCC:-${CUDA_HOME}/bin/nvcc}"
NCU_BIN="${NCU:-${CUDA_HOME}/bin/ncu}"
NSYS_BIN="${NSYS:-${CUDA_HOME}/bin/nsys}"

LOAD_BIN="${BUILD_DIR}/load_xor"
XOR_BIN="${BUILD_DIR}/xor_only"
POWER_BIN="${BUILD_DIR}/power_driver"

usage() {
  cat <<'EOF'
Usage:
  ./bench.sh build
  ./bench.sh single <load-xor|xor-only> [kernel options]
  ./bench.sh ncu <verify|full> <load-xor|xor-only> [kernel options]
  ./bench.sh nsys [single|power] <load-xor|xor-only> [options]
  ./bench.sh power [load-xor|xor-only] [power options]
  ./bench.sh differential [differential options]
  ./bench.sh instruction-mix [--ncu-dir DIR] [--out-dir DIR]

Kernel options include:
  --device N --mib N --blocks-per-sm N --active-loads <64|128>
  --input <zero|random|one-point-one> --trials N

active-loads=64 is r1 and 128 is r2. `load-xor` is the final LDG-dominant
variant: r2 adds 64 vector LDG + 32 LOP3 per thread/group; r1 padding
balances every other executed SASS opcode.
EOF
}

build() {
  mkdir -p "${BUILD_DIR}"
  [[ -x "${NVCC_BIN}" ]] || { echo "nvcc not found: ${NVCC_BIN}" >&2; exit 1; }
  "${NVCC_BIN}" -O3 -std=c++17 -arch=sm_80 -lineinfo \
    "${ROOT_DIR}/load_xor.cu" -o "${LOAD_BIN}"
  "${NVCC_BIN}" -O3 -std=c++17 -arch=sm_80 -lineinfo \
    "${ROOT_DIR}/xor_only.cu" -o "${XOR_BIN}"
}

build_power() {
  mkdir -p "${BUILD_DIR}"
  [[ -x "${NVCC_BIN}" ]] || { echo "nvcc not found: ${NVCC_BIN}" >&2; exit 1; }
  "${NVCC_BIN}" -O3 -std=c++17 -arch=sm_80 -lineinfo \
    "${ROOT_DIR}/power_driver.cu" -o "${POWER_BIN}" \
    -lnvidia-ml -Xcompiler -pthread
}

ensure_power_binary() {
  [[ -x "${POWER_BIN}" && "${POWER_BIN}" -nt "${ROOT_DIR}/power_driver.cu" &&
     "${POWER_BIN}" -nt "${ROOT_DIR}/load_xor.cu" &&
     "${POWER_BIN}" -nt "${ROOT_DIR}/xor_only.cu" &&
     "${POWER_BIN}" -nt "${ROOT_DIR}/common.cuh" ]] || build_power
}

binary_for() {
  case "$1" in
    load-xor) printf '%s\n' "${LOAD_BIN}" ;;
    xor-only) printf '%s\n' "${XOR_BIN}" ;;
    *) echo "unknown kernel kind: $1" >&2; exit 2 ;;
  esac
}

run_single() {
  local kind="$1"; shift
  ensure_binary "${kind}"
  "$(binary_for "${kind}")" "$@"
}

ensure_binary() {
  local kind="$1" source binary
  case "${kind}" in
    load-xor) source="${ROOT_DIR}/load_xor.cu"; binary="${LOAD_BIN}" ;;
    xor-only) source="${ROOT_DIR}/xor_only.cu"; binary="${XOR_BIN}" ;;
    *) echo "unknown kernel kind: ${kind}" >&2; exit 2 ;;
  esac
  [[ -x "${binary}" && "${binary}" -nt "${source}" && "${binary}" -nt "${ROOT_DIR}/common.cuh" ]] || build
}

run_ncu() {
  local mode="$1" kind="$2"; shift 2
  local binary kernel output
  ensure_binary "${kind}"
  binary="$(binary_for "${kind}")"
  case "${kind}" in
    load-xor) kernel='regex:hbmLoadDominantKernel' ;;
    xor-only) kernel='regex:hbmXorOnlyKernel' ;;
  esac
  output="${NCU_OUTPUT:-${BUILD_DIR}/ncu/${kind}_${mode}}"
  mkdir -p "$(dirname "${output}")"

  case "${mode}" in
    verify)
      "${NCU_BIN}" --target-processes all --clock-control none --replay-mode kernel \
        --kernel-name "${kernel}" --launch-count 1 \
        --metrics sm__sass_thread_inst_executed.sum,sm__sass_thread_inst_executed_op_memory_pred_on.sum,sm__sass_thread_inst_executed_op_bit_pred_on.sum,sm__sass_thread_inst_executed_op_integer_pred_on.sum,sm__sass_thread_inst_executed_op_control_pred_on.sum,sm__sass_inst_executed_op_branch.sum,l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,lts__t_sectors_op_read.sum,dram__bytes_read.sum \
        --csv --page raw --log-file "${output}.csv" \
        "${binary}" "$@"
      ;;
    full)
      "${NCU_BIN}" --target-processes all --clock-control none --replay-mode kernel \
        --kernel-name "${kernel}" --launch-count 1 --set full --force-overwrite \
        --export "${output}" "${binary}" "$@"
      ;;
    *) echo "ncu mode must be verify or full" >&2; exit 2 ;;
  esac
}


run_nsys() {
  local mode="${1:-single}" kind prefix tmp_dir
  local -a target=()
  case "${mode}" in
    single)
      shift
      kind="${1:-}"
      [[ "${kind}" == "load-xor" || "${kind}" == "xor-only" ]] || { usage >&2; exit 2; }
      shift
      ensure_binary "${kind}"
      target=("$(binary_for "${kind}")" "$@")
      ;;
    power)
      shift
      kind="${1:-}"
      [[ "${kind}" == "load-xor" || "${kind}" == "xor-only" ]] || {
        echo "nsys power requires load-xor or xor-only" >&2; exit 2; }
      shift
      ensure_power_binary
      target=("${POWER_BIN}" --kind "${kind}" "$@")
      ;;
    load-xor|xor-only)
      kind="${mode}"
      ensure_binary "${kind}"
      target=("$(binary_for "${kind}")" "${@:2}")
      ;;
    *) usage >&2; exit 2 ;;
  esac
  [[ -x "${NSYS_BIN}" ]] || { echo "nsys not found: ${NSYS_BIN}" >&2; exit 1; }
  prefix="${NSYS_OUTPUT:-${BUILD_DIR}/nsys/hbmv2_${mode}_${kind}}"
  tmp_dir="${TMPDIR:-${BUILD_DIR}/nsys/tmp}"
  mkdir -p "$(dirname "${prefix}")" "${tmp_dir}"
  TMPDIR="${tmp_dir}" "${NSYS_BIN}" profile \
    --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
    --cuda-memory-usage=true \
    --gpu-metrics-devices=cuda-visible --gpu-metrics-frequency=100 \
    --enable=nvml_metrics,-i10 \
    --force-overwrite true --output "${prefix}" \
    "${target[@]}"
}
run_power() {
  local kind=""
  case "${1:-}" in
    load-xor|xor-only) kind="$1"; shift ;;
  esac
  ensure_power_binary
  if [[ -n "${kind}" ]]; then
    "${POWER_BIN}" --kind "${kind}" "$@"
  else
    "${POWER_BIN}" "$@"
  fi
}

# Kept in this dispatcher so each benchmark has a single shell entry point.
run_differential() (
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="${ROOT_DIR}/bench.sh"

DEVICE="${DEVICE:-0}"
P_CONST_W="${P_CONST_W:-33.154}"
P_STATIC_W="${P_STATIC_W:-34.103}"
PRECONDITION_SECONDS="${PRECONDITION_SECONDS:-20}"
MEASURE_SECONDS="${MEASURE_SECONDS:-60}"
COOLDOWN_SECONDS="${COOLDOWN_SECONDS:-30}"
SAMPLE_MS="${SAMPLE_MS:-10}"
REPETITIONS="${REPETITIONS:-2}"
MIB="${MIB:-1024}"
BLOCKS_PER_SM="${BLOCKS_PER_SM:-8}"
LOAD_KIND="${LOAD_KIND:-load-xor}"
INNER_REPEATS="${INNER_REPEATS:-64}"
LAUNCH_BATCH="${LAUNCH_BATCH:-8}"
INPUT="${INPUT:-random}"
RUN_TAG="${RUN_TAG:-sass_slope_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/build/power/${RUN_TAG}}"

usage() {
  cat <<'EOF'
Usage: ./bench.sh differential [options]
  --device N --p-const-w W --p-static-w W --mib N --blocks-per-sm N
  --precondition-seconds S --measure-seconds S --cooldown-seconds S
  --sample-ms MS --repetitions N --inner-repeats N --launch-batch N
  --input random|zero|one-point-one --output-dir DIR

For each kernel kind, r1 is first calibrated to approximately the requested
duration.  The resulting graph-node count and inner-repeat count are then
held fixed for r1 and r2.  The summary derives LOP3 pJ/warp-instruction from
the XOR pair and HBM LDG.E.128 pJ/warp-instruction from `load-xor`. The
number of varying LOP3 per added LDG is 0.5, verified from the raw
instruction counts.
EOF
}

while (($#)); do
  case "$1" in
    --device) DEVICE="$2"; shift 2 ;;
    --p-const-w) P_CONST_W="$2"; shift 2 ;;
    --p-static-w) P_STATIC_W="$2"; shift 2 ;;
    --mib) MIB="$2"; shift 2 ;;
    --blocks-per-sm) BLOCKS_PER_SM="$2"; shift 2 ;;
    --load-kind) LOAD_KIND="$2"; shift 2 ;;
    --precondition-seconds) PRECONDITION_SECONDS="$2"; shift 2 ;;
    --measure-seconds) MEASURE_SECONDS="$2"; shift 2 ;;
    --cooldown-seconds) COOLDOWN_SECONDS="$2"; shift 2 ;;
    --sample-ms) SAMPLE_MS="$2"; shift 2 ;;
    --repetitions) REPETITIONS="$2"; shift 2 ;;
    --inner-repeats) INNER_REPEATS="$2"; shift 2 ;;
    --launch-batch) LAUNCH_BATCH="$2"; shift 2 ;;
    --input) INPUT="$2"; shift 2 ;;
    --output-dir) OUT_DIR="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ "${LOAD_KIND}" == "load-xor" ]] || {
  echo "HBM retains only --load-kind load-xor" >&2
  exit 2
}

mkdir -p "${OUT_DIR}/telemetry"
RAW_CSV="${OUT_DIR}/raw.csv"
SUMMARY_CSV="${OUT_DIR}/summary.csv"
rm -f "${RAW_CSV}" "${SUMMARY_CSV}"

calibrate_kind() {
  local kind="$1" calibration_raw calibration_telemetry calibration_output
  local active_s graph_nodes inner_repeats node_s nodes
  calibration_raw="${OUT_DIR}/calibration_${kind}.csv"
  calibration_telemetry="${OUT_DIR}/telemetry/calibration_${kind}.csv"
  calibration_output="$("${RUNNER}" power \
    --device "${DEVICE}" --kind "${kind}" --active-loads 64 \
    --mib "${MIB}" --blocks-per-sm "${BLOCKS_PER_SM}" --input "${INPUT}" \
    --precondition-seconds 0 --measure-seconds 1 --sample-ms "${SAMPLE_MS}" \
    --inner-repeats "${INNER_REPEATS}" --launch-batch "${LAUNCH_BATCH}" \
    --p-constant-w "${P_CONST_W}" --p-static-w "${P_STATIC_W}" --repetition 1 \
    --telemetry-output "${calibration_telemetry}" --output "${calibration_raw}")"
  echo "${calibration_output}"
  IFS=, read -r graph_nodes inner_repeats active_s < <(
    awk -F, 'NR == 2 {offset = ($3 == "executed_warp_sass") ? 1 : 0; print $(4 + offset) "," $(5 + offset) "," $(7 + offset)}' "${calibration_raw}")
  [[ -n "${graph_nodes}" && -n "${inner_repeats}" && -n "${active_s}" ]] || {
    echo "calibration parse failed for ${kind}" >&2; exit 1;
  }
  node_s="$(awk -v t="${active_s}" -v n="${graph_nodes}" 'BEGIN {printf "%.12f", t/n}')"
  nodes="$(awk -v seconds="${MEASURE_SECONDS}" -v node="${node_s}" \
    'BEGIN {n = int(seconds / node + 0.999999); if (n < 1) n = 1; print n}')"
  printf '%s,%s\n' "${nodes}" "${inner_repeats}"
}

run_condition() {
  local kind="$1" active="$2" rep="$3" graph_nodes="$4" inner_repeats="$5"
  local label="${kind}_r$((active / 64))_rep${rep}"
  echo "== ${kind} r$((active / 64)), repetition ${rep}/${REPETITIONS} =="
  "${RUNNER}" power \
    --device "${DEVICE}" --kind "${kind}" --active-loads "${active}" \
    --mib "${MIB}" --blocks-per-sm "${BLOCKS_PER_SM}" --input "${INPUT}" \
    --precondition-seconds "${PRECONDITION_SECONDS}" --measure-seconds "${MEASURE_SECONDS}" \
    --sample-ms "${SAMPLE_MS}" --inner-repeats "${inner_repeats}" \
    --launch-batch "${LAUNCH_BATCH}" --graph-nodes "${graph_nodes}" \
    --p-constant-w "${P_CONST_W}" \
    --p-static-w "${P_STATIC_W}" --repetition "${rep}" \
    --telemetry-output "${OUT_DIR}/telemetry/${label}.csv" --output "${RAW_CSV}"
  if awk -v seconds="${COOLDOWN_SECONDS}" 'BEGIN { exit !(seconds > 0) }'; then
    echo "   CUDA-idle cooldown ${COOLDOWN_SECONDS}s"
    sleep "${COOLDOWN_SECONDS}"
  fi
}

IFS=, read -r XOR_GRAPH_NODES XOR_INNER_REPEATS < <(calibrate_kind xor-only | tail -n 1)
IFS=, read -r LOAD_GRAPH_NODES LOAD_INNER_REPEATS < <(calibrate_kind "${LOAD_KIND}" | tail -n 1)
echo "Using fixed xor-only graph: ${XOR_GRAPH_NODES} nodes × ${XOR_INNER_REPEATS} inner repeats"
echo "Using fixed ${LOAD_KIND} graph: ${LOAD_GRAPH_NODES} nodes × ${LOAD_INNER_REPEATS} inner repeats"

for ((rep=1; rep<=REPETITIONS; ++rep)); do
  run_condition xor-only 64 "${rep}" "${XOR_GRAPH_NODES}" "${XOR_INNER_REPEATS}"
  run_condition xor-only 128 "${rep}" "${XOR_GRAPH_NODES}" "${XOR_INNER_REPEATS}"
  run_condition "${LOAD_KIND}" 64 "${rep}" "${LOAD_GRAPH_NODES}" "${LOAD_INNER_REPEATS}"
  run_condition "${LOAD_KIND}" 128 "${rep}" "${LOAD_GRAPH_NODES}" "${LOAD_INNER_REPEATS}"
done

summarize_slope() {
  python3 - "$@" <<'PY'
"""Attribute HBM-v2 SASS energy from four CUDA-Graph/NVML conditions."""
import csv
import math
import sys
from collections import defaultdict


def mean(values):
    return sum(values) / len(values)


def main(raw_path, output_path, load_kind, count_scale):
    rows = defaultdict(list)
    with open(raw_path, newline="") as source:
        for row in csv.DictReader(source):
            rows[(row["kind"], int(row["active_loads"]))].append(row)

    required = [("xor-only", 64), ("xor-only", 128),
                (load_kind, 64), (load_kind, 128)]
    missing = [f"{kind}/{active}" for kind, active in required if not rows[(kind, active)]]
    if missing:
        raise SystemExit("missing power conditions: " + ", ".join(missing))

    # r1/r2 use equal graph work counts.  Their duration differs naturally,
    # so subtract P_const/P_static in joules *per condition*, then difference
    # those energies.  This is the Wattchmen-style energy attribution, not a
    # power-versus-throughput fit (the saturated issue rate can be similar at
    # r1 and r2).
    point = {}
    for key in required:
        samples = rows[key]
        point[key] = {
            "board_w": mean([float(r["board_avg_w"]) for r in samples]),
            "board_energy": mean([float(r["board_energy_j"]) for r in samples]),
            "const_energy": mean([float(r["energy_after_const_j"]) for r in samples]),
            "const_static_energy": mean([float(r["energy_after_const_static_j"]) for r in samples]),
            "ldg_count": mean([float(r["variable_ldg_warp_instructions"]) * count_scale for r in samples]),
            "lop3_count": mean([float(r["variable_lop3_warp_instructions"]) * count_scale for r in samples]),
            "ldg_rate": mean([float(r["variable_ldg_warp_instr_per_s"]) * count_scale for r in samples]),
            "lop3_rate": mean([float(r["variable_lop3_warp_instr_per_s"]) * count_scale for r in samples]),
            "throttle": mean([float(r["throttle_any_pct"]) for r in samples]),
            "sm_clock": mean([float(r["sm_clock_avg_mhz"]) for r in samples]),
            "mem_clock": mean([float(r["mem_clock_avg_mhz"]) for r in samples]),
        }

    def delta(kind, field):
        return point[(kind, 128)][field] - point[(kind, 64)][field]

    delta_xor_lop = delta("xor-only", "lop3_count")
    delta_load_ldg = delta(load_kind, "ldg_count")
    delta_load_lop = delta(load_kind, "lop3_count")
    if min(abs(delta_xor_lop), abs(delta_load_ldg)) <= 0.0:
        raise SystemExit("zero instruction-count delta; inspect raw.csv")
    lop3_per_ldg = delta_load_lop / delta_load_ldg

    def attribute(energy_field):
        control = delta("xor-only", energy_field) / delta_xor_lop
        combined = delta(load_kind, energy_field) / delta_load_ldg
        ldg = combined - lop3_per_ldg * control
        return control, combined, ldg

    control_board, combined_board, ldg_board = attribute("board_energy")
    control_const, combined_const, ldg_const = attribute("const_energy")
    control_static, combined_static, ldg_static = attribute("const_static_energy")

    def pj_per_bit(energy_j):
        # LDG.E.128 is a 16 B per-lane vector load: one warp-level
        # instruction corresponds to 32 × 16 B = 4096 logical bits.
        return energy_j * 1.0e12 / 4096.0

    result = [
        ("method", "fixed graph work; r2-r1 energy difference", "", "P_const/P_static subtracted per actual duration"),
        ("sass_count_scale", count_scale, "scale", "applied to raw instruction counts before attribution"),
        ("xor_delta_lop3_count", delta_xor_lop, "warp LOP3", "xor-only r2-r1"),
        ("load_delta_ldg_count", delta_load_ldg, "warp LDG", f"{load_kind} r2-r1"),
        ("load_delta_lop3_count", delta_load_lop, "warp LOP3", f"{load_kind} r2-r1"),
        ("lop3_per_added_ldg", lop3_per_ldg, "warp LOP3/warp LDG",
         f"{load_kind} r2-r1; NCU-verified 0.5 for the retained load-xor kernel"),
        ("lop3_energy_board", control_board * 1.0e12, "pJ/warp LOP3", "raw board-energy delta"),
        ("hbm_ldg_e128_energy_board", ldg_board * 1.0e12, "pJ/warp LDG.E.128", "raw board-energy delta"),
        ("hbm_ldg_e128_energy_board", pj_per_bit(ldg_board), "pJ/logical bit", "512 B per warp LDG.E.128"),
        ("lop3_energy_after_const", control_const * 1.0e12, "pJ/warp LOP3", "r2-r1 after P_const"),
        ("hbm_ldg_e128_energy_after_const", ldg_const * 1.0e12, "pJ/warp LDG.E.128",
         f"load delta minus {lop3_per_ldg:g} × LOP3"),
        ("hbm_ldg_e128_energy_after_const", pj_per_bit(ldg_const), "pJ/logical bit", "512 B per warp LDG.E.128"),
        ("lop3_energy_after_const_static", control_static * 1.0e12, "pJ/warp LOP3", "r2-r1 after P_const + P_static"),
        ("hbm_ldg_e128_energy_after_const_static", ldg_static * 1.0e12, "pJ/warp LDG.E.128",
         f"load delta minus {lop3_per_ldg:g} × LOP3"),
        ("hbm_ldg_e128_energy_after_const_static", pj_per_bit(ldg_static), "pJ/logical bit", "512 B per warp LDG.E.128"),
    ]
    for kind, active in required:
        p = point[(kind, active)]
        prefix = f"{kind}_r{active // 64}"
        result.extend([
            (prefix + "_board_power", p["board_w"], "W", "mean across repetitions"),
            (prefix + "_throttle", p["throttle"], "%", "NVML samples"),
            (prefix + "_sm_clock", p["sm_clock"], "MHz", "NVML samples"),
            (prefix + "_mem_clock", p["mem_clock"], "MHz", "NVML samples"),
        ])

    with open(output_path, "w", newline="") as destination:
        writer = csv.writer(destination)
        writer.writerow(["metric", "value", "unit", "method"])
        for metric, value, unit, method in result:
            writer.writerow([metric, value, unit, method])
    for metric, value, unit, _ in result[:14]:
        print(f"{metric}: {value} {unit}".rstrip())


if __name__ == "__main__":
    if len(sys.argv) not in (3, 4, 5):
        raise SystemExit(f"Usage: {sys.argv[0]} RAW.csv SUMMARY.csv [load-xor] [count-scale]")
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) >= 4 else "load-xor",
         float(sys.argv[4]) if len(sys.argv) == 5 else 1.0)
PY
}

summarize_slope "${RAW_CSV}" "${SUMMARY_CSV}" "${LOAD_KIND}"
echo "Raw telemetry summary: ${RAW_CSV}"
echo "Attributed SASS energy: ${SUMMARY_CSV}"
)

# Kept in this dispatcher so each benchmark has a single shell entry point.
run_instruction_mix() (
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  echo "Usage: ./bench.sh instruction-mix [--ncu-dir DIR] [--out-dir DIR]"
  echo "Reads {xor,load}_r{1,2}_full.ncu-rep from --ncu-dir (default: build/ncu)."
  exit 0
fi
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NCU_DIR="${ROOT_DIR}/build/ncu"
OUT_DIR=""
while (($#)); do
  case "$1" in
    --ncu-dir) NCU_DIR="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    *) echo "unknown instruction-mix option: $1" >&2; exit 2 ;;
  esac
done
[[ -n "${OUT_DIR}" ]] || OUT_DIR="${NCU_DIR}/executed_instruction_mix"
if [[ -n "${NCU:-}" ]]; then
  NCU_BIN="${NCU}"
elif command -v ncu >/dev/null 2>&1; then
  NCU_BIN="$(command -v ncu)"
elif [[ -x /usr/local/cuda-13.1/bin/ncu ]]; then
  NCU_BIN=/usr/local/cuda-13.1/bin/ncu
else
  echo "ncu not found; set NCU=/path/to/ncu" >&2
  exit 1
fi
mkdir -p "${OUT_DIR}"

declare -a CASES=(xor_r1 xor_r2 load_r1 load_r2)
declare -A REPORTS=(
  [xor_r1]="${NCU_DIR}/xor_r1_full.ncu-rep"
  [xor_r2]="${NCU_DIR}/xor_r2_full.ncu-rep"
  [load_r1]="${NCU_DIR}/load_r1_full.ncu-rep"
  [load_r2]="${NCU_DIR}/load_r2_full.ncu-rep"
)

for c in "${CASES[@]}"; do
  [[ -f "${REPORTS[$c]}" ]] || { echo "missing report: ${REPORTS[$c]}" >&2; exit 1; }
  # This is the same per-SASS-line data displayed by NCU's Source > SASS page.
  "${NCU_BIN}" --import "${REPORTS[$c]}" --page source --print-source sass --csv \
    > "${OUT_DIR}/${c}_sass.csv"
done

python3 - "${OUT_DIR}" <<'PY'
import csv, os, re, sys

out = sys.argv[1]
cases = ("xor_r1", "xor_r2", "load_r1", "load_r2")
fields = {
    # Match the value shown in NCU GUI's Executed Instruction Mix panel.
    "executed_instruction_mix.csv": "Instructions Executed",
    # Keep the thread-expanded form separately; it is 32x the warp value for
    # fully active warps and must not be mixed with the GUI's count.
    "executed_instruction_mix_thread.csv": "Thread Instructions Executed",
    "executed_instruction_mix_pred_on.csv": "Predicated-On Thread Instructions Executed",
}

def num(value):
    try: return int(round(float(value.replace(',', ''))))
    except (ValueError, AttributeError): return 0

def opcode(sass):
    sass = sass.strip()
    sass = re.sub(r'^@!?P\d+\s+', '', sass)
    if not sass or sass.startswith('//'): return None
    token = sass.split()[0]
    # Keep an opcode family rather than a separate row for cache/address suffixes.
    return token.split('.')[0]

def parse(path, wanted):
    with open(path, newline='') as f:
        rows = list(csv.reader(f))
    header_index = next(i for i, row in enumerate(rows)
                        if row and row[0] == 'Address' and 'Source' in row)
    header = rows[header_index]
    source_i = header.index('Source')
    count_i = header.index(wanted)
    totals = {}
    for row in rows[header_index + 1:]:
        if len(row) <= max(source_i, count_i): continue
        op = opcode(row[source_i])
        if op: totals[op] = totals.get(op, 0) + num(row[count_i])
    return totals

for filename, wanted in fields.items():
    counts = {c: parse(os.path.join(out, c + '_sass.csv'), wanted) for c in cases}
    opcodes = sorted({op for d in counts.values() for op in d})
    with open(os.path.join(out, filename), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['opcode', *cases])
        for op in opcodes:
            writer.writerow([op, *(counts[c].get(op, 0) for c in cases)])

# ``Instructions Executed`` is the GUI's static-SASS issue count.  A
# predicated instruction is included even when its predicate is false, so it
# cannot express a runtime r1/r2 load-count difference.  For the differential
# experiment every participating warp is uniform; divide NCU's predicate-on
# thread count by 32 to make the actual issued work directly comparable with
# the GUI's warp-level unit.
active = {c: parse(os.path.join(out, c + '_sass.csv'),
                   'Predicated-On Thread Instructions Executed') for c in cases}
opcodes = sorted({op for d in active.values() for op in d})
with open(os.path.join(out, 'executed_instruction_mix_active_warp_equiv.csv'),
          'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['opcode', *cases])
    for op in opcodes:
        writer.writerow([op, *(active[c].get(op, 0) // 32 for c in cases)])
PY

echo "Wrote ${OUT_DIR}/executed_instruction_mix.csv (NCU GUI: Instructions Executed)"
echo "Wrote ${OUT_DIR}/executed_instruction_mix_thread.csv"
echo "Wrote ${OUT_DIR}/executed_instruction_mix_pred_on.csv"
echo "Wrote ${OUT_DIR}/executed_instruction_mix_active_warp_equiv.csv (actual predicate-on work / 32)"
)

command="${1:-}"
case "${command}" in
  build) build ;;
  single) shift; run_single "${1:?kernel kind required}" "${@:2}" ;;
  ncu) shift; run_ncu "${1:?ncu mode required}" "${2:?kernel kind required}" "${@:3}" ;;
  nsys) shift; run_nsys "$@" ;;
  power) shift; run_power "$@" ;;
  differential) shift; run_differential "$@" ;;
  instruction-mix) shift; run_instruction_mix "$@" ;;
  *) usage; exit 2 ;;
esac
