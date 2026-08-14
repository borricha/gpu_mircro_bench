#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${ROOT_DIR}/build"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.1}"
NVCC="${NVCC:-${CUDA_HOME}/bin/nvcc}"
NCU="${NCU:-${CUDA_HOME}/bin/ncu}"
NSYS="${NSYS:-nsys}"
CUDA_ARCH="${CUDA_ARCH:-sm_80}"
COMMON="${ROOT_DIR}/common.cuh"
LOAD_SOURCE="${ROOT_DIR}/load_xor.cu"
XOR_SOURCE="${ROOT_DIR}/xor_only.cu"
LOAD_BIN="${BUILD_DIR}/load_xor"
XOR_BIN="${BUILD_DIR}/xor_only"
POWER_SOURCE="${ROOT_DIR}/power_driver.cu"
POWER_BIN="${BUILD_DIR}/power_driver"

usage() {
  cat <<'EOF'
Usage:
  ./bench.sh build
  ./bench.sh single load-xor|xor-only [kernel options]
  ./bench.sh ncu [verify|table|basic|detailed|full] load-xor|xor-only [kernel options]
  ./bench.sh nsys [single|power] load-xor|xor-only [options]
  ./bench.sh power load-xor|xor-only [power options]
  ./bench.sh differential [differential options]
  ./bench.sh instruction-mix REPORT.ncu-rep ...

Kernel options:
  --device N --active-loads 64|128|192|256 (or --rounds 1..4)
  --iterations N --input random|zero|one-point-one
  --launch-mode single|graph --graph-nodes N --graph-replays N

Power options are handled by the in-process CUDA-Graph/NVML driver:
  --iterations N --precondition-seconds S --sample-ms MS --graph-nodes N --launch-batch N
  --p-constant-w W --p-static-w W --input PATTERN --telemetry-output FILE --output FILE
EOF
}

binary_for() {
  case "$1" in
    load-xor) printf '%s\n' "${LOAD_BIN}" ;;
    xor-only) printf '%s\n' "${XOR_BIN}" ;;
    *) echo "invalid kernel kind: $1 (expected load-xor or xor-only)" >&2; exit 2 ;;
  esac
}

source_for() {
  case "$1" in
    load-xor) printf '%s\n' "${LOAD_SOURCE}" ;;
    xor-only) printf '%s\n' "${XOR_SOURCE}" ;;
    *) echo "invalid kernel kind: $1" >&2; exit 2 ;;
  esac
}

kernel_regex_for() {
  case "$1" in
    load-xor) printf '%s\n' 'regex:l1LoadXorKernel' ;;
    xor-only) printf '%s\n' 'regex:l1XorOnlyKernel' ;;
  esac
}

build_one() {
  local kind="$1" source binary
  source="$(source_for "${kind}")"
  binary="$(binary_for "${kind}")"
  "${NVCC}" -std=c++17 -O3 -lineinfo -arch="${CUDA_ARCH}" "${source}" -o "${binary}"
}

build() {
  mkdir -p "${BUILD_DIR}"
  [[ -x "${NVCC}" ]] || { echo "nvcc not found: ${NVCC}" >&2; exit 1; }
  build_one load-xor
  build_one xor-only
}

build_power() {
  mkdir -p "${BUILD_DIR}"
  [[ -x "${NVCC}" ]] || { echo "nvcc not found: ${NVCC}" >&2; exit 1; }
  "${NVCC}" -std=c++17 -O3 -lineinfo -arch="${CUDA_ARCH}" "${POWER_SOURCE}" \
    -o "${POWER_BIN}" -lnvidia-ml -Xcompiler -pthread
}

ensure_binary() {
  local kind="$1" binary source
  binary="$(binary_for "${kind}")"
  source="$(source_for "${kind}")"
  [[ -x "${binary}" && "${binary}" -nt "${source}" && "${binary}" -nt "${COMMON}" ]] || build
}

run_single() {
  local kind="$1"
  shift
  ensure_binary "${kind}"
  "$(binary_for "${kind}")" "$@"
}

run_ncu() {
  local profile="${1:-verify}"
  case "${profile}" in verify|table|basic|detailed|full) shift ;; *) profile=verify ;; esac
  local kind="${1:-}"
  [[ -n "${kind}" ]] || { usage >&2; exit 2; }
  shift
  ensure_binary "${kind}"
  [[ -x "${NCU}" ]] || { echo "ncu not found: ${NCU}" >&2; exit 1; }
  local out_dir="${BUILD_DIR}/ncu"
  local prefix="${NCU_OUTPUT:-${out_dir}/l1v2_${kind}_${profile}}"
  mkdir -p "$(dirname "${prefix}")"
  local args=(--target-processes all --clock-control none --replay-mode kernel
    --kernel-name "$(kernel_regex_for "${kind}")" --launch-count 1
    --export "${prefix}" --force-overwrite)
  if [[ "${profile}" == verify ]]; then
    args+=(--metrics 'sm__sass_thread_inst_executed,sm__sass_thread_inst_executed_op_integer_pred_on,sm__sass_thread_inst_executed_op_control_pred_on,sm__sass_thread_inst_executed_op_memory_pred_on,l1tex__t_requests_pipe_lsu_mem_global_op_ld,l1tex__t_sector_hit_rate,lts__t_sectors_op_read,dram__bytes_read,dram__bytes_write')
  elif [[ "${profile}" == table ]]; then
    # Paper-table counters. All instruction counters are thread instructions,
    # except smsp__inst_executed, which is explicitly a warp-instruction count.
    args+=(--metrics 'sm__sass_thread_inst_executed,smsp__inst_executed,smsp__inst_executed_op_branch,sm__sass_thread_inst_executed_op_control_pred_on,sm__sass_thread_inst_executed_op_integer_pred_on,sm__sass_thread_inst_executed_op_hfma_pred_on,l1tex__m_xbar2l1tex_read_bytes,lts__t_sectors_op_read,dram__bytes_read,dram__bytes_write,l1tex__t_requests_pipe_lsu_mem_global_op_ld,l1tex__t_sector_hit_rate')
  else
    args+=(--set "${profile}")
  fi
  "${NCU}" "${args[@]}" "$(binary_for "${kind}")" "$@"
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
      target=("${POWER_BIN}" --power-trial --kind "${kind}" "$@")
      ;;
    load-xor|xor-only)
      kind="${mode}"
      ensure_binary "${kind}"
      target=("$(binary_for "${kind}")" "${@:2}")
      ;;
    *) usage >&2; exit 2 ;;
  esac
  command -v "${NSYS}" >/dev/null || { echo "nsys not found: ${NSYS}" >&2; exit 1; }
  prefix="${NSYS_OUTPUT:-${BUILD_DIR}/nsys/l1v2_${mode}_${kind}}"
  tmp_dir="${TMPDIR:-${BUILD_DIR}/nsys/tmp}"
  mkdir -p "$(dirname "${prefix}")" "${tmp_dir}"
  TMPDIR="${tmp_dir}" "${NSYS}" profile \
    --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
    --cuda-memory-usage=true \
    --gpu-metrics-devices=cuda-visible --gpu-metrics-frequency=100 \
    --enable=nvml_metrics,-i10 \
    --force-overwrite true --output "${prefix}" \
    "${target[@]}"
}

ensure_power_binary() {
  [[ -x "${POWER_BIN}" && "${POWER_BIN}" -nt "${POWER_SOURCE}" &&
     "${POWER_BIN}" -nt "${COMMON}" ]] || build_power
}

run_power() {
  local kind="${1:-}"
  [[ "${kind}" == "load-xor" || "${kind}" == "xor-only" ]] || {
    echo "power requires load-xor or xor-only" >&2; exit 2; }
  shift
  ensure_power_binary
  "${POWER_BIN}" --power-trial --kind "${kind}" "$@"
}

run_differential() (
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="${ROOT_DIR}/bench.sh"
BIN="${ROOT_DIR}/build/power_driver"

DEVICE="${DEVICE:-0}"
P_CONST_W="${P_CONST_W:-33.154}"
P_STATIC_W="${P_STATIC_W:-34.103}"
PRECONDITION_SECONDS="${PRECONDITION_SECONDS:-20}"
COOLDOWN_SECONDS="${COOLDOWN_SECONDS:-30}"
SAMPLE_MS="${SAMPLE_MS:-10}"
REPETITIONS="${REPETITIONS:-2}"
ITERATIONS="${ITERATIONS:-61037}"
GRAPH_NODES="${GRAPH_NODES:-1322}"
LAUNCH_BATCH="${LAUNCH_BATCH:-32}"
INPUT="${INPUT:-random}"
RUN_TAG="${RUN_TAG:-differential_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/build/power/${RUN_TAG}}"

usage() {
  cat <<'EOF'
Usage: ./bench.sh differential [options]
  --device N --p-const-w W --p-static-w W
  --precondition-seconds S --cooldown-seconds S --sample-ms MS --repetitions N
  --iterations N --graph-nodes N --launch-batch N
  --input random|zero|one-point-one --output-dir DIR
EOF
}

while (($#)); do
  case "$1" in
    --device) DEVICE="$2"; shift 2 ;;
    --p-const-w) P_CONST_W="$2"; shift 2 ;;
    --p-static-w) P_STATIC_W="$2"; shift 2 ;;
    --iterations) ITERATIONS="$2"; shift 2 ;;
    --precondition-seconds) PRECONDITION_SECONDS="$2"; shift 2 ;;
    --cooldown-seconds) COOLDOWN_SECONDS="$2"; shift 2 ;;
    --sample-ms) SAMPLE_MS="$2"; shift 2 ;;
    --repetitions) REPETITIONS="$2"; shift 2 ;;
    --graph-nodes) GRAPH_NODES="$2"; shift 2 ;;
    --launch-batch) LAUNCH_BATCH="$2"; shift 2 ;;
    --input) INPUT="$2"; shift 2 ;;
    --output-dir) OUT_DIR="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

build_power
mkdir -p "${OUT_DIR}/telemetry"
RAW_CSV="${OUT_DIR}/raw.csv"
SUMMARY_CSV="${OUT_DIR}/summary.csv"
rm -f "${RAW_CSV}" "${SUMMARY_CSV}"

run_condition() {
  local kind="$1" active="$2" rep="$3"
  local telemetry="${OUT_DIR}/telemetry/${kind}_r$((active / 64))_rep${rep}.csv"
  echo "== ${kind} r$((active / 64)), repetition ${rep}/${REPETITIONS}: in-process graph =="
  "${BIN}" --power-trial --device "${DEVICE}" --kind "${kind}" --active-loads "${active}" \
    --precondition-seconds "${PRECONDITION_SECONDS}" --sample-ms "${SAMPLE_MS}" \
    --p-constant-w "${P_CONST_W}" --p-static-w "${P_STATIC_W}" \
    --iterations "${ITERATIONS}" --input "${INPUT}" --launch-batch "${LAUNCH_BATCH}" \
    --graph-nodes "${GRAPH_NODES}" --telemetry-output "${telemetry}" --output "${RAW_CSV}"
  if awk -v s="${COOLDOWN_SECONDS}" 'BEGIN {exit !(s > 0)}'; then
    echo "   CUDA-idle cooldown ${COOLDOWN_SECONDS}s"
    sleep "${COOLDOWN_SECONDS}"
  fi
}

for ((rep=1; rep<=REPETITIONS; ++rep)); do
  run_condition xor-only 64 "${rep}"
  run_condition xor-only 128 "${rep}"
done
for ((rep=1; rep<=REPETITIONS; ++rep)); do
  run_condition load-xor 64 "${rep}"
  run_condition load-xor 128 "${rep}"
done

mean() {
  awk -F, -v kind="$1" -v active="$2" -v col="$3" \
    '$1==kind && $2==active {sum+=$col; n++} END {if (!n) exit 1; printf "%.12f", sum/n}' "${RAW_CSV}"
}
x_lop1="$(mean xor-only 64 7)"; x_lop2="$(mean xor-only 128 7)"
l_ldg1="$(mean load-xor 64 6)"; l_ldg2="$(mean load-xor 128 6)"
l_lop1="$(mean load-xor 64 7)"; l_lop2="$(mean load-xor 128 7)"
x_ec1="$(mean xor-only 64 11)"; x_ec2="$(mean xor-only 128 11)"
l_ec1="$(mean load-xor 64 11)"; l_ec2="$(mean load-xor 128 11)"
x_es1="$(mean xor-only 64 12)"; x_es2="$(mean xor-only 128 12)"
l_es1="$(mean load-xor 64 12)"; l_es2="$(mean load-xor 128 12)"

attribute() {
  awk -v xe1="$1" -v xe2="$2" -v le1="$3" -v le2="$4" \
      -v xl1="${x_lop1}" -v xl2="${x_lop2}" -v ll1="${l_lop1}" -v ll2="${l_lop2}" \
      -v ld1="${l_ldg1}" -v ld2="${l_ldg2}" \
    'BEGIN {ec=(xe2-xe1)/(xl2-xl1); eld=((le2-le1)-(ll2-ll1)*ec)/(ld2-ld1); printf "%.9f %.9f\n", ec*1e12, eld*1e12/32}'
}
read -r control_const ldg_const < <(attribute "${x_ec1}" "${x_ec2}" "${l_ec1}" "${l_ec2}")
read -r control_static ldg_static < <(attribute "${x_es1}" "${x_es2}" "${l_es1}" "${l_es2}")
printf '%s\n' 'metric,value,unit,method' > "${SUMMARY_CSV}"
printf '%s\n' "control_energy_after_const,${control_const},pJ/LOP3,delta xor-only" >> "${SUMMARY_CSV}"
printf '%s\n' "ldg_energy_after_const,${ldg_const},pJ/bit,load delta minus control delta" >> "${SUMMARY_CSV}"
printf '%s\n' "control_energy_after_const_static,${control_static},pJ/LOP3,delta xor-only" >> "${SUMMARY_CSV}"
printf '%s\n' "ldg_energy_after_const_static,${ldg_static},pJ/bit,load delta minus control delta" >> "${SUMMARY_CSV}"
cat "${SUMMARY_CSV}"
)


# Kept in this dispatcher so each benchmark has a single shell entry point.
run_instruction_mix() (
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.1}"
NCU="${NCU:-${CUDA_HOME}/bin/ncu}"
OUT_DIR="${ROOT_DIR}/build/ncu/executed_instruction_mix"
MD_OUT="${OUT_DIR}/executed_instruction_mix.md"
COMPARISON_CSV="${OUT_DIR}/executed_instruction_mix_comparison.csv"
SOURCE_ALL_COMPARISON_CSV="${OUT_DIR}/executed_instruction_mix_source_all_comparison.csv"
THREAD_COMPARISON_CSV="${OUT_DIR}/executed_instruction_mix_thread_comparison.csv"
PRED_ON_COMPARISON_CSV="${OUT_DIR}/executed_instruction_mix_predicated_on_comparison.csv"

usage() {
  cat <<'EOF'
Usage:
  ./bench.sh instruction-mix [--out-dir DIR] [--markdown OUT.md] REPORT.ncu-rep ...

Use the four `ncu full` reports. The report filename must include r1, r2, r3,
or r4. One CSV and one Markdown table are written per report.
EOF
}

label_from_report() {
  local report="$1" base
  base="$(basename "${report}" .ncu-rep)"
  case "${base}" in
    *load_xor*) printf 'load-xor' ;;
    *xor_only*) printf 'xor-only' ;;
    *) printf '%s' "${base}" ;;
  esac
  if [[ "${base}" =~ r([1-4])([^0-9]|$) ]]; then
    printf ' r%s\n' "${BASH_REMATCH[1]}"
  else
    printf '\n'
  fi
}

extract_mix() {
  local report="$1" out="$2" tmp
  tmp="$(mktemp)"
  "${NCU}" --import "${report}" --csv --page source 2>/dev/null |
    awk '
      BEGIN { OFS="," }
      /^"0x/ {
        line=$0
        # Remove Address; next quoted field is the source/SASS string.
        sub(/^"[^"]*","/, "", line)
        if (!match(line, /^[^"]*"/)) next
        src=substr(line, 1, RLENGTH-1)
        rest=substr(line, RLENGTH+1)

        # Remaining fields are quoted.  4/5/6 are respectively warp, thread,
        # and predicate-true thread instruction counts.
        n=0
        while (match(rest, /"[^"]*"/)) {
          field=substr(rest, RSTART+1, RLENGTH-2)
          values[++n]=field
          rest=substr(rest, RSTART+RLENGTH+1)
        }
        if (n < 6) next

        sub(/^[[:space:]]+/, "", src)
        sub(/^@!?P[0-9]+[[:space:]]+/, "", src)
        split(src, words, /[[:space:]]+/)
        opcode=words[1]
        sub(/\..*$/, "", opcode)
        if (opcode == "" || opcode == "-") next
        # Source page also carries static PCs in never-taken/cold paths.
        # The GUI Executed Instruction Mix hides those zero-execution rows.
        if ((values[4] + 0) == 0 && (values[5] + 0) == 0 && (values[6] + 0) == 0) next

        warp[opcode] += values[4] + 0
        thread[opcode] += values[5] + 0
        pred_on[opcode] += values[6] + 0
      }
      END {
        for (opcode in warp)
          printf "%s,%.0f,%.0f,%.0f\n", opcode, warp[opcode], thread[opcode], pred_on[opcode]
      }
    ' > "${tmp}"
  {
    echo 'opcode,warp_instructions_executed,thread_instructions_executed,predicated_on_thread_instructions_executed'
    sort -t, -k4,4nr -k1,1 "${tmp}"
  } > "${out}"
  rm -f "${tmp}"
}

declare -a reports=()
while (($#)); do
  case "$1" in
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --markdown) MD_OUT="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) reports+=("$1"); shift ;;
  esac
done
[[ ${#reports[@]} -ge 1 ]] || { usage >&2; exit 2; }
[[ -x "${NCU}" ]] || { echo "ncu not found: ${NCU}" >&2; exit 1; }
mkdir -p "${OUT_DIR}" "$(dirname "${MD_OUT}")"

printf '%s\n\n' '# L1 v2 — NCU Executed Instruction Mix' > "${MD_OUT}"
printf '%s\n\n' 'Counts are grouped from the NCU **Source** page by SASS opcode family. `Predicated-on thread instructions` is the most useful column for actual predicate-true work.' >> "${MD_OUT}"

declare -a csvs=() columns=()
for report in "${reports[@]}"; do
  [[ -f "${report}" ]] || { echo "report not found: ${report}" >&2; exit 1; }
  base="$(basename "${report}" .ncu-rep)"
  csv="${OUT_DIR}/${base}.csv"
  extract_mix "${report}" "${csv}"
  csvs+=("${csv}")
  columns+=("${base}_instructions_executed_warp")
  label="$(label_from_report "${report}")"
  {
    printf '## %s\n\n' "${label}"
    printf '| SASS opcode | Warp instructions executed | Thread instructions executed | Predicated-on thread instructions |\n'
    printf '| --- | ---: | ---: | ---: |\n'
    tail -n +2 "${csv}" | awk -F, '{printf "| %s | %s | %s | %s |\n", $1, $2, $3, $4}'
    printf '\n'
  } >> "${MD_OUT}"
  echo "Wrote: ${csv}"
done

# Spreadsheet-friendly comparison: one opcode per row, four run conditions
# side-by-side. This is NCU GUI's `Instructions Executed` value (warp count).
{
  printf 'opcode'
  for col in "${columns[@]}"; do printf ',%s' "${col}"; done
  printf '\n'
  {
    for csv in "${csvs[@]}"; do tail -n +2 "${csv}" | cut -d, -f1; done
  } | sort -u | while IFS= read -r opcode; do
    printf '%s' "${opcode}"
    for csv in "${csvs[@]}"; do
      value="$(awk -F, -v opcode="${opcode}" '$1 == opcode {print $2; exit}' "${csv}")"
      printf ',%s' "${value:-0}"
    done
    printf '\n'
  done
} > "${COMPARISON_CSV}"

# Keep the raw Source-page union, then make the default comparison match the
# opcode families shown by the NCU GUI Executed Instruction Mix chart. Uniform
# prologue instructions (ULDC/USHF) remain available in the raw file.
cp "${COMPARISON_CSV}" "${SOURCE_ALL_COMPARISON_CSV}"
awk -F, 'NR == 1 || $1 ~ /^(LDG|LOP3|ISETP|BRA|IADD3|IMAD|S2R|EXIT)$/' \
  "${SOURCE_ALL_COMPARISON_CSV}" > "${COMPARISON_CSV}"

# Same mix in lane/thread units. This is exactly 32x the warp count for a
# non-divergent full warp, but is useful when a source row is predicated.
{
  printf 'opcode'
  for csv in "${csvs[@]}"; do
    base="$(basename "${csv}" .csv)"
    printf ',%s_thread_instructions_executed' "${base}"
  done
  printf '\n'
  {
    for csv in "${csvs[@]}"; do tail -n +2 "${csv}" | cut -d, -f1; done
  } | sort -u | while IFS= read -r opcode; do
    printf '%s' "${opcode}"
    for csv in "${csvs[@]}"; do
      value="$(awk -F, -v opcode="${opcode}" '$1 == opcode {print $3; exit}' "${csv}")"
      printf ',%s' "${value:-0}"
    done
    printf '\n'
  done
} > "${THREAD_COMPARISON_CSV}"

# Keep predicate-true lane counts as a separate, optional diagnostic file.
{
  printf 'opcode'
  for csv in "${csvs[@]}"; do
    base="$(basename "${csv}" .csv)"
    printf ',%s_predicated_on_thread_instructions' "${base}"
  done
  printf '\n'
  {
    for csv in "${csvs[@]}"; do tail -n +2 "${csv}" | cut -d, -f1; done
  } | sort -u | while IFS= read -r opcode; do
    printf '%s' "${opcode}"
    for csv in "${csvs[@]}"; do
      value="$(awk -F, -v opcode="${opcode}" '$1 == opcode {print $4; exit}' "${csv}")"
      printf ',%s' "${value:-0}"
    done
    printf '\n'
  done
} > "${PRED_ON_COMPARISON_CSV}"

echo "Wrote: ${COMPARISON_CSV}"
echo "Wrote: ${SOURCE_ALL_COMPARISON_CSV}"
echo "Wrote: ${THREAD_COMPARISON_CSV}"
echo "Wrote: ${PRED_ON_COMPARISON_CSV}"
echo "Wrote: ${MD_OUT}"
)

case "${1:-}" in
  build) build ;;
  build-power) build_power ;;
  single) shift; run_single "$@" ;;
  ncu) shift; run_ncu "$@" ;;
  nsys) shift; run_nsys "$@" ;;
  power) shift; run_power "$@" ;;
  differential) shift; run_differential "$@" ;;
  instruction-mix) shift; run_instruction_mix "$@" ;;
  *) usage; exit 2 ;;
esac
