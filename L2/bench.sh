#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${ROOT_DIR}/build"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.1}"
NVCC="${NVCC:-${CUDA_HOME}/bin/nvcc}"
NCU="${NCU:-${CUDA_HOME}/bin/ncu}"
NSYS="${NSYS:-${CUDA_HOME}/bin/nsys}"
CUDA_ARCH="${CUDA_ARCH:-sm_80}"
SOURCE="${ROOT_DIR}/load_xor.cu"
XOR_SOURCE="${ROOT_DIR}/xor_only.cu"
POWER_SOURCE="${ROOT_DIR}/power_driver.cu"
COMMON="${ROOT_DIR}/common.cuh"
BIN="${BUILD_DIR}/load_xor"
XOR_BIN="${BUILD_DIR}/xor_only"
POWER_BIN="${BUILD_DIR}/power_driver"

usage() {
  cat <<'EOF'
Usage:
  ./bench.sh build
  ./bench.sh single [load-xor|xor-only] [kernel options]
  ./bench.sh ncu [verify|full] [load-xor|xor-only] [kernel options]
  ./bench.sh nsys [single|power] load-xor|xor-only [options]
  ./bench.sh power [load-xor|xor-only] [power options]
  ./bench.sh instruction-mix REPORT.ncu-rep ...
  ./bench.sh differential [differential options]

Kernel options:
  --device N --active-loads 64|128 (or --rounds 1|2)
  --iterations N --input random|zero|one-point-one
  --launch-mode single|graph --graph-nodes N --graph-replays N
EOF
}

binary_for() {
  case "$1" in
    load-xor) printf '%s\n' "${BIN}" ;;
    xor-only) printf '%s\n' "${XOR_BIN}" ;;
    *) echo "invalid kernel kind: $1" >&2; exit 2 ;;
  esac
}

source_for() {
  case "$1" in
    load-xor) printf '%s\n' "${SOURCE}" ;;
    xor-only) printf '%s\n' "${XOR_SOURCE}" ;;
    *) echo "invalid kernel kind: $1" >&2; exit 2 ;;
  esac
}

kernel_regex_for() {
  case "$1" in
    load-xor) printf '%s\n' 'regex:l2LoadXorKernel' ;;
    xor-only) printf '%s\n' 'regex:l2XorOnlyKernel' ;;
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

ensure_power_binary() {
  [[ -x "${POWER_BIN}" && "${POWER_BIN}" -nt "${POWER_SOURCE}" &&
     "${POWER_BIN}" -nt "${COMMON}" ]] || build_power
}

run_power() {
  local kind=""
  case "${1:-}" in
    load-xor|xor-only) kind="$1"; shift ;;
  esac
  ensure_power_binary
  if [[ -n "${kind}" ]]; then
    "${POWER_BIN}" --power-trial --kind "${kind}" "$@"
  else
    "${POWER_BIN}" --power-trial "$@"
  fi
}

run_ncu() {
  local profile="${1:-verify}"
  case "${profile}" in verify|full) shift ;; *) profile=verify ;; esac
  local kind="${1:-load-xor}"
  case "${kind}" in load-xor|xor-only) shift ;; *) kind=load-xor ;; esac
  ensure_binary "${kind}"
  [[ -x "${NCU}" ]] || { echo "ncu not found: ${NCU}" >&2; exit 1; }
  local prefix="${NCU_OUTPUT:-${BUILD_DIR}/ncu/l2v2_${profile}}"
  mkdir -p "$(dirname "${prefix}")"
  local args=(--target-processes all --clock-control none --replay-mode kernel
    --kernel-name "$(kernel_regex_for "${kind}")" --launch-count 1
    --export "${prefix}" --force-overwrite)
  if [[ "${profile}" == verify ]]; then
    args+=(--metrics 'sm__sass_thread_inst_executed,sm__sass_thread_inst_executed_op_integer_pred_on,sm__sass_thread_inst_executed_op_control_pred_on,smsp__inst_executed,l1tex__t_requests_pipe_lsu_mem_global_op_ld,l1tex__t_sector_hit_rate,lts__t_sector_hit_rate,lts__t_sectors_op_read,dram__bytes_read,dram__bytes_write')
  else
    args+=(--set full)
  fi
  "${NCU}" "${args[@]}" "$(binary_for "${kind}")" "$@"
}


run_nsys() {
  local mode="${1:-single}" kind prefix tmp_dir
  local -a target=()
  case "${mode}" in
    single)
      shift
      kind="${1:-load-xor}"
      case "${kind}" in load-xor|xor-only) shift ;; *) kind=load-xor ;; esac
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
  [[ -x "${NSYS}" ]] || { echo "nsys not found: ${NSYS}" >&2; exit 1; }
  prefix="${NSYS_OUTPUT:-${BUILD_DIR}/nsys/l2v2_${mode}_${kind}}"
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
# Kept in this dispatcher so each benchmark has a single shell entry point.
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
ALLOCATED_MIB="${ALLOCATED_MIB:-1024}"
LOAD_GRAPH_NODES="${LOAD_GRAPH_NODES:-5000}"
XOR_GRAPH_NODES="${XOR_GRAPH_NODES:-50000}"
INPUT="${INPUT:-random}"
RUN_TAG="${RUN_TAG:-sass_slope_v1framing_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/build/power/${RUN_TAG}}"

usage() {
  cat <<'EOF'
Usage: ./bench.sh differential [options]
  --device N --p-const-w W --p-static-w W --allocated-mib N
  --precondition-seconds S --cooldown-seconds S --sample-ms MS --repetitions N
  --load-graph-nodes N --xor-graph-nodes N --input random|zero|one-point-one
  --output-dir DIR
EOF
}

while (($#)); do
  case "$1" in
    --device) DEVICE="$2"; shift 2 ;;
    --p-const-w) P_CONST_W="$2"; shift 2 ;;
    --p-static-w) P_STATIC_W="$2"; shift 2 ;;
    --allocated-mib) ALLOCATED_MIB="$2"; shift 2 ;;
    --precondition-seconds) PRECONDITION_SECONDS="$2"; shift 2 ;;
    --cooldown-seconds) COOLDOWN_SECONDS="$2"; shift 2 ;;
    --sample-ms) SAMPLE_MS="$2"; shift 2 ;;
    --repetitions) REPETITIONS="$2"; shift 2 ;;
    --load-graph-nodes) LOAD_GRAPH_NODES="$2"; shift 2 ;;
    --xor-graph-nodes) XOR_GRAPH_NODES="$2"; shift 2 ;;
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
  local kind="$1" active="$2" rep="$3" nodes telemetry
  if [[ "${kind}" == "load-xor" ]]; then nodes="${LOAD_GRAPH_NODES}"; else nodes="${XOR_GRAPH_NODES}"; fi
  telemetry="${OUT_DIR}/telemetry/${kind}_r$((active / 64))_rep${rep}.csv"
  echo "== ${kind} r$((active / 64)), repetition ${rep}/${REPETITIONS}: v1 in-process framing =="
  "${BIN}" --power-trial --device "${DEVICE}" --kind "${kind}" --active-loads "${active}" \
    --precondition-seconds "${PRECONDITION_SECONDS}" --sample-ms "${SAMPLE_MS}" \
    --p-constant-w "${P_CONST_W}" --p-static-w "${P_STATIC_W}" \
    --allocated-mib "${ALLOCATED_MIB}" --input "${INPUT}" --launch-batch 32 \
    --graph-nodes "${nodes}" --telemetry-output "${telemetry}" --output "${RAW_CSV}"
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

usage() {
  echo "Usage: $0 [--out-dir DIR] REPORT.ncu-rep [REPORT.ncu-rep ...]" >&2
}

extract_one() {
  local report="$1" out="$2" tmp
  tmp="$(mktemp)"
  "${NCU}" --import "${report}" --csv --page source 2>/dev/null |
    awk '
      BEGIN { OFS="," }
      /^"0x/ {
        line=$0
        sub(/^"[^"]*","/, "", line)
        if (!match(line, /^[^"]*"/)) next
        src=substr(line, 1, RLENGTH-1)
        rest=substr(line, RLENGTH+1)
        n=0
        while (match(rest, /"[^"]*"/)) {
          value[++n]=substr(rest, RSTART+1, RLENGTH-2)
          rest=substr(rest, RSTART+RLENGTH+1)
        }
        # Source page fields 4/5/6: warp, thread, predicate-true thread counts.
        if (n < 6) next
        sub(/^[[:space:]]+/, "", src)
        sub(/^@!?P[0-9]+[[:space:]]+/, "", src)
        split(src, words, /[[:space:]]+/)
        opcode=words[1]
        sub(/\..*$/, "", opcode)
        if (opcode == "" || opcode == "-") next
        # BRA/BPT/NOP can exist in SASS even when never executed. Preserve
        # zero-count rows because the opcode union itself is a verification target.
        seen[opcode] = 1
        warp[opcode] += value[4]+0
        thread[opcode] += value[5]+0
        pred_on[opcode] += value[6]+0
      }
      END {
        for (opcode in seen)
          printf "%s,%.0f,%.0f,%.0f\n", opcode, warp[opcode], thread[opcode], pred_on[opcode]
      }
    ' > "${tmp}"
  {
    echo 'opcode,warp_instructions_executed,thread_instructions_executed,predicated_on_thread_instructions_executed'
    sort -t, -k2,2nr -k1,1 "${tmp}"
  } > "${out}"
  rm -f "${tmp}"
}

OUT_OVERRIDE=""
declare -a reports=()
while (($#)); do
  case "$1" in
    --out-dir) OUT_OVERRIDE="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) reports+=("$1"); shift ;;
  esac
done
[[ ${#reports[@]} -ge 1 ]] || { usage; exit 2; }
[[ -z "${OUT_OVERRIDE}" ]] || OUT_DIR="${OUT_OVERRIDE}"
[[ -x "${NCU}" ]] || { echo "ncu not found: ${NCU}" >&2; exit 1; }
mkdir -p "${OUT_DIR}"

declare -a csvs=() labels=()
for report in "${reports[@]}"; do
  [[ -f "${report}" ]] || { echo "report not found: ${report}" >&2; exit 1; }
  base="$(basename "${report}" .ncu-rep)"
  csv="${OUT_DIR}/${base}.csv"
  extract_one "${report}" "${csv}"
  csvs+=("${csv}")
  if [[ "${base}" == *load_xor* || "${base}" == *v1style* ]]; then
    kind="load-xor"
  elif [[ "${base}" == *xor_only* ]]; then
    kind="xor-only"
  else
    kind="${base}"
  fi
  if [[ "${base}" =~ r([1-4])([^0-9]|$) ]]; then
    labels+=("${kind}_r${BASH_REMATCH[1]}")
  else
    labels+=("${base}")
  fi
done

# Use the union of every opcode executed in at least one report as the default
# table. This includes branch, control, conversion, and special-register
# opcodes in addition to LDG/LOP3, with zeroes retained in every condition.
raw="${OUT_DIR}/executed_instruction_mix_source_all_comparison.csv"
main="${OUT_DIR}/executed_instruction_mix_comparison.csv"
thread="${OUT_DIR}/executed_instruction_mix_thread_comparison.csv"
pred="${OUT_DIR}/executed_instruction_mix_predicated_on_comparison.csv"

write_comparison() {
  local field="$1" output="$2" suffix="$3"
  {
    printf 'opcode'
    for label in "${labels[@]}"; do printf ',%s%s' "${label}" "${suffix}"; done
    printf '\n'
    { for csv in "${csvs[@]}"; do tail -n +2 "${csv}" | cut -d, -f1; done; } | sort -u |
      while IFS= read -r opcode; do
        printf '%s' "${opcode}"
        for csv in "${csvs[@]}"; do
          value="$(awk -F, -v opcode="${opcode}" -v f="${field}" '$1 == opcode {print $f; exit}' "${csv}")"
          printf ',%s' "${value:-0}"
        done
        printf '\n'
      done
  } > "${output}"
}

write_comparison 2 "${raw}" '_instructions_executed_warp'
cp "${raw}" "${main}"
write_comparison 3 "${thread}" '_thread_instructions_executed'
write_comparison 4 "${pred}" '_predicated_on_thread_instructions'

echo "Wrote: ${main}"
echo "Wrote: ${raw}"
echo "Wrote: ${thread}"
echo "Wrote: ${pred}"
)

case "${1:-}" in
  build) build ;;
  single)
    shift
    kind="${1:-load-xor}"
    case "${kind}" in load-xor|xor-only) shift ;; *) kind=load-xor ;; esac
    ensure_binary "${kind}"
    "$(binary_for "${kind}")" "$@"
    ;;
  ncu) shift; run_ncu "$@" ;;
  nsys) shift; run_nsys "$@" ;;
  power) shift; run_power "$@" ;;
  instruction-mix) shift; run_instruction_mix "$@" ;;
  differential) shift; run_differential "$@" ;;
  *) usage >&2; exit 2 ;;
esac
