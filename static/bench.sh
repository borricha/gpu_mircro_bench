#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${ROOT_DIR}/build"
BIN="${BUILD_DIR}/static_nanosleep"
SOURCE="${ROOT_DIR}/static_nanosleep.cu"
CUDA_ARCH="${CUDA_ARCH:-sm_80}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.1}"
NVCC="${NVCC:-${CUDA_HOME}/bin/nvcc}"
NCU="${NCU:-${CUDA_HOME}/bin/ncu}"
NSYS="${NSYS:-${CUDA_HOME}/bin/nsys}"

build() {
  mkdir -p "${BUILD_DIR}"
  if [[ ! -x "${NVCC}" ]]; then
    echo "nvcc not found: ${NVCC} (set CUDA_HOME or NVCC)" >&2
    exit 1
  fi
  "${NVCC}" -std=c++17 -O3 -lineinfo -arch="${CUDA_ARCH}" \
    -Xcompiler -pthread "${SOURCE}" -o "${BIN}" -lnvidia-ml
}

ensure_binary() {
  if [[ ! -x "${BIN}" || "${SOURCE}" -nt "${BIN}" ]]; then build; fi
}

prepare_output_dirs() {
  local args=("$@")
  local i path
  for ((i = 0; i < ${#args[@]}; ++i)); do
    case "${args[i]}" in
      --idle-telemetry-output|--telemetry-output|--output)
        ((++i < ${#args[@]})) || { echo "missing path after option" >&2; exit 2; }
        path="${args[i]}"
        mkdir -p "$(dirname "${path}")"
        ;;
    esac
  done
}

case "${1:-}" in
  build)
    build
    ;;
  run)
    ensure_binary
    shift
    prepare_output_dirs "$@"
    "${BIN}" "$@"
    ;;
  ncu)
    ensure_binary
    shift
    if [[ ! -x "${NCU}" ]]; then
      echo "ncu not found: ${NCU} (set CUDA_HOME or NCU)" >&2
      exit 1
    fi
    mkdir -p "${BUILD_DIR}/ncu"
    REPORT_PREFIX="${NCU_OUTPUT:-${BUILD_DIR}/ncu/static_nanosleep}"
    mkdir -p "$(dirname "${REPORT_PREFIX}")"
    "${NCU}" --target-processes all --clock-control none --replay-mode kernel \
      --kernel-name 'regex:nanosleepKernel' --launch-count 1 --set full \
      --export "${REPORT_PREFIX}" --force-overwrite "${BIN}" "$@"
    ;;
  nsys)
    ensure_binary
    shift
    if [[ ! -x "${NSYS}" ]]; then
      echo "nsys not found: ${NSYS} (set CUDA_HOME or NSYS)" >&2
      exit 1
    fi
    prepare_output_dirs "$@"
    OUT_DIR="${BUILD_DIR}/nsys"
    mkdir -p "${OUT_DIR}"
    # Keep the command substitution out of a nested parameter expansion.
    # Besides being easier to read, this avoids shell-parser differences when
    # the script is invoked through a login wrapper.
    if [[ -n "${NSYS_OUTPUT:-}" ]]; then
      REPORT_PREFIX="${NSYS_OUTPUT}"
    else
      REPORT_PREFIX="${OUT_DIR}/static_nanosleep_$(date +%Y%m%d_%H%M%S)"
    fi
    mkdir -p "$(dirname "${REPORT_PREFIX}")"
    NSYS_TMPDIR="${NSYS_TMPDIR:-${OUT_DIR}/tmp}"
    mkdir -p "${NSYS_TMPDIR}"
    TMPDIR="${NSYS_TMPDIR}" "${NSYS}" profile \
      --force-overwrite true \
      --output "${REPORT_PREFIX}" \
      --trace=cuda,nvtx,osrt \
      --sample=none \
      --cuda-memory-usage=true \
      --gpu-metrics-devices=cuda-visible \
      --gpu-metrics-frequency=100 \
      --enable=nvml_metrics,-i10 \
      "${BIN}" "$@"
    if [[ -f "${REPORT_PREFIX}.nsys-rep" ]]; then
      "${NSYS}" stats --force-overwrite true \
        --report nvtx_sum,nvtx_gpu_proj_sum,cuda_gpu_kern_sum,cuda_kern_exec_sum,cuda_api_sum \
        --format csv --output "${REPORT_PREFIX}_stats" "${REPORT_PREFIX}.nsys-rep"
    else
      echo "ERROR: Nsight Systems did not create ${REPORT_PREFIX}.nsys-rep" >&2
      exit 1
    fi
    ;;
  *)
    cat >&2 <<'EOF'
Usage:
  ./bench.sh build
  ./bench.sh run [static_nanosleep options]
  ./bench.sh ncu [static_nanosleep options]
  ./bench.sh nsys [static_nanosleep options]
EOF
    exit 2
    ;;
esac
