#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_PATH="${1:-${PROJECT_ROOT}/third_party/Grounded-SAM-2}"

if [[ ! -f "${REPO_PATH}/setup.py" || ! -d "${REPO_PATH}/sam2" ]]; then
  echo "Grounded-SAM-2 source was not found at: ${REPO_PATH}" >&2
  echo "Upload or clone the official repository, then rerun this script." >&2
  exit 1
fi

if [[ "${SKIP_DEPENDENCIES:-0}" != "1" ]]; then
  if [[ "${OFFLINE:-0}" == "1" ]]; then
    WHEELHOUSE="${WHEELHOUSE:-${PROJECT_ROOT}/wheelhouse}"
    if [[ ! -d "${WHEELHOUSE}" ]]; then
      echo "Offline wheelhouse was not found at: ${WHEELHOUSE}" >&2
      exit 1
    fi
    python -m pip install --no-index --find-links "${WHEELHOUSE}" \
      -r "${PROJECT_ROOT}/requirements-grounded-sam2.txt"
  else
    python -m pip install -r "${PROJECT_ROOT}/requirements-grounded-sam2.txt"
  fi
fi

# Preserve the verified torch 2.5.1+cu124 environment and avoid requiring nvcc
# for the first image-pipeline smoke test. Editable mode keeps upstream changes live.
SAM2_BUILD_CUDA="${SAM2_BUILD_CUDA:-0}" \
  python -m pip install --no-build-isolation --no-deps -e "${REPO_PATH}"

python "${PROJECT_ROOT}/scripts/check_grounded_sam2_env.py" \
  --config "${PROJECT_ROOT}/configs/grounded_sam2.yaml"
