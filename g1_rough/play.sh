#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common/env.sh
source "${TASK_ROOT}/../common/env.sh"

cd "${TASK_ROOT}"
CKPT_ARGS=()
if [[ $# -gt 0 && -f "$1" ]]; then
  CKPT_ARGS=(--checkpoint "$1")
  shift
elif [[ -d "${TASK_ROOT}/logs/rsl_rl/g1_rough" ]]; then
  LATEST="$(ls -1dt "${TASK_ROOT}/logs/rsl_rl/g1_rough"/*/ 2>/dev/null | head -1 || true)"
  if [[ -n "${LATEST}" ]]; then
    CKPT="$(ls -1t "${LATEST}"model_*.pt 2>/dev/null | head -1 || true)"
    if [[ -n "${CKPT}" ]]; then
      CKPT_ARGS=(--checkpoint "${CKPT}")
    fi
  fi
fi

# Fallback: flat base ckpt for zero-shot / scene preview.
if [[ ${#CKPT_ARGS[@]} -eq 0 ]]; then
  BASE="../g1_run_to_goal/logs/rsl_rl/g1_run_to_goal/2026-07-11_16-04-20/model_1749.pt"
  if [[ -f "${BASE}" ]]; then
    CKPT_ARGS=(--checkpoint "${BASE}")
  fi
fi

VIDEO_LENGTH="${VIDEO_LENGTH:-300}"
VIDEO_NAME="${VIDEO_NAME:-play-rough-preview}"

exec "${PYTHON_EXE}" "${COMMON_DIR}/play.py" \
  --task Isaac-G1-Rough-Play-v0 \
  --headless --enable_cameras --video --video_length "${VIDEO_LENGTH}" \
  --video_name "${VIDEO_NAME}" --num_envs 4 \
  "${CKPT_ARGS[@]}" "$@"
