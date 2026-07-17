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
elif [[ -d "${TASK_ROOT}/logs/rsl_rl/g1_run_to_goal" ]]; then
  LATEST="$(ls -1dt "${TASK_ROOT}/logs/rsl_rl/g1_run_to_goal"/*/ 2>/dev/null | head -1 || true)"
  if [[ -n "${LATEST}" ]]; then
    CKPT="$(ls -1t "${LATEST}"model_*.pt 2>/dev/null | head -1 || true)"
    if [[ -n "${CKPT}" ]]; then
      CKPT_ARGS=(--checkpoint "${CKPT}")
    fi
  fi
fi

exec "${PYTHON_EXE}" "${COMMON_DIR}/play.py" \
  --task Isaac-G1-RunToGoal-Play-v0 \
  --headless --enable_cameras --video --video_length 400 --num_envs 4 \
  "${CKPT_ARGS[@]}" "$@"
