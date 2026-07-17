#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common/env.sh
source "${TASK_ROOT}/../common/env.sh"

cd "${TASK_ROOT}"
DEFAULT_CKPT="${TASK_ROOT}/logs/rsl_rl/g1_flat/2026-07-11_12-45-01/model_1499.pt"
CKPT_ARGS=()
if [[ $# -eq 0 && -f "${DEFAULT_CKPT}" ]]; then
  CKPT_ARGS=(--checkpoint "${DEFAULT_CKPT}")
fi

exec "${PYTHON_EXE}" "${COMMON_DIR}/play.py" \
  --task Isaac-Velocity-Flat-G1-Play-v0 \
  --headless --enable_cameras --video --video_length 400 --num_envs 4 \
  "${CKPT_ARGS[@]}" "$@"
