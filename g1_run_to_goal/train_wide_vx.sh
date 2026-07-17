#!/usr/bin/env bash
# Fine-tune base G1 velocity policy with widened vx (and shrunk vy/wz) sampling.
set -euo pipefail
TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common/env.sh
source "${TASK_ROOT}/../common/env.sh"

cd "${TASK_ROOT}"

BASE_CKPT="${1:-logs/rsl_rl/g1_run_to_goal/2026-07-11_16-04-20/model_1749.pt}"
if [[ ! -f "${BASE_CKPT}" ]]; then
  echo "Base checkpoint not found: ${BASE_CKPT}" >&2
  exit 1
fi
shift || true

NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITERS="${MAX_ITERS:-1000}"

echo "[INFO] Fine-tune from ${BASE_CKPT}"
echo "[INFO] Command ranges: vx∈[0,2], vy∈[-0.2,0.2], wz∈[-0.5,0.5]"
echo "[INFO] Logs -> logs/rsl_rl/g1_wide_vx/"

exec "${PYTHON_EXE}" "${COMMON_DIR}/train.py" \
  --task Isaac-G1-RunToGoal-v0 \
  --headless --no-video \
  --num_envs "${NUM_ENVS}" --max_iterations "${MAX_ITERS}" --seed 42 \
  --resume --checkpoint "${BASE_CKPT}" \
  "$@"
