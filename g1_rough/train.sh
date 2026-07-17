#!/usr/bin/env bash
# Fine-tune flat G1 velocity policy on rough terrain.
set -euo pipefail
TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common/env.sh
source "${TASK_ROOT}/../common/env.sh"

cd "${TASK_ROOT}"

BASE_CKPT="${1:-../g1_run_to_goal/logs/rsl_rl/g1_run_to_goal/2026-07-11_16-04-20/model_1749.pt}"
if [[ ! -f "${BASE_CKPT}" ]]; then
  echo "Base checkpoint not found: ${BASE_CKPT}" >&2
  exit 1
fi
shift || true

NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITERS="${MAX_ITERS:-1000}"

echo "[INFO] Fine-tune from ${BASE_CKPT}"
echo "[INFO] Rough terrain, flat-compatible obs (no height scan)"
echo "[INFO] Commands: vx∈[0,2], vy∈[-0.5,0.5], wz∈[-1,1]"
echo "[INFO] Logs -> logs/rsl_rl/g1_rough/"

exec "${PYTHON_EXE}" "${COMMON_DIR}/train.py" \
  --task Isaac-G1-Rough-v0 \
  --headless --no-video \
  --num_envs "${NUM_ENVS}" --max_iterations "${MAX_ITERS}" --seed 42 \
  --resume --checkpoint "${BASE_CKPT}" \
  "$@"
