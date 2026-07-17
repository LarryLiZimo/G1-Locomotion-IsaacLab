#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common/env.sh
source "${TASK_ROOT}/../common/env.sh"

cd "${TASK_ROOT}"
NUM_ENVS="${NUM_ENVS:-1024}"
MAX_ITERS="${MAX_ITERS:-1500}"

exec "${PYTHON_EXE}" "${COMMON_DIR}/train.py" \
  --task Isaac-Velocity-Flat-G1-v0 \
  --headless --no-video \
  --num_envs "${NUM_ENVS}" --max_iterations "${MAX_ITERS}" --seed 42 \
  "$@"
