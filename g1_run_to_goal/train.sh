#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common/env.sh
source "${TASK_ROOT}/../common/env.sh"

cd "${TASK_ROOT}"
# Official G1 flat defaults: 4096 envs, 1500 iters. Lower NUM_ENVS on small GPUs.
NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITERS="${MAX_ITERS:-1500}"

exec "${PYTHON_EXE}" "${COMMON_DIR}/train.py" \
  --task Isaac-G1-RunToGoal-v0 \
  --headless --no-video \
  --num_envs "${NUM_ENVS}" --max_iterations "${MAX_ITERS}" --seed 42 \
  "$@"
