#!/usr/bin/env bash
# 36 robots, even tile spawn, 3 long hard clips.
set -euo pipefail
TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common/env.sh
source "${TASK_ROOT}/../common/env.sh"
cd "${TASK_ROOT}"

CKPT="${1:-logs/rsl_rl/g1_rough/2026-07-12_00-40-00/model_2100.pt}"
if [[ ! -f "${CKPT}" ]]; then
  echo "Missing checkpoint: ${CKPT}" >&2
  exit 1
fi

VIDEO_LENGTH="${VIDEO_LENGTH:-900}"
NUM_CLIPS="${NUM_CLIPS:-3}"

echo "[INFO] ckpt=${CKPT}  envs=36  clips=${NUM_CLIPS}  length=${VIDEO_LENGTH}"
exec "${PYTHON_EXE}" "${TASK_ROOT}/play_hard_demo.py" \
  --task Isaac-G1-Rough-Play-Hard-v0 \
  --headless --enable_cameras --video \
  --video_length "${VIDEO_LENGTH}" \
  --video_name_prefix play-rough-2100-hard-36 \
  --num_clips "${NUM_CLIPS}" --num_envs 36 --seed 42 \
  --checkpoint "${CKPT}"
