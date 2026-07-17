#!/usr/bin/env bash
# Forward-only clips at several fixed speeds (vy=0, wz=0).
set -euo pipefail
TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common/env.sh
source "${TASK_ROOT}/../common/env.sh"

cd "${TASK_ROOT}"

CKPT="${1:-logs/rsl_rl/g1_run_to_goal/2026-07-11_16-04-20/model_1749.pt}"
if [[ ! -f "${CKPT}" ]]; then
  echo "Checkpoint not found: ${CKPT}" >&2
  exit 1
fi

NUM_ENVS="${NUM_ENVS:-1}"
VIDEO_LENGTH="${VIDEO_LENGTH:-400}"
OUT_DIR="${TASK_ROOT}/videos/play_forward"
mkdir -p "${OUT_DIR}"

# NAME_PREFIX: "play-fwd" (in-dist) or "play-fwd-ood" (vx > 1)
NAME_PREFIX="${NAME_PREFIX:-play-fwd}"
# Default in-dist speeds; OOD: SPEEDS="1.1 1.2 ... 2.0" NAME_PREFIX=play-fwd-ood
SPEEDS=(${SPEEDS:-0.2 0.4 0.6 0.8 1.0})

echo "[INFO] Checkpoint: ${CKPT}"
echo "[INFO] Speeds: ${SPEEDS[*]}"
echo "[INFO] Output: ${OUT_DIR} (prefix=${NAME_PREFIX})"

idx=0
for VX in "${SPEEDS[@]}"; do
  NAME=$(printf "%s-%02d-vx%.2f" "${NAME_PREFIX}" "${idx}" "${VX}")
  echo "======== [${idx}/${#SPEEDS[@]}] ${NAME} ========"

  "${PYTHON_EXE}" "${COMMON_DIR}/play.py" \
    --task Isaac-G1-RunToGoal-Play-v0 \
    --headless --enable_cameras --video \
    --video_length "${VIDEO_LENGTH}" --video_name "${NAME}" \
    --num_envs "${NUM_ENVS}" --seed 42 \
    --checkpoint "${CKPT}" \
    env.commands.base_velocity.heading_command=false \
    env.commands.base_velocity.rel_standing_envs=0.0 \
    "env.commands.base_velocity.resampling_time_range=[100.0,100.0]" \
    "env.commands.base_velocity.ranges.lin_vel_x=[${VX},${VX}]" \
    "env.commands.base_velocity.ranges.lin_vel_y=[0.0,0.0]" \
    "env.commands.base_velocity.ranges.ang_vel_z=[0.0,0.0]"

  SRC="$(ls -1t "${TASK_ROOT}/videos/play/${NAME}"*.mp4 2>/dev/null | head -1 || true)"
  if [[ -n "${SRC}" ]]; then
    mv -f "${SRC}" "${OUT_DIR}/"
    echo "[INFO] Saved ${OUT_DIR}/$(basename "${SRC}")"
  else
    echo "[WARN] No video for ${NAME}" >&2
  fi
  idx=$((idx + 1))
done

echo "[INFO] Done:"
ls -lt "${OUT_DIR}"
