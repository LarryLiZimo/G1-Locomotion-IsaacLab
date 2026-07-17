#!/usr/bin/env bash
# Inference: run the trained velocity policy to a 2D goal (goal → vx,vy,wz).
# Records one or more clips with different goal positions.
set -euo pipefail
TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common/env.sh
source "${TASK_ROOT}/../common/env.sh"

cd "${TASK_ROOT}"

CKPT="${1:-}"
if [[ -n "${CKPT}" && -f "${CKPT}" ]]; then
  shift
elif [[ -f "${TASK_ROOT}/logs/rsl_rl/g1_run_to_goal/2026-07-11_16-04-20/model_1749.pt" ]]; then
  CKPT="${TASK_ROOT}/logs/rsl_rl/g1_run_to_goal/2026-07-11_16-04-20/model_1749.pt"
elif [[ -d "${TASK_ROOT}/logs/rsl_rl/g1_run_to_goal" ]]; then
  LATEST="$(ls -1dt "${TASK_ROOT}/logs/rsl_rl/g1_run_to_goal"/*/ 2>/dev/null | head -1 || true)"
  CKPT="$(ls -1t "${LATEST}"model_*.pt 2>/dev/null | head -1 || true)"
fi
if [[ -z "${CKPT}" || ! -f "${CKPT}" ]]; then
  echo "No checkpoint. Pass: ./play_to_goal.sh path/to/model.pt" >&2
  exit 1
fi

NUM_ENVS="${NUM_ENVS:-1}"
VIDEO_LENGTH="${VIDEO_LENGTH:-600}"  # longer: walk then stand
OUT_DIR="${TASK_ROOT}/videos/play_to_goal"
mkdir -p "${OUT_DIR}"

# Goals relative to env origin: tag gx gy
# Default set covers straight, left, right, near, far.
GOALS=(
  "front_3m 3.0 0.0"
  "front_5m 5.0 0.0"
  "left_3m 3.0 1.5"
  "right_3m 3.0 -1.5"
  "near_2m 2.0 0.5"
)

echo "[INFO] Checkpoint: ${CKPT}"
echo "[INFO] Clips: ${#GOALS[@]} -> ${OUT_DIR}"

idx=0
for line in "${GOALS[@]}"; do
  read -r TAG GX GY <<<"${line}"
  NAME=$(printf "goal-%02d-%s-x%+.1f-y%+.1f" "${idx}" "${TAG}" "${GX}" "${GY}")
  echo "======== [${idx}/${#GOALS[@]}] ${NAME} ========"

  "${PYTHON_EXE}" "${COMMON_DIR}/play.py" \
    --task Isaac-G1-RunToGoal-GoalPlay-v0 \
    --headless --enable_cameras --video \
    --video_length "${VIDEO_LENGTH}" --video_name "${NAME}" \
    --num_envs "${NUM_ENVS}" --seed 42 \
    --checkpoint "${CKPT}" \
    "env.commands.base_velocity.ranges.pos_x=[${GX},${GX}]" \
    "env.commands.base_velocity.ranges.pos_y=[${GY},${GY}]" \
    "$@"

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
