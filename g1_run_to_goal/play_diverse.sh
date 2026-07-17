#!/usr/bin/env bash
# Record several play clips, each with a fixed velocity command (vx, vy, wz).
# Samples from the official G1 flat ranges, plus a few hand-picked cases.
set -euo pipefail
TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common/env.sh
source "${TASK_ROOT}/../common/env.sh"

cd "${TASK_ROOT}"

CKPT="${1:-}"
if [[ -n "${CKPT}" && -f "${CKPT}" ]]; then
  shift
elif [[ -d "${TASK_ROOT}/logs/rsl_rl/g1_run_to_goal" ]]; then
  LATEST="$(ls -1dt "${TASK_ROOT}/logs/rsl_rl/g1_run_to_goal"/*/ 2>/dev/null | head -1 || true)"
  CKPT="$(ls -1t "${LATEST}"model_*.pt 2>/dev/null | head -1 || true)"
fi
if [[ -z "${CKPT}" || ! -f "${CKPT}" ]]; then
  echo "No checkpoint found. Pass one: ./play_diverse.sh path/to/model_XXXX.pt" >&2
  exit 1
fi

NUM_RANDOM="${NUM_RANDOM:-6}"
SEED="${SEED:-42}"
NUM_ENVS="${NUM_ENVS:-1}"
VIDEO_LENGTH="${VIDEO_LENGTH:-400}"
OUT_DIR="${TASK_ROOT}/videos/play_diverse"
mkdir -p "${OUT_DIR}"

# Sample commands: print lines "tag vx vy wz"
mapfile -t CMDS < <("${PYTHON_EXE}" - "${NUM_RANDOM}" "${SEED}" <<'PY'
import random, sys
n, seed = int(sys.argv[1]), int(sys.argv[2])
rng = random.Random(seed)
# Always include these fixed demos first.
fixed = [
    ("stand", 0.0, 0.0, 0.0),
    ("forward", 1.0, 0.0, 0.0),
    ("forward_left", 0.8, 0.0, 0.8),
    ("forward_right", 0.8, 0.0, -0.8),
    ("side_left", 0.3, 0.4, 0.0),
    ("slow", 0.3, 0.0, 0.0),
]
for tag, vx, vy, wz in fixed:
    print(f"{tag} {vx:.3f} {vy:.3f} {wz:.3f}")
# Random samples from official G1 flat ranges.
for i in range(n):
    vx = rng.uniform(0.0, 1.0)
    vy = rng.uniform(-0.5, 0.5)
    wz = rng.uniform(-1.0, 1.0)
    print(f"rand{i:02d} {vx:.3f} {vy:.3f} {wz:.3f}")
PY
)

echo "[INFO] Checkpoint: ${CKPT}"
echo "[INFO] Recording ${#CMDS[@]} clips -> ${OUT_DIR}"

idx=0
for line in "${CMDS[@]}"; do
  read -r TAG VX VY WZ <<<"${line}"
  NAME=$(printf "play-%02d-%s-vx%+.2f-vy%+.2f-wz%+.2f" "${idx}" "${TAG}" "${VX}" "${VY}" "${WZ}")
  echo "======== [${idx}/${#CMDS[@]}] ${NAME} ========"

  # Fix command for the whole clip: disable heading mode so wz is used as-is.
  # Hydra keys are under env.* (see isaaclab_tasks.utils.hydra).
  "${PYTHON_EXE}" "${COMMON_DIR}/play.py" \
    --task Isaac-G1-RunToGoal-Play-v0 \
    --headless --enable_cameras --video \
    --video_length "${VIDEO_LENGTH}" --video_name "${NAME}" \
    --num_envs "${NUM_ENVS}" --seed "${SEED}" \
    --checkpoint "${CKPT}" \
    env.commands.base_velocity.heading_command=false \
    env.commands.base_velocity.rel_standing_envs=0.0 \
    "env.commands.base_velocity.resampling_time_range=[100.0,100.0]" \
    "env.commands.base_velocity.ranges.lin_vel_x=[${VX},${VX}]" \
    "env.commands.base_velocity.ranges.lin_vel_y=[${VY},${VY}]" \
    "env.commands.base_velocity.ranges.ang_vel_z=[${WZ},${WZ}]" \
    "$@"

  # Move clip from default videos/play into play_diverse/
  SRC="$(ls -1t "${TASK_ROOT}/videos/play/${NAME}"*.mp4 2>/dev/null | head -1 || true)"
  if [[ -n "${SRC}" ]]; then
    mv -f "${SRC}" "${OUT_DIR}/"
    echo "[INFO] Saved ${OUT_DIR}/$(basename "${SRC}")"
  else
    echo "[WARN] No video found for ${NAME}" >&2
  fi
  idx=$((idx + 1))
done

echo "[INFO] Done. Clips in ${OUT_DIR}:"
ls -lt "${OUT_DIR}"
