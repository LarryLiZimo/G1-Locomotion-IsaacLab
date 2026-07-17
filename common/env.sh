# Shared Isaac Lab / conda setup. Source from task train.sh / play.sh.
# Expects TASK_ROOT to be set by the caller.

ASSIGNMENT_ROOT="$(cd "${TASK_ROOT}/.." && pwd)"
export ISAACLAB_PATH="${ISAACLAB_PATH:-/data/Isaac-platform/IsaacLab}"
export PYTHONPATH="${ASSIGNMENT_ROOT}:${PYTHONPATH:-}"

if [[ -n "${CONDA_PREFIX:-}" ]]; then
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
  PYTHON_EXE="${CONDA_PREFIX}/bin/python"
else
  PYTHON_EXE="${HOME}/miniconda3/envs/env_isaaclab/bin/python"
  export LD_LIBRARY_PATH="$(dirname "${PYTHON_EXE}")/../lib:${LD_LIBRARY_PATH:-}"
fi

COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
