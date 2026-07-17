# Assignment — Robotics HW5 (Isaac Lab)

I recommend reading the original report report.pdf since report-CN.pdf is an AI-translated version. 

Isaac Lab + RSL-RL (PPO) locomotion. Needs `env_isaaclab` and
`ISAACLAB_PATH` (default `/data/Isaac-platform/IsaacLab`).

## Layout

```
assignment/
├── common/                 # shared env.sh, train.py, play.py
├── humanoid_run_to_goal/   # classic humanoid → far target
├── g1_velocity/            # official Unitree G1 flat velocity baseline
├── g1_run_to_goal/         # G1 velocity + run-to-goal + OOD speed
├── g1_rough/               # fine-tune flat G1 on rough terrain
├── README.md
└── report.md               # experiment log / narrative
```

Task dirs each have `train.sh` / `play.sh`, logs, and videos. Entrypoints wrap
`common/train.py` and `common/play.py`.

## Tasks

### 1. `humanoid_run_to_goal/`

Classic humanoid MDP (`Isaac-Humanoid-RunToGoal-v0`), same shaping as
`Isaac-Humanoid-v0` toward `(1000, 0, 0)`. Blue disk at 12 m is visual-only.

```bash
cd humanoid_run_to_goal && ./train.sh && ./play.sh
```

Primary: `logs/.../2026-07-11_15-24-02/model_999.pt`  
Also: longer run `model_1999.pt`

### 2. `g1_velocity/`

Thin shell over built-in `Isaac-Velocity-Flat-G1-v0`.

```bash
cd g1_velocity && ./train.sh && ./play.sh
```

Checkpoint: `logs/rsl_rl/g1_flat/2026-07-11_12-45-01/model_1499.pt`

### 3. `g1_run_to_goal/`

1. **Train** velocity tracking `(vx, vy, ωz)` (G1 flat interface).
2. **Play-to-goal:** freeze policy; `GoalVelocityCommand` maps 2D goal → velocity
   (stand when close).

**OOD forward-speed experiment**

| Policy | Train `vx` | Checkpoint (kept) |
|--------|------------|-------------------|
| Base | `[0, 1]` | `.../g1_run_to_goal/.../model_1749.pt` |
| Wide-vx | `[0, 2]` (finetune; narrower `vy`/`wz`) | `.../g1_wide_vx/.../model_2600.pt` |

Forward-only eval videos: `videos/play_forward/` (`play-fwd`, `play-fwd-ood`, `play-fwd-wide`).

```bash
cd g1_run_to_goal
./train.sh
./play_to_goal.sh
./play_forward.sh
./train_wide_vx.sh <base_ckpt>
```

Details: [`g1_run_to_goal/README.md`](g1_run_to_goal/README.md)

### 4. `g1_rough/`

Fine-tune flat `model_1749` on official rough terrains (stairs / boxes / slopes /
noise). **No height scan** so the flat obs dim matches and resume works.
Train commands: `vx∈[0,2]`, `vy∈[-0.5,0.5]`, `wz∈[-1,1]`, terrain curriculum on.

Stopped early for deadline at **`model_2100.pt`**. Hard demos: 16- and 36-robot
videos under `videos/play/`.

```bash
cd g1_rough
./train.sh ../g1_run_to_goal/logs/rsl_rl/g1_run_to_goal/2026-07-11_16-04-20/model_1749.pt
./play_hard_demo.sh logs/rsl_rl/g1_rough/2026-07-12_00-40-00/model_2100.pt
```

Details: [`g1_rough/README.md`](g1_rough/README.md)

## Environment

```bash
export ISAACLAB_PATH=/data/Isaac-platform/IsaacLab
conda activate env_isaaclab
```

`common/env.sh` sets `PYTHONPATH`, `LD_LIBRARY_PATH`, and the Python binary.
