# G1 Run-to-Goal

Unitree G1 on flat ground. Train a **velocity-tracking** policy, then at inference convert a
**2D goal** into the same velocity command the policy expects.

## Idea

| Phase | What the policy sees | Who produces `(vx, vy, ωz)` |
|-------|----------------------|----------------------------|
| Train | base velocity command | random sampler (command manager) |
| Play-to-goal | same 3-D command | PD from goal position (`GoalVelocityCommand`) |

Observation / action spaces stay compatible with official G1 flat locomotion.

## Gym IDs

| ID | Role |
|----|------|
| `Isaac-G1-RunToGoal-v0` | Train (velocity tracking) |
| `Isaac-G1-RunToGoal-Play-v0` | Play with fixed / random velocity commands |
| `Isaac-G1-RunToGoal-GoalPlay-v0` | Play with goal → velocity PD |

---

## OOD forward-speed experiment

**Question:** The base policy only ever saw forward commands with `vx ≤ 1` during training.
If we ask it to track `vx > 1` at test time, does tracking / gait break? After fine-tuning with
a wider `vx` range, does that failure go away?

Here **OOD = out-of-distribution velocity commands**, not a new robot or terrain. Everything
else (flat plane, same obs/actions) stays the same; only the commanded forward speed leaves
the training support.

### Setup

Two policies, same architecture:

| Policy | Train command ranges | Checkpoint |
|--------|----------------------|------------|
| **Base** | `vx ∈ [0, 1]`, `vy ∈ [-0.5, 0.5]`, `wz ∈ [-1, 1]` | `logs/rsl_rl/g1_run_to_goal/2026-07-11_16-04-20/model_1749.pt` |
| **Wide-vx** | `vx ∈ [0, 2]`, `vy ∈ [-0.2, 0.2]`, `wz ∈ [-0.5, 0.5]` (fine-tune from base) | `logs/rsl_rl/g1_wide_vx/2026-07-11_23-01-24/model_2600.pt` |

Wide-vx keeps side-step and yaw **random but narrower**, so capacity is spent on covering
faster forward speeds rather than large lateral / spin commands.

### Evaluation protocol (fair comparison)

Play **forward-only** clips: command `(vx, 0, 0)` held fixed for the whole video
(`heading_command` off, no standing envs).

| Split | Speeds | Role |
|-------|--------|------|
| In-distribution | `vx ∈ {0.2, 0.4, 0.6, 0.8, 1.0}` | Both policies should handle these |
| OOD (for base) | `vx ∈ {1.1, 1.2, …, 2.0}` | Stress test; in-range only for wide-vx |

Videos for the **base** policy (already recorded):

- In-dist: `videos/play_forward/play-fwd-*-vx0.20…1.00-*.mp4`
- OOD: `videos/play_forward/play-fwd-ood-*-vx1.10…2.00-*.mp4`

After wide-vx training finishes, re-run the **same** speed list with that checkpoint (use a
distinct `NAME_PREFIX`, e.g. `play-fwd-wide`) and compare side-by-side at each `vx`.

### Reproduce

```bash
# Base policy — in-dist
./play_forward.sh logs/rsl_rl/g1_run_to_goal/2026-07-11_16-04-20/model_1749.pt

# Base policy — OOD vx
NAME_PREFIX=play-fwd-ood SPEEDS="1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0" \
  ./play_forward.sh logs/rsl_rl/g1_run_to_goal/2026-07-11_16-04-20/model_1749.pt

# Fine-tune with wider vx (then play the same sweeps on the new ckpt)
./train_wide_vx.sh logs/rsl_rl/g1_run_to_goal/2026-07-11_16-04-20/model_1749.pt
NAME_PREFIX=play-fwd-wide SPEEDS="0.2 0.4 0.6 0.8 1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0" \
  ./play_forward.sh logs/rsl_rl/g1_wide_vx/2026-07-11_23-01-24/model_2600.pt
```

Wide-vx videos: `videos/play_forward/play-fwd-wide-*-vx0.20…2.00-*.mp4`
---

## Run-to-goal (inference)

`mdp/commands.py` → `GoalVelocityCommand`:

1. Hold a world-frame XY goal (green marker).
2. Each step: goal → base frame `(dx, dy)`.
3. If distance `< 0.35 m` → `(0, 0, 0)` (stand).
4. Else → proportional `(vx, vy, wz)`, clipped to velocity limits.

```bash
./play_to_goal.sh [checkpoint]
# → videos/play_to_goal/
```

## Other play scripts

```bash
./play.sh              # default velocity play
./play_diverse.sh      # stand / turns / random fixed commands → videos/play_diverse/
```

## Layout

```
g1_run_to_goal/
├── g1_env_cfg.py              # train / velocity-play / goal-play
├── mdp/commands.py            # GoalVelocityCommand
├── agents/rsl_rl_ppo_cfg.py   # experiment_name=g1_wide_vx for fine-tune
├── train.sh / train_wide_vx.sh
├── play.sh / play_to_goal.sh / play_forward.sh / play_diverse.sh
└── logs/  videos/
```
