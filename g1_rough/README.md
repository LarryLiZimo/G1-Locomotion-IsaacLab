# G1 Rough Terrain

Fine-tune a **flat** G1 velocity policy on Isaac Lab **rough** terrains, starting
from `g1_run_to_goal` `model_1749.pt`.

## Design

| Piece | Choice |
|-------|--------|
| Terrain | Official `ROUGH_TERRAINS_CFG` (stairs, boxes, slopes, noise) |
| Observations | **No height scan** — same dim as flat ckpt so `--resume` works |
| Commands | `vx∈[0,2]` (widened); `vy∈[-0.5,0.5]`, `wz∈[-1,1]` (unchanged) |
| Rewards | Official G1 rough (velocity tracking + regularizers; `lin_vel_z_l2=0`) |
| Curriculum | Terrain levels auto up/down from level 0 |

## Checkpoints / demos

| Item | Path |
|------|------|
| Inference ckpt | `logs/rsl_rl/g1_rough/2026-07-12_00-40-00/model_2100.pt` |
| Hard ×16 | `videos/play/play-rough-2100-hard-16-step-0.mp4` |
| Hard ×36 clips | `videos/play/play-rough-2100-hard-36-clip*.mp4` |

Training was cut early (~iter 2100 of a planned ~2749) for machine time limits.

## Gym IDs

| ID | Role |
|----|------|
| `Isaac-G1-Rough-v0` | Train / fine-tune |
| `Isaac-G1-Rough-Play-v0` | Generic play |
| `Isaac-G1-Rough-Play-{Easy,Mid,Hard}-v0` | Fixed-difficulty play terrains |

## Run

```bash
cd g1_rough

# Fine-tune (nohup recommended on shared servers)
./train.sh ../g1_run_to_goal/logs/rsl_rl/g1_run_to_goal/2026-07-11_16-04-20/model_1749.pt

# Spectacular hard play: even 1-robot-per-tile spawn, multiple long clips
./play_hard_demo.sh logs/rsl_rl/g1_rough/2026-07-12_00-40-00/model_2100.pt
```

`play_hard_demo.py` remaps curriculum origins so each env owns a unique tile
(Isaac’s default random row sampling otherwise clusters robots).

## Layout

```
g1_rough/
├── g1_env_cfg.py
├── play_hard_demo.py / play_hard_demo.sh
├── train.sh / play.sh
├── agents/
└── logs/  videos/
```
