# NewRL

NewRL is a GPU-batched robot learning codebase built on Newton and PyTorch.
It provides locomotion and motion tracking tasks, PPO and GenPO++ learners,
and tools for inspecting, recording, and branching rollouts.

## Installation

Use Python 3.13, [uv](https://docs.astral.sh/uv/), and a CUDA-capable NVIDIA GPU.
Clone NewRL and Newton as sibling checkouts, then create the NewRL environment.
The project points to `../newton` for its editable Newton dependency.

```sh
git clone https://github.com/LearningFromEmbodiedExperienceLab/newrl.git
git clone https://github.com/newton-physics/newton.git
cd newrl
uv sync
```

For Viser, use `uv sync --extra viser`. The custom browser viewer also needs
Node.js 22.12 or newer to build its local frontend:

```sh
npm --prefix viewer ci
npm --prefix viewer run build
```

To install zsh completion, run `scripts/install_zsh_completion.sh` from the
checkout and open a new zsh. Activate a project's environment before running the
installer if you want its presets included. Refresh the generated completion
after changing NewRL or installing projects.

## Basic usage

```sh
# Inspect a task
uv run newrl play task=AnymalLocoFlat

# Train a policy
uv run newrl train task=AnymalLocoFlat algo=ppo task.num_envs=4096

# Play a checkpoint in the custom viewer
uv run newrl play checkpoint=path/to/policy.pt num_envs=1

# Rewind and branch a stochastic rollout in one world
uv run newrl play-timeline checkpoint=path/to/policy.pt deterministic=false
```

Playback opens the custom viewer with the 1 m grid enabled. Open the URL it
prints. Use `viewer=none steps=100` for a short headless run, `viewer=gl`
for Newton GL, `viewer=viser` for Viser, or `viewer.port=0` to request a free port.

## Features

- **Tasks and terrain.** Quadruped and wheeled locomotion, G1 motion tracking,
  procedural terrain, reconstructed meshes, contact sensors, and RGB or depth
  cameras. [Simulation and task design](docs/simulation.md)
- **Training.** PPO, PPO SymAug, GenPO++, and SuperMimic. The built-in trainer
  supports checkpoints, W&B logging, and multi-GPU training for on-policy
  learners. [Training reference](docs/training.md)
- **Visualization.** Robot follow, free camera controls, Gaussian splats,
  camera images, markers, video recording, and a timeline for rewinding and
  branching. [Viewer guide](docs/viewer.md)
- **Research projects.** Install separately managed projects with their own
  dependencies and namespaced presets. [Project guide](docs/projects.md) ·
  [Robot contribution guide](docs/robots.md)
- **Simulation APIs.** Stateful and stateless environments support explicit
  resets and checkpoints. [Design overview](design.md) ·
  [Benchmark guide](benchmarks/README.md)

## Troubleshooting

- **The browser viewer assets are missing.** Run the two npm commands under
  Installation from the source checkout.
- **Chrome cannot use WebGPU or picks the integrated GPU.** Check
  `chrome://gpu`. See the [NVIDIA offload flags](docs/viewer.md#chrome-webgpu-troubleshooting),
  or append `?renderer=webgl` to the viewer URL to use WebGL2.
- **A Gaussian camera task takes a long time to start.** The first simulator
  image can compile gsplat's CUDA extension. For a static scene preview with a
  policy that has no camera inputs, use `discard_unused_obs=true`.
  [Gaussian camera setup](docs/viewer.md#gaussian-camera-sensors)
- **Completion is missing a command or preset.** Rerun the zsh installer in the
  environment that supplies your presets, then open a new shell.

## Citation

If you use NewRL in research, cite the software and record the Git revision
used for your experiments.

```bibtex
@misc{newrl,
  title = {NewRL},
  howpublished = {\url{https://github.com/LearningFromEmbodiedExperienceLab/newrl}},
  note = {Research software}
}
```
