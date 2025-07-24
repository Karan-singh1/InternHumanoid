<p align="center">
  <img src="../doc/InternHumaniod.png" alt="InternHumanoid Logo" width="50%">
</p>

# 🧩 Adding New Simulation Environments

A step-by-step guide for extending InternHumanoid with new simulation environments, robots, and tasks. This guide covers environment scripts, configuration, and modular YAML setup for rapid experimentation.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Basic Simulation Environment](#basic-simulation-environment)
- [How to Add a New Environment](#how-to-add-a-new-environment)
- [Modifying Configurations](#modifying-configurations)
  - [Task Config](#example-modifying-a-task-config)
  - [Robot Config](#example-modifying-a-robot-config)
  - [Algorithm Config](#example-modifying-an-algorithm-config)
  - [Terrain & Simulator Configs](#example-modifying-terrain-or-simulator-configs)
- [Tips: Config Composition](#tips-config-composition)

---

## 📝 Overview

InternHumanoid is designed for flexibility. You can easily add new robots, tasks, and environments by following a modular configuration approach. This document provides a concise workflow for extending the simulation suite.

---

## 1. Basic Simulation Environment

The base environment `legged_gym/envs/base/legged_robot` implements a rough terrain locomotion task. The corresponding task config is in `legged_gym/config/task/base/legged_robot.yaml`.

- No robot asset (URDF/MJCF) specified
- No reward scales by default
- Use this as a template for your own environments and configs

---

## 2. How to Add a New Environment

Follow these steps to add a new simulation environment:

1. **Create Environment Script**  
   Add a new Python file in `legged_gym/envs/<your_robot>/<your_task>.py`, following the template in `legged_gym/envs/base/legged_robot.py`.
2. **Register the Task**  
   Register your new environment in `legged_gym/envs/__init__.py`.
3. **Add Task Config**  
   Place your task YAML in `legged_gym/config/task/<your_robot>/<your_task>.yaml`.
4. **Add Robot Asset (if needed)**  
   If introducing a new robot, add its URDF or MJCF to `resources/robot/<your_robot>/`.

---

## 3. Modifying Configurations

The configuration system is modular and organized by function. Customize or extend your environment by editing/creating YAML files in these folders:

| Folder                  | Purpose / Example Configs                                                    |
|-------------------------|-------------------------------------------------------------------------------|
| `config/task/`          | Task-specific configs (e.g., imitation, flomo). Example: `imitation/g1_29dof.yaml` |
| `config/robot/`         | Robot-specific configs. Example: `g1/g1_29dof.yaml`                              |
| `config/terrain/`       | Terrain settings. Example: `plane.yaml`, `locomotion.yaml`                      |
| `config/algo/`          | RL algorithm settings. Example: `ppo.yaml`                                      |
| `config/simulator/`     | Simulator parameters. Example: `base.yaml`                                      |
| `config/base/`          | Base templates for inheritance. Example: `base.yaml`                            |

### Example: Modifying a Task Config

```yaml
task:
  name: g1_im
  seed: 0
  env:
    num_envs: 5120
    max_episode_length_sec: 20
  # ... more task-specific settings ...
```

### Example: Modifying a Robot Config

```yaml
robot:
  control:
    control_type: "P"
    stiffness: {"hip": 150, "knee": 200}
    # ... more control parameters ...
  asset:
    file: "./resources/robots/G1/urdf/g1_29dof.urdf"
    # ... more asset parameters ...
```

### Example: Modifying an Algorithm Config

```yaml
algo:
  seed: 1
  policy:
    actor_hidden_dims: [1024, 512, 256, 256]
    # ... more policy parameters ...
  algorithm:
    learning_rate: 1e-4
    # ... more algorithm parameters ...
```

### Example: Modifying Terrain or Simulator Configs

- `config/terrain/plane.yaml` for flat or rough terrain settings.
- `config/simulator/base.yaml` for simulation time step, gravity, and physics engine parameters.

---

## 💡 Tips: Config Composition

You can compose configs using the `defaults` key in your YAML files to inherit and override settings from base configs, e.g.:

```yaml
defaults:
  - /simulator/base
  - /terrain/plane
```

This modular approach allows you to easily experiment with different robots, tasks, terrains, and algorithms by mixing and matching config files.

---
