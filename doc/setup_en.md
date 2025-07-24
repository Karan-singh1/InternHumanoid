<p align="center">
  <img src="../doc/InternHumaniod.png" alt="InternHumanoid Logo" width="50%">
</p>

# 🛠️ Installation Guide

A step-by-step guide to set up InternHumanoid for simulation and reinforcement learning on humanoid robots.

---

## 📋 Table of Contents

- [System Requirements](#system-requirements)
- [Clone the Repository](#1-clone-the-repository)
- [Create a Virtual Environment](#2-create-a-virtual-environment-and-install-basic-dependencies)
  - [Install PyTorch](#23-install-pytorch)
  - [Install Isaac Gym](#24-install-isaac-gym)
  - [Install This Repository](#25-install-this-repository)
- [Install RL Framework & Environments](#26-install-reinforcement-learning-framework-and-simulation-environments)

---

## System Requirements

- **Operating System**: Recommended Ubuntu 18.04 or later  
- **GPU**: Nvidia GPU, Recommended RTX 4060ti or better
- **Driver Version**: Recommended version 525 or later  

---

## 1. Clone the Repository

Clone this repository from Github before installation:

```bash
git clone git@github.com:InternRobotics/InternHumanoid.git
```

---

## 2. Create a Virtual Environment and Install Basic Dependencies

We recommend using Anaconda to manage your Python environment.

### 2.1 Create a New Environment and Activate It

```bash
conda create -n rlgpu python=3.8
conda activate rlgpu
```

### 2.2 Install PyTorch

PyTorch is used for model training and inference:

```bash
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.1 -c pytorch -c nvidia
```

### 2.3 Install Isaac Gym

Isaac Gym is a high-performance GPU-based robot simulator from Nvidia.

#### 2.3.1 Download

Download [Isaac Gym](https://developer.nvidia.com/isaac-gym) from Nvidia’s official website.

#### 2.3.2 Install

After extracting the package, navigate to the `isaacgym/python` folder and install:

```bash
cd isaacgym/python
pip install -e .
cd ../..
```

---

### 2.4 Install This Repository

Navigate to the repository root:

```bash
cd InternHumanoid
```

---

### 2.5 Install Reinforcement Learning Framework and Simulation Environments

#### 2.5.1 Install RL Framework (`rsl_rl`)

```bash
cd rsl_rl
pip install -e .
cd ..
```

#### 2.5.2 Install Simulation Environments (`legged_gym`)

```bash
pip install -e .
```

---
