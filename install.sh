#!/usr/bin/env bash
set -euo pipefail

# Install all required packages for this repo, including the validated
# MuJoCo/Gymnasium-Robotics compatibility pair used for custom deformable XMLs.

PYTHON_BIN="${PYTHON_BIN:-python}"

"$PYTHON_BIN" -m pip install --upgrade pip wheel setuptools

"$PYTHON_BIN" -m pip install \
  "mujoco==3.3.1" \
  "gymnasium==1.2.0" \
  "gymnasium-robotics==1.4.2" \
  "stable-baselines3==2.7.1" \
  "sb3-contrib==2.7.1" \
  "wandb==0.25.1" \
  "protobuf>=6.31.1,<7" \
  "moviepy==2.2.1" \
  "imageio" \
  "tensorboard" \
  "botorch" \
  "pyyaml"

echo "[OK] Installation complete."
echo "[INFO] Installed compatibility core: mujoco==3.3.1, gymnasium==1.2.0, gymnasium-robotics==1.4.2"
