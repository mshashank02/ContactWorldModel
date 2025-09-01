# ShadowHand-TQC

Implementation of touch-sensor–based in-hand manipulation tasks for the Shadow Hand, trained with **Truncated Quantile Critics (TQC)** and **Hindsight Experience Replay (HER)**.

---

## Installation

```
conda create -n ShadowHand python=3.10
conda activate ShadowHand
pip install -r requirements.txt
```

---

## Training with built-in Mujoco environments

Rotate a block with 16 parallel environments:
```
python ShadowHand_TQC.py --env-id HandManipulateBlockRotateXYZ-v1 --seed 4 --num-envs 16
```

Train with continuous touch sensors:
```
python ShadowHand_TQC.py --env-id HandManipulateBlock_ContinuousTouchSensors-v1 --seed 4 --num-envs 16
```

---

## Run the BO-GP optimization problem 

`GPBO_sensor_optimize.py` runs the GP Bo optimization loop by calling generate_and_train.py to run simulations.

```
python bo_botorch.py \
  --base hand/manipulate_block_touch_sensors.xml \
  --tasks block,egg,pen \
  --Ap 1.0 --Apx 1.0 --At 1.0 --Ap1 1.0 --Ap2 1.0 \
  --N-min 40 --N-max 140 \
  --alpha-min 0.05 --alpha-max 0.95 \
  --beta-min 0.05 --beta-max 0.95 \
  --init 12 --iters 30 --q 6 --device cuda \
  --bo-root generated/bo_qnei_cuda_multitask \
  --seeds 0,1,2 \
  --trainer-args -- \
      --n-timesteps 300000 \
      --eval-freq 20000 \
      --eval-episodes 20 \
      --save-freq 100000 \
      --verbose 1
```
---

## Generate a custom touch-sensor configuration and train

`generate_and_train.py` builds a standalone MuJoCo XML with a specified number of touch sensors and launches training.  
Any flags after `--` are forwarded directly to `ShadowHand_TQC.py`.

```
python generate_and_train.py \
    --base assets/hand_base.xml \
    --task pen \
    --Ntotal 16 --Rppx 0.4 --Rpt 0.3 \
    --Ap 0.5 --Apx 0.3 --At 0.2 --Ap1 0.25 --Ap2 0.25 \
    --out-root generated --force -- \
    --num-envs 1 --seed 0 --learning-starts 1000
```

The helper script assigns a stable environment ID using the absolute XML path.

---

## Standalone asset generation only

To produce a standalone environment without launching training:

```
python pipeline_generate_and_plug_in.py --standalone \
    --base assets/hand_base.xml --task block \
    --Ntotal 16 --Rppx 0.4 --Rpt 0.3 \
    --Ap 0.5 --Apx 0.3 --At 0.2 --Ap1 0.25 --Ap2 0.25 \
    --out-root generated --force
```

The pipeline copies:
- Mesh assets referenced by `shared_asset.xml` into `stls/hand/`
- Texture assets referenced by the generated XML into `textures/`

It raises clear errors if any source file is missing.

---

## Project Structure

```
ShadowHand-TQC/
│── ShadowHand_TQC.py                # Main training entry point
│── generate_and_train.py             # Generate custom XMLs and launch training
│── pipeline_generate_and_plug_in.py  # Standalone XML and asset generation
│── assets/                           # Base hand XMLs and shared assets
│── generated/                        # Auto-generated sensor configs
│── stls/hand/                        # Mesh assets
│── textures/                         # Texture assets
│── requirements.txt                  # Python dependencies
```

---

## Features

- Shadow Hand manipulation with touch-sensor augmentation  
- Training with **TQC + HER** for sample efficiency  
- Flexible **sensor configuration pipeline** for exploring design spaces  
- Ready-to-use XMLs for block rotation and continuous-touch tasks  