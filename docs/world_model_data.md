# World-model tactile data collection

The collection pipeline records two deliberately separate tactile streams:

- `policy_touch` is the sparse, layout-specific MuJoCo touch-sensor stream. It
  is ordered exactly as the active model's touch sensors and remains part of
  the ordinary policy observation.
- `oracle_touch` is dense privileged supervision over a fixed canonical set of
  candidate volumes. It is read by the recorder only and is never passed to
  TQC.

## Why the oracle path does not change physics

The active generated XML is loaded without modification. `OracleTouchWrapper`
queries MuJoCo's existing contacts after each physics step, transforms each
canonical sensor volume using the current pose of its hand body, and assigns a
real hand-object contact when its reported contact position lies in that
volume. It adds no geoms, contacts, forces, actuators, or sensors to the active
model.

For each canonical site the recorder stores contact activity, normal force,
two tangential force components in MuJoCo's contact frame, mean contact
position, contact normal, contacting geom id when one exists, contributing
contact count, and the current world position of the site. Missing contact
positions/normals and unavailable geom IDs (notably flex contacts) remain
`NaN`/`-1`; they are not fabricated.

The canonical site XML must be a true superset of every generated layout in a
collection job. Collection fails early if an active generated policy site is
absent. The repository's 90-site files can be used for layouts contained by
that set:

```text
--oracle-site-xml assets/Sensors_withPos.xml
--oracle-sensor-xml assets/shared_touch_sensors_90_1_1.xml
```

For layouts with sites beyond that set, generate one fixed, denser canonical
layout and use its `Sensors_withPos_*.xml` and `shared_touch_sensors_*.xml`
files for every compared policy layout. The canonical XML is parsed as
metadata only; it is never loaded into the policy simulator.

## Collect a random-policy smoke run

Run this in the same Python environment used for training:

```bash
python pipeline_generate.py --standalone \
  --base assets/hand_base.xml \
  --task block \
  --Ntotal 16 --Rppx 0.4 --Rpt 0.3 \
  --out-root generated_world_model_smoke --force

python scripts/collect_world_model_data.py \
  --xml-path generated_world_model_smoke/block_16_0.4_0.3/manipulate_block_touch_sensors_16_0.4_0.3.xml \
  --oracle-site-xml assets/Sensors_withPos.xml \
  --oracle-sensor-xml assets/shared_touch_sensors_90_1_1.xml \
  --dataset-root datasets/world_model_smoke \
  --object-id block \
  --candidate-id block_16_0.4_0.3 \
  --physics-mode rigid \
  --policy-stage random \
  --episodes 1 \
  --seed 2026 \
  --validate-after
```

## Collect a TQC checkpoint

Use the matching VecNormalize statistics whenever training used
`VecNormalize`. A sibling `vecnorm_<step>.pkl` is detected automatically for
standard checkpoint names.

```bash
python scripts/collect_world_model_data.py \
  --xml-path /path/to/generated/environment.xml \
  --oracle-site-xml /path/to/fixed/oracle/Sensors_withPos.xml \
  --oracle-sensor-xml /path/to/fixed/oracle/shared_touch_sensors.xml \
  --dataset-root datasets/world_model_pilot \
  --object-id obj_size-small_ar-low_macro-high_rough-low \
  --candidate-id n0050_a0p5_b0p5 \
  --physics-mode rigid \
  --policy-stage 2M \
  --policy-checkpoint-step 2000000 \
  --checkpoint /path/to/model_2000000_steps.zip \
  --vecnormalize /path/to/vecnorm_2000000.pkl \
  --episodes 100 \
  --seed 2026
```

Checkpoint inference is deterministic by default. Pass `--stochastic-policy`
to sample the learned policy.

## Pilot matrix

[`configs/world_model_pilot.example.json`](../configs/world_model_pilot.example.json)
is a compact template for the requested 6 objects × 8 layouts × 5 stages
(random, 0.5M, 2M, 8M, and 16M), with 100 episodes per combination. The six
placeholder objects are stratified across physics mode, size, aspect ratio,
roughness, and macro geometry. Replace their IDs and the path templates with
real artifacts, then run:

```bash
python scripts/collect_world_model_data.py \
  --config configs/world_model_pilot.json
```

The flat `jobs` config form is also supported when paths do not follow a
template. Every job may set its own action conditioning, episode length,
oracle contact margin, seed, object metadata, and checkpoint metadata.

Eventful episodes receive a larger `sampling_weight` based on contact onsets,
releases, slips, regrasps, drops, and success transitions. World-model loaders
can use that value for weighted oversampling without duplicating identical
trajectories.

## Dataset layout

```text
dataset_root/
  runs/<run_id>/
    run_manifest.json
    sensor_manifest.json
    object_manifest.json
    episodes/episode_000000.npz
  index/
    episodes.parquet
    runs.parquet
    objects.parquet
    layouts.parquet
    checkpoints.parquet
    summary.json
    validation_report.json
```

Each shard contains timestep-aligned actions, observations, next observations,
rewards, termination/truncation flags, success, and derived event arrays.
Slash-separated logical names are stored in NPZ with `__` separators, for
example `next_observations__oracle_touch__normal_force`.

`sensor_count_total`, `sensor_count_palm`, `sensor_count_tip`, and
`sensor_count_non_tip` describe the active policy layout.
`oracle_sensor_count` is the dense canonical axis length; `sensor_mask` maps
the active layout into that axis.

## Index and validate

```bash
python scripts/build_world_model_index.py datasets/world_model_pilot
python scripts/validate_world_model_dataset.py datasets/world_model_pilot
```

Validation checks required arrays and metadata, leading dimensions, dtypes,
dense sensor counts, manifest paths, corrupt shards, and coverage. Empty-contact
episodes are warnings by default and can be made fatal with
`--empty-contact-is-error`. The command returns a nonzero status when the
dataset is not ready for training.
