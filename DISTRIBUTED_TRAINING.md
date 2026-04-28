# Distributed Training Runbook

This runbook covers the distributed GPBO workflow in this repo for object studies, with an emphasis on previewing the initial Sobol candidates before any training jobs are dispatched.

## What the distributed pipeline does

- `optimize_dataset_gpbo.py` acts as the coordinator.
- `study_worker.py` runs on each worker host and claims jobs from the coordinator.
- `study_queue.py` stores candidate and job state in `generated/studies/<study_name>/study.db`.
- The first candidates come from Sobol sampling.
- Later candidates come from BO using completed results.

Important behavior:

- The coordinator only enqueues one candidate at a time when there are no active candidates.
- Workers only start training after jobs have been enqueued and the worker processes are running.

## Before you start

Make sure these are ready:

- `cluster_hosts.yaml` has the correct hosts and `num_envs_per_job` values.
- Your object manifest exists, for example:
  `study_objects/sphere_study_v1/manifest.csv`
- Remote machines have SSH access.
- The environment on each remote host has been bootstrapped.

Important for the current sphere study:

- `study_objects/sphere_study_v1/manifest.csv` currently contains `24` rows across `4` base objects.
- For this manifest, pass `--expected-base-objects 4`.
- Each enqueued candidate will create `48` jobs total:
  `24 objects x 2 physics modes`

## Step 1: Check cluster sizing

Review [cluster_hosts.yaml](/home/mshashank02/ShadowHand-TQC/cluster_hosts.yaml) and verify:

- `gpu_count`
- `cpu_cores`
- `num_envs_per_job`
- `python_root`
- `work_root`

The worker launcher will inject `--num-envs` from this file unless you explicitly override it in trainer args.

## Step 2: Bootstrap the remote machines

Run this once from the repo root:

```bash
python bootstrap_cluster.py --include-coordinator
```

If you want to see the commands first:

```bash
python bootstrap_cluster.py --include-coordinator --dry-run
```

## Step 3: Initialize the study and preview the initial Sobol candidates

This is the safest first command. It creates the study DB, registers the candidate grid, chooses the initial Sobol points, writes reports, and exits before enqueuing any jobs.

Example:

```bash
python optimize_dataset_gpbo.py \
  --study-name sphere_v1_demo \
  --objects-root study_objects/sphere_study_v1 \
  --cluster-config cluster_hosts.yaml \
  --init-candidates 4 \
  --bo-candidates 4 \
  --expected-base-objects 4 \
  --preview-initial-only
```

What this gives you:

- Prints the chosen initial Sobol candidates to the terminal.
- Writes the study DB to:
  `generated/studies/sphere_v1_demo/study.db`
- Writes the study spec to:
  `generated/studies/sphere_v1_demo/study_spec.json`
- Writes the initial candidate list to:
  `generated/studies/sphere_v1_demo/initial_candidates.json`

At this point, no jobs have been enqueued and no worker can train anything yet.

## Step 4: Inspect the initial Sobol candidates again if needed

You can inspect the saved file directly:

```bash
cat generated/studies/sphere_v1_demo/initial_candidates.json
```

Or inspect candidates via the queue helper:

```bash
python study_queue.py \
  --db generated/studies/sphere_v1_demo/study.db \
  list-candidates
```

At this stage, the initial Sobol candidates should still be in status `new`.

## Step 5: Enqueue the first candidate, but do not start workers yet

Run one coordinator iteration:

```bash
python optimize_dataset_gpbo.py \
  --study-name sphere_v1_demo \
  --objects-root study_objects/sphere_study_v1 \
  --cluster-config cluster_hosts.yaml \
  --init-candidates 4 \
  --bo-candidates 4 \
  --expected-base-objects 4 \
  --once
```

This will enqueue exactly one candidate if none is active.

## Step 6: Inspect what was queued before any training starts

List candidates:

```bash
python study_queue.py \
  --db generated/studies/sphere_v1_demo/study.db \
  list-candidates
```

List only queued candidates:

```bash
python study_queue.py \
  --db generated/studies/sphere_v1_demo/study.db \
  list-candidates --statuses queued
```

List the actual per-object jobs created for that candidate:

```bash
python study_queue.py \
  --db generated/studies/sphere_v1_demo/study.db \
  list-jobs --statuses pending
```

This is the point where you can verify:

- which `candidate_id` was selected
- its `N`, `alpha`, and `beta`
- which objects it will run on
- both `deformable` and `rigid` jobs
- artifact output paths

No training starts until a worker claims these pending jobs.

## Step 7: Start the coordinator loop

Once the queued candidate looks correct, start the coordinator in long-running mode:

```bash
python optimize_dataset_gpbo.py \
  --study-name sphere_v1_demo \
  --objects-root study_objects/sphere_study_v1 \
  --cluster-config cluster_hosts.yaml \
  --expected-base-objects 4
```

This process:

- requeues stale jobs if needed
- waits for the active candidate to finish
- enqueues the next Sobol or BO candidate
- exports reports each loop

## Step 8: Start workers on the remote hosts

Run one worker process per host. Example:

```bash
python study_worker.py --study-name sphere_v1_demo --host-name lara156
```

Repeat on each configured worker host:

- `lara156`
- `lara114`
- `lara98`
- `lara83`
- optionally the coordinator host if `run_worker: true`

The worker will:

- detect free GPUs
- claim pending jobs from the coordinator
- launch training with that host's configured `--num-envs`

## Step 9: Monitor progress

Check high-level summary:

```bash
python study_queue.py \
  --db generated/studies/sphere_v1_demo/study.db \
  summary
```

Inspect candidates:

```bash
python study_queue.py \
  --db generated/studies/sphere_v1_demo/study.db \
  list-candidates
```

Inspect running jobs:

```bash
python study_queue.py \
  --db generated/studies/sphere_v1_demo/study.db \
  list-jobs --statuses running
```

Inspect saved reports:

- `generated/studies/sphere_v1_demo/candidate_history.json`
- `generated/studies/sphere_v1_demo/leaderboard.csv`
- `generated/studies/sphere_v1_demo/per_condition_scores.csv`

## Recommended safe operating pattern

Use this order each time:

1. `--preview-initial-only`
2. inspect `initial_candidates.json`
3. `--once`
4. inspect `list-jobs --statuses pending`
5. start coordinator loop
6. start workers

This gives you visibility both before the first candidate is enqueued and again before any worker begins training.

## What To Verify

Before starting workers, check these things:

- `initial_candidates.json` contains the Sobol points you expect.
- `list-candidates --statuses queued` shows exactly one queued candidate after `--once`.
- `list-jobs --statuses pending` shows the full per-object expansion for that candidate.
- For `study_objects/sphere_study_v1`, expect `48` pending jobs per candidate:
  `24 deformable` and `24 rigid`.

## Example end-to-end command sequence

```bash
python bootstrap_cluster.py --include-coordinator
```

```bash
python optimize_dataset_gpbo.py \
  --study-name sphere_v1_demo \
  --objects-root study_objects/sphere_study_v1 \
  --cluster-config cluster_hosts.yaml \
  --init-candidates 4 \
  --bo-candidates 4 \
  --expected-base-objects 4 \
  --preview-initial-only
```

```bash
python optimize_dataset_gpbo.py \
  --study-name sphere_v1_demo \
  --objects-root study_objects/sphere_study_v1 \
  --cluster-config cluster_hosts.yaml \
  --init-candidates 4 \
  --bo-candidates 4 \
  --expected-base-objects 4 \
  --once
```

```bash
python study_queue.py \
  --db generated/studies/sphere_v1_demo/study.db \
  list-jobs --statuses pending
```

```bash
python optimize_dataset_gpbo.py \
  --study-name sphere_v1_demo \
  --objects-root study_objects/sphere_study_v1 \
  --cluster-config cluster_hosts.yaml \
  --expected-base-objects 4
```

```bash
python study_worker.py --study-name sphere_v1_demo --host-name lara156
```

## Notes

- `--preview-initial-only` does not enqueue jobs.
- `--once` may enqueue one candidate if there are no active candidates.
- Workers should be started only after you have inspected the queued jobs if you want full visibility before training starts.
- If the manifest changes, update `--expected-base-objects` to match the actual number of unique `base_object` values in `manifest.csv`.
