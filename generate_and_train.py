import argparse, os, re, subprocess, sys
from pipeline_generate import (
    build_candidate_standalone,
    resolve_task_template,
    parse_task_arg,
)
from registration import stable_env_id


def sanitize_label(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("_") or "run"


def has_flag(flag: str, argv: list[str]) -> bool:
    return flag in argv


def has_opt(opt: str, argv: list[str]) -> bool:
    if opt in argv:
        return True
    return any(arg.startswith(opt + "=") for arg in argv)


SIZE_SCALE_MULTIPLIERS = {
    "small": 0.75,
    "medium": 1.0,
    "large": 1.25,
}


def scale_triplet(base_value: float, multiplier: float) -> str:
    scaled = base_value * multiplier
    return f"{scaled:.6f} {scaled:.6f} {scaled:.6f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', required=True)
    p.add_argument(
        '--task',
        default='block',
        help="Built-in task name (block/egg/pen) OR absolute path to a custom .msh file.",
    )
    p.add_argument("--Ntotal", type=int, required=True)
    p.add_argument('--Rppx', type=float, required=True)
    p.add_argument('--Rpt',  type=float, required=True)
    p.add_argument("--Ap",   type=float, default= 6557, help="Area weight: Palm")
    p.add_argument("--Apx",  type=float, default=26885, help="Area weight: Phalanx")
    p.add_argument("--At",   type=float, default=7193, help="Area weight: Tips")
    p.add_argument("--Ap1",  type=float, default=5557, help="Palm sub-area 1 (palm)")
    p.add_argument("--Ap2",  type=float, default=1000, help="Palm sub-area 2 (lfmetacarpal)")
    p.add_argument('--out-root', default='generated')
    p.add_argument('--artifact-root', default=None,
                   help="Optional run-specific root for generated XMLs and training artifacts.")
    p.add_argument('--object-id', default=None,
                   help="Stable object identifier used for artifact naming and metrics metadata.")
    p.add_argument('--run-label', default=None,
                   help="Stable run label used for env_id and logging.")
    p.add_argument('--candidate-id', default=None,
                   help="Optional candidate identifier forwarded into metrics metadata.")
    p.add_argument('--physics-mode', choices=["rigid", "deformable"], default=None,
                   help="Optional physics mode metadata. Defaults to deformable when --deformable is set, else rigid.")
    p.add_argument('--object-size', choices=sorted(SIZE_SCALE_MULTIPLIERS), default=None,
                   help="Optional object size label used to scale custom .msh objects in generated XMLs.")
    p.add_argument('--deformable', action='store_true',
                   help="Generate a deformable custom object when --task points to a .msh file.")
    p.add_argument('--force', action='store_true')
    # everything after “--” is passed directly to ShadowHand_TQC.py
    args, train_args = p.parse_known_args()
    if train_args and train_args[0] == "--":
        train_args = train_args[1:]

    task_cfg = parse_task_arg(args.task)
    physics_mode = args.physics_mode or ("deformable" if args.deformable else "rigid")
    out_root = os.path.abspath(args.artifact_root or args.out_root)
    size_multiplier = SIZE_SCALE_MULTIPLIERS.get(args.object_size or "medium", 1.0)
    flex_scale = scale_triplet(0.025, size_multiplier)

    if args.object_id and task_cfg["custom_msh"] is not None:
        task_cfg["task_label"] = f"custom_{sanitize_label(args.object_id)}"

    # -----------------------------
    # Inject per-task env defaults
    # -----------------------------
    # Desired defaults by task
    if task_cfg["template_task"] == "pen":
        desired_target_position = "ignore"   # no position goal
        desired_ignore_z = True              # XY-only rotation
    else:  # block, egg
        desired_target_position = "random"   # keep position goal
        desired_ignore_z = False             # full xyz

    # Only append if not already set by the caller
    if not has_opt("--target-position", train_args):
        train_args += ["--target-position", desired_target_position]

    # --ignore-z-rot is a boolean flag; only add it when desired and not present
    if desired_ignore_z and not has_flag("--ignore-z-rot", train_args):
        train_args += ["--ignore-z-rot"]
        
    tmpl = resolve_task_template(task_cfg["template_task"], None, None)
    custom_msh_name = None
    if task_cfg["custom_msh"] is not None and args.object_id:
        custom_msh_name = f"{sanitize_label(args.object_id)}_{os.path.basename(task_cfg['custom_msh'])}"
    paths = build_candidate_standalone(
        task=task_cfg["task_label"],
        Ntotal=args.Ntotal, Rppx=args.Rppx, Rpt=args.Rpt,
        Ap=args.Ap, Apx=args.Apx, At=args.At, Ap1=args.Ap1, Ap2=args.Ap2,
        base_xml=args.base, template_xml=tmpl,
        out_root=out_root, force=args.force,
        custom_msh=task_cfg["custom_msh"],
        custom_msh_name=custom_msh_name,
        deformable_object=args.deformable,
        flex_scale=flex_scale,
    )

    xml_abs = os.path.abspath(paths["env"])  # <-- make it absolute
    env_id = sanitize_label(args.run_label) if args.run_label else stable_env_id(xml_abs)

    if args.artifact_root and not has_opt("--artifact-root", train_args):
        train_args += ["--artifact-root", os.path.abspath(args.artifact_root)]
    if args.object_id and not has_opt("--object-id", train_args):
        train_args += ["--object-id", sanitize_label(args.object_id)]
    if args.candidate_id and not has_opt("--candidate-id", train_args):
        train_args += ["--candidate-id", sanitize_label(args.candidate_id)]
    if physics_mode and not has_opt("--physics-mode", train_args):
        train_args += ["--physics-mode", physics_mode]
    if args.run_label and not has_opt("--wandb-name", train_args):
        train_args += ["--wandb-name", sanitize_label(args.run_label)]

    #env_id = stable_env_id(paths["env"])           # optional: for naming only
    cmd = [
        "python", "ShadowHand_TQC.py",
        "--env-id", env_id,                        # just for logs/dirs
        "--xml-path", xml_abs,                # REQUIRED for direct construction
        *train_args,
    ]
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
