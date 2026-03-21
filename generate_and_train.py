import argparse, json, sys, subprocess, os
from pipeline_generate import (
    build_candidate_standalone,
    resolve_task_template,
    parse_task_arg,
)
from registration import stable_env_id


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
    p.add_argument('--force', action='store_true')
    # everything after “--” is passed directly to ShadowHand_TQC.py
    args, train_args = p.parse_known_args()
    if train_args and train_args[0] == "--":
        train_args = train_args[1:]

    task_cfg = parse_task_arg(args.task)
    
    # -----------------------------
    # Inject per-task env defaults
    # -----------------------------
    # Helpers to detect if user already specified these flags downstream
    def has_flag(flag: str, argv: list[str]) -> bool:
        return flag in argv

    def has_opt(opt: str, argv: list[str]) -> bool:
        # matches --opt <value> or --opt=<value>
        if opt in argv:
            return True
        return any(a.startswith(opt + "=") for a in argv)

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
    paths = build_candidate_standalone(
        task=task_cfg["task_label"],
        Ntotal=args.Ntotal, Rppx=args.Rppx, Rpt=args.Rpt,
        Ap=args.Ap, Apx=args.Apx, At=args.At, Ap1=args.Ap1, Ap2=args.Ap2,
        base_xml=args.base, template_xml=tmpl,
        out_root=args.out_root, force=args.force,
        custom_msh=task_cfg["custom_msh"],
    )

    xml_abs = os.path.abspath(paths["env"])  # <-- make it absolute
    env_id = stable_env_id(xml_abs)  

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
