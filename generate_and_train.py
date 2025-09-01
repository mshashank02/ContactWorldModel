import argparse, json, sys, subprocess, os
from pipeline_generate_and_plug_in import build_candidate_standalone, resolve_task_template
from registration import stable_env_id


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', required=True)
    p.add_argument('--task', choices=['block','egg','pen'], default='block')
    p.add_argument('--Ntotal', type=int, required=True)
    p.add_argument('--Rppx', type=float, required=True)
    p.add_argument('--Rpt',  type=float, required=True)
    p.add_argument('--Ap',   type=float, required=True)
    p.add_argument('--Apx',  type=float, required=True)
    p.add_argument('--At',   type=float, required=True)
    p.add_argument('--Ap1',  type=float, required=True)
    p.add_argument('--Ap2',  type=float, required=True)
    p.add_argument('--out-root', default='generated')
    p.add_argument('--force', action='store_true')
    # everything after “--” is passed directly to ShadowHand_TQC.py
    args, train_args = p.parse_known_args()
    if train_args and train_args[0] == "--":
        train_args = train_args[1:]
    tmpl = resolve_task_template(args.task, None, None)
    paths = build_candidate_standalone(
        task=args.task,
        Ntotal=args.Ntotal, Rppx=args.Rppx, Rpt=args.Rpt,
        Ap=args.Ap, Apx=args.Apx, At=args.At, Ap1=args.Ap1, Ap2=args.Ap2,
        base_xml=args.base, template_xml=tmpl,
        out_root=args.out_root, force=args.force
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
