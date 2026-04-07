import argparse
import os
import shlex
import subprocess
from pathlib import Path

from study_common import load_cluster_config, resolve_repo_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync study meshes and manifest to all worker hosts.")
    parser.add_argument("--cluster-config", default="cluster_hosts.yaml")
    parser.add_argument("--objects-root", required=True, help="Path to the study object directory inside the repo.")
    parser.add_argument("--include-coordinator", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run(command, dry_run: bool) -> None:
    print("$", " ".join(shlex.quote(str(part)) for part in command))
    if dry_run:
        return
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    repo_root = resolve_repo_root()
    objects_root = Path(args.objects_root).expanduser().resolve()
    if not objects_root.is_dir():
        raise FileNotFoundError(objects_root)
    relpath = os.path.relpath(objects_root, repo_root)
    if relpath.startswith(".."):
        raise ValueError("objects-root must live inside this repo so workers can use the same relative path.")

    cluster_cfg = load_cluster_config(args.cluster_config, repo_dirname=repo_root.name)
    hosts = list(cluster_cfg.hosts)
    if args.include_coordinator:
        hosts = [cluster_cfg.coordinator, *hosts]
    deduped = []
    seen = set()
    for host in hosts:
        if host.host in seen:
            continue
        seen.add(host.host)
        deduped.append(host)
    hosts = deduped

    for host_cfg in hosts:
        remote_dir = os.path.join(host_cfg.repo_path, relpath)
        run(["ssh", host_cfg.ssh_target, f"mkdir -p {shlex.quote(remote_dir)}"], args.dry_run)
        run(
            [
                "rsync",
                "-az",
                f"{str(objects_root)}/",
                f"{host_cfg.ssh_target}:{remote_dir}/",
            ],
            args.dry_run,
        )


if __name__ == "__main__":
    main()
