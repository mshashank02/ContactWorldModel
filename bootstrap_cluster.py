import argparse
import os
import shlex
import subprocess
from pathlib import Path
from typing import Iterable

from study_common import load_cluster_config, resolve_repo_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap the distributed study environment on all configured hosts.")
    parser.add_argument("--cluster-config", default="cluster_hosts.yaml")
    parser.add_argument("--repo-source", default=None,
                        help="Local repo path to rsync to each host. Defaults to the current repo.")
    parser.add_argument("--repo-url", default=None,
                        help="Optional git URL to clone/pull instead of rsyncing the local repo.")
    parser.add_argument("--python-version", default="3.11")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--include-coordinator", action="store_true",
                        help="Also bootstrap the coordinator host entry.")
    return parser.parse_args()


def run_command(command, dry_run: bool = False) -> None:
    print("$", " ".join(shlex.quote(str(part)) for part in command))
    if dry_run:
        return
    subprocess.run(command, check=True)


def remote_shell(ssh_target: str, command: str, dry_run: bool = False) -> None:
    run_command(["ssh", ssh_target, command], dry_run=dry_run)


def ensure_miniconda(host_cfg, python_version: str, dry_run: bool) -> None:
    conda_root = os.path.join(host_cfg.work_root, "miniconda3")
    installer = os.path.join(host_cfg.work_root, "Miniconda3-latest-Linux-x86_64.sh")
    command = (
        f"mkdir -p {shlex.quote(host_cfg.work_root)} && "
        f"if [ ! -x {shlex.quote(os.path.join(conda_root, 'bin', 'conda'))} ]; then "
        f"(command -v curl >/dev/null 2>&1 && curl -L -o {shlex.quote(installer)} https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh "
        f"|| wget -O {shlex.quote(installer)} https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh) && "
        f"bash {shlex.quote(installer)} -b -p {shlex.quote(conda_root)}; "
        f"fi && "
        f"if [ ! -x {shlex.quote(host_cfg.python_bin)} ]; then "
        f"{shlex.quote(os.path.join(conda_root, 'bin', 'conda'))} create -y -p {shlex.quote(host_cfg.python_root)} python={shlex.quote(python_version)}; "
        f"fi"
    )
    remote_shell(host_cfg.ssh_target, command, dry_run=dry_run)


def sync_repo(host_cfg, repo_source: Path, dry_run: bool) -> None:
    remote_parent = os.path.dirname(host_cfg.repo_path)
    remote_shell(host_cfg.ssh_target, f"mkdir -p {shlex.quote(remote_parent)}", dry_run=dry_run)
    command = [
        "rsync",
        "-az",
        "--delete",
        "--exclude=.git",
        "--exclude=__pycache__",
        "--exclude=generated",
        "--exclude=models",
        "--exclude=runs",
        "--exclude=videos",
        "--exclude=wandb",
        f"{str(repo_source)}/",
        f"{host_cfg.ssh_target}:{host_cfg.repo_path}/",
    ]
    run_command(command, dry_run=dry_run)


def clone_or_update_repo(host_cfg, repo_url: str, dry_run: bool) -> None:
    command = (
        f"mkdir -p {shlex.quote(os.path.dirname(host_cfg.repo_path))} && "
        f"if [ -d {shlex.quote(os.path.join(host_cfg.repo_path, '.git'))} ]; then "
        f"git -C {shlex.quote(host_cfg.repo_path)} pull --ff-only; "
        f"else git clone {shlex.quote(repo_url)} {shlex.quote(host_cfg.repo_path)}; fi"
    )
    remote_shell(host_cfg.ssh_target, command, dry_run=dry_run)


def install_repo_deps(host_cfg, dry_run: bool) -> None:
    command = (
        f"cd {shlex.quote(host_cfg.repo_path)} && "
        f"PYTHON_BIN={shlex.quote(host_cfg.python_bin)} bash install.sh"
    )
    remote_shell(host_cfg.ssh_target, command, dry_run=dry_run)


def target_hosts(cluster_cfg, include_coordinator: bool):
    hosts = list(cluster_cfg.hosts)
    if include_coordinator:
        hosts = [cluster_cfg.coordinator, *hosts]
    deduped = []
    seen = set()
    for host in hosts:
        if host.host in seen:
            continue
        seen.add(host.host)
        deduped.append(host)
    return deduped


def main() -> None:
    args = parse_args()
    repo_root = resolve_repo_root()
    repo_source = Path(args.repo_source).expanduser().resolve() if args.repo_source else repo_root
    cluster_cfg = load_cluster_config(args.cluster_config, repo_dirname=repo_root.name)

    for host_cfg in target_hosts(cluster_cfg, args.include_coordinator):
        ensure_miniconda(host_cfg, args.python_version, args.dry_run)
        if args.repo_url:
            clone_or_update_repo(host_cfg, args.repo_url, args.dry_run)
        else:
            sync_repo(host_cfg, repo_source, args.dry_run)
        install_repo_deps(host_cfg, args.dry_run)


if __name__ == "__main__":
    main()
