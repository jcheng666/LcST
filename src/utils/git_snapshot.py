import json

import os
import shlex
import shutil
import subprocess
import sys
from argparse import Namespace
from typing import Any


def _git(*args: str, cwd: str | None = None) -> tuple[bool, str]:
    r = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    return r.returncode == 0, r.stdout.rstrip("\n")


def _copy_src(dest: str) -> None:
    src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    shutil.copytree(
        src_dir,
        dest,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )


def snapshot_args(log_dir: str, args: Namespace) -> None:
    """Record parsed experiment args as JSON for machine-readable replay."""
    snap_dir = os.path.join(log_dir, "snapshot")
    os.makedirs(snap_dir, exist_ok=True)

    args_dict: dict[str, Any] = vars(args)
    with open(os.path.join(snap_dir, "args.json"), "w") as f:
        json.dump(args_dict, f, indent=2, sort_keys=True, default=str)
        f.write("\n")


def snapshot_git(log_dir: str) -> None:
    """Record HEAD hash, diff, and status into snapshot dir."""
    snap_dir = os.path.join(log_dir, "snapshot")
    os.makedirs(snap_dir, exist_ok=True)

    ok, repo_root = _git("rev-parse", "--show-toplevel")
    if not ok:
        _copy_src(os.path.join(snap_dir, "src"))
        with open(os.path.join(snap_dir, "HEAD"), "w") as f:
            f.write("<no git>\n")
        return

    _, head = _git("rev-parse", "HEAD", cwd=repo_root)
    _, diff = _git("diff", "HEAD", cwd=repo_root)
    _, status = _git("status", "--short", cwd=repo_root)

    with open(os.path.join(snap_dir, "HEAD"), "w") as f:
        f.write(head + "\n")

    if diff:
        with open(os.path.join(snap_dir, "diff.patch"), "w") as f:
            f.write(diff + "\n")

    if status:
        with open(os.path.join(snap_dir, "status"), "w") as f:
            f.write(status + "\n")


def snapshot_command(log_dir: str) -> None:
    """Record the exact Python command used to start the run."""
    snap_dir = os.path.join(log_dir, "snapshot")
    os.makedirs(snap_dir, exist_ok=True)
    command = shlex.join([sys.executable, *sys.argv])
    with open(os.path.join(snap_dir, "command.txt"), "w") as f:
        f.write(command + "\n")


def snapshot_experiment(log_dir: str, args: Namespace) -> None:
    """Record args and git state required to reproduce an experiment run."""
    snapshot_args(log_dir, args)
    snapshot_git(log_dir)
    _copy_src(os.path.join(log_dir, "snapshot", "src"))
    snapshot_command(log_dir)
