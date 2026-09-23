#!/usr/bin/env python3
"""Launch helpers for the accepted Go2 SHAC/JAVE deployment policies.

This module binds the currently accepted training runs to the existing C++
Unitree Go2 deployment executable. It intentionally stays as a thin wrapper:
the real-time DDS control loop remains in deploy_cpp, while this script makes
the policy export/build/run commands repeatable.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class PolicyProfile:
    key: str
    label: str
    run_dir: Path
    checkpoint: str = "policy_best_tracking.pkl"
    variant: str = "blind_nolinvel_nokinref"

    @property
    def checkpoint_path(self) -> Path:
        return self.run_dir / self.checkpoint

    @property
    def deploy_path(self) -> Path:
        return self.run_dir / self.checkpoint.replace(".pkl", "_deploy.npz")


POLICIES = {
    "shac": PolicyProfile(
        key="shac",
        label="SHAC 20260820_002033",
        run_dir=Path("training_runs/shac_20260820_002033"),
    ),
    "jave": PolicyProfile(
        key="jave",
        label="JAVE 20260820_151021",
        run_dir=Path("training_runs/jave_20260820_151021"),
    ),
}


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _profile_keys(value: str) -> Iterable[str]:
    if value == "all":
        return POLICIES.keys()
    return [value]


def _require_file(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {_rel(path)}")


def export_policy(profile: PolicyProfile, force: bool = False) -> Path:
    checkpoint = REPO_ROOT / profile.checkpoint_path
    deploy_path = REPO_ROOT / profile.deploy_path
    _require_file(checkpoint, f"{profile.label} checkpoint")

    if (
        deploy_path.exists()
        and not force
        and deploy_path.stat().st_mtime >= checkpoint.stat().st_mtime
    ):
        print(f"{profile.label}: export already up to date: {_rel(deploy_path)}")
        return deploy_path

    old_cwd = Path.cwd()
    os.chdir(REPO_ROOT)
    try:
        from src.deploy.export_policy import export

        out = Path(export(_rel(checkpoint)))
    finally:
        os.chdir(old_cwd)

    return REPO_ROOT / out


def build_deploy(build_dir: Path, ros2: bool, jobs: int | None) -> None:
    build_dir = build_dir if build_dir.is_absolute() else REPO_ROOT / build_dir
    configure = [
        "cmake",
        "-S",
        ".",
        "-B",
        _rel(build_dir),
        f"-DOPEN_DIFFLOCO_ENABLE_ROS2={'ON' if ros2 else 'OFF'}",
    ]
    build = ["cmake", "--build", _rel(build_dir)]
    if jobs:
        build.extend(["-j", str(jobs)])

    print("Configuring deployment build:", " ".join(configure))
    subprocess.run(configure, cwd=REPO_ROOT, check=True)
    print("Building deployment target:", " ".join(build))
    subprocess.run(build, cwd=REPO_ROOT, check=True)


def executable_path(profile: PolicyProfile, build_dir: Path) -> Path:
    build_dir = build_dir if build_dir.is_absolute() else REPO_ROOT / build_dir
    return (
        build_dir
        / "src/envs/go2/variants"
        / profile.variant
        / "deploy_cpp"
        / f"deploy_{profile.variant}"
    )


def run_policy(args: argparse.Namespace) -> int:
    profile = POLICIES[args.policy]
    deploy_policy = export_policy(profile, force=args.export_force)
    build_dir = Path(
        args.build_dir
        if args.build_dir is not None
        else ("build-ros2" if args.command_source == "ros2" else "build")
    )

    exe = executable_path(profile, build_dir)
    if not exe.exists():
        if args.dry_run:
            print(f"Warning: executable is not built yet: {_rel(exe)}", file=sys.stderr)
        elif args.build_if_missing:
            build_deploy(build_dir, ros2=args.command_source == "ros2", jobs=args.jobs)
            if not exe.exists():
                print(
                    f"Build finished but executable is still missing: {_rel(exe)}",
                    file=sys.stderr,
                )
                return 2
        else:
            print(f"Missing executable: {_rel(exe)}", file=sys.stderr)
            ros2_flag = "--ros2 " if args.command_source == "ros2" else ""
            print(
                "Build it first, for example: python -m "
                "src.deploy.accepted_go2_deploy "
                f"build {ros2_flag}--build-dir {_rel(build_dir)}",
                file=sys.stderr,
            )
            return 2

    cmd = [
        str(exe),
        "--policy",
        _rel(deploy_policy),
        "--interface",
        args.interface,
        "--domain-id",
        str(args.domain_id),
        "--command-source",
        args.command_source,
    ]
    if args.command_source == "ros2":
        cmd.extend(["--cmd-topic", args.cmd_topic])
    if args.kp is not None:
        cmd.extend(["--kp", str(args.kp)])
    if args.kd is not None:
        cmd.extend(["--kd", str(args.kd)])

    print(f"Policy: {profile.label}")
    print("Command:")
    command_text = (" \\" + "\n  ").join(cmd)
    print("  " + command_text)
    print(
        "Controls: Enter=stand, Enter=walk, w/s=vx, a/d=vy, q/e=yaw, "
        "0=zero, x=estop, Ctrl-C=sit down"
    )
    if args.dry_run:
        return 0
    return subprocess.run(cmd, cwd=REPO_ROOT).returncode


def list_policies(_: argparse.Namespace) -> int:
    for profile in POLICIES.values():
        print(f"{profile.key}: {profile.label}")
        print(f"  checkpoint: {_rel(REPO_ROOT / profile.checkpoint_path)}")
        print(f"  deploy npz: {_rel(REPO_ROOT / profile.deploy_path)}")
        print(f"  variant:    {profile.variant}")
    return 0


def export_cmd(args: argparse.Namespace) -> int:
    for key in _profile_keys(args.policy):
        export_policy(POLICIES[key], force=args.force)
    return 0


def build_cmd(args: argparse.Namespace) -> int:
    build_deploy(Path(args.build_dir), ros2=args.ros2, jobs=args.jobs)
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export, build, and launch accepted Go2 SHAC/JAVE deployment policies."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="List accepted policy profiles")
    p_list.set_defaults(func=list_policies)

    p_export = sub.add_parser(
        "export", help="Export policy_best_tracking.pkl to deployment .npz"
    )
    p_export.add_argument("policy", choices=["all", *POLICIES.keys()])
    p_export.add_argument(
        "--force", action="store_true", help="Re-export even if the .npz is newer"
    )
    p_export.set_defaults(func=export_cmd)

    p_build = sub.add_parser(
        "build", help="Configure and build the C++ deployment executable"
    )
    p_build.add_argument(
        "--ros2", action="store_true", help="Enable ROS2 command-topic support"
    )
    p_build.add_argument("--build-dir", default="build", help="CMake build directory")
    p_build.add_argument(
        "--jobs", type=int, default=os.cpu_count(), help="Parallel build jobs"
    )
    p_build.set_defaults(func=build_cmd)

    p_run = sub.add_parser("run", help="Run one accepted policy on simulator or real Go2")
    p_run.add_argument("policy", choices=POLICIES.keys())
    p_run.add_argument(
        "--interface", default="eth0", help="DDS network interface; use lo for simulator"
    )
    p_run.add_argument("--domain-id", type=int, default=0, help="DDS domain ID")
    p_run.add_argument(
        "--command-source",
        choices=["terminal", "wireless", "ros2"],
        default="terminal",
        help="Velocity command source for the C++ controller",
    )
    p_run.add_argument("--cmd-topic", default="/velocity_command", help="ROS2 PointStamped topic")
    p_run.add_argument(
        "--build-dir", help="CMake build directory; defaults to build or build-ros2"
    )
    p_run.add_argument(
        "--build-if-missing",
        action="store_true",
        help="Run CMake if the executable is missing",
    )
    p_run.add_argument(
        "--jobs", type=int, default=os.cpu_count(), help="Parallel build jobs"
    )
    p_run.add_argument(
        "--export-force", action="store_true", help="Re-export policy before running"
    )
    p_run.add_argument("--kp", type=float, help="Override walking kp")
    p_run.add_argument("--kd", type=float, help="Override walking kd")
    p_run.add_argument(
        "--dry-run",
        action="store_true",
        help="Print command without launching the robot controller",
    )
    p_run.set_defaults(func=run_policy)

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
