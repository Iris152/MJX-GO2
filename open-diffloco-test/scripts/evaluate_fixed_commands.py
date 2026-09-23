#!/usr/bin/env python3
"""Evaluate a Go2 policy on fixed velocity commands without rendering."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jp
import numpy as np

from src.core.data_structures import Normalizer
from src.core.networks import Actor, SAPOActor
from src.envs.go2.environment import Go2Env
from src.visualization.go2 import (
    _extract_local_vel,
    _load_env_kwargs,
    _make_get_action,
    _override_cmd,
)


COMMANDS = [
    ("stand", (0.0, 0.0, 0.0)),
    ("fwd_+1.0", (1.0, 0.0, 0.0)),
    ("fwd_+1.2", (1.2, 0.0, 0.0)),
    ("back_-0.6", (-0.6, 0.0, 0.0)),
    ("back_-1.0", (-1.0, 0.0, 0.0)),
    ("vy_+0.3", (0.0, 0.3, 0.0)),
    ("vy_-0.3", (0.0, -0.3, 0.0)),
    ("vy_+0.6", (0.0, 0.6, 0.0)),
    ("vy_-0.6", (0.0, -0.6, 0.0)),
    ("yaw_+0.8", (0.0, 0.0, 0.8)),
    ("yaw_-0.8", (0.0, 0.0, -0.8)),
]


def _mean_metric(samples: list[dict], key: str) -> float:
    vals = [s[key] for s in samples if key in s and np.isfinite(s[key])]
    return float(np.mean(vals)) if vals else float("nan")


def evaluate_policy(
    policy_path: str,
    steps: int,
    warmup: int,
    seed: int,
    commands: list[tuple[str, tuple[float, float, float]]],
) -> list[dict]:
    with open(policy_path, "rb") as f:
        state = pickle.load(f)

    env_kwargs = _load_env_kwargs(policy_path)
    env_kwargs = dict(env_kwargs)
    env_kwargs["max_episode_length"] = int(1e9)
    env = Go2Env(**env_kwargs)
    get_action = _make_get_action(env, policy_path)

    @jax.jit
    def step_env(env_state, action):
        return env.step(env_state, action)

    jit_override = jax.jit(_override_cmd)

    rng = jax.random.PRNGKey(seed)
    rows = []
    for name, cmd in commands:
        rng, reset_key = jax.random.split(rng)
        env_state = env.reset(reset_key, jp.array(0.0))
        cmd_arr = jp.array(cmd, dtype=jp.float64)
        samples = []
        prev_action = np.zeros(env.action_dim, dtype=np.float64)

        for step_idx in range(steps):
            env_state = jit_override(env_state, cmd_arr)
            action = get_action(state.actor_params, state.normalizer, env_state.obs)
            env_state = step_env(env_state, action)

            if step_idx >= warmup:
                local_linvel, local_angvel = _extract_local_vel(env_state)
                action_np = np.array(action, dtype=np.float64)
                joint_offsets = np.array(env_state.data.qpos[7:] - env.default_joints)
                samples.append(
                    {
                        "vx": float(local_linvel[0]),
                        "vy": float(local_linvel[1]),
                        "yaw": float(local_angvel[2]),
                        "contact": float(env_state.metrics["foot_contact_mean"]),
                        "slip": float(env_state.metrics["foot_slip_speed"]),
                        "clear": float(env_state.metrics["foot_clearance_mean"]),
                        "height": float(env_state.metrics["height"]),
                        "tilt": float(env_state.metrics["tilt"]),
                        "qdev_l2": float(np.linalg.norm(joint_offsets)),
                        "qmax": float(np.max(np.abs(joint_offsets))),
                        "action_abs": float(np.mean(np.abs(action_np))),
                        "action_rate_abs": float(np.mean(np.abs(action_np - prev_action))),
                        "done": float(env_state.done),
                    }
                )
                prev_action = action_np

        cmd_np = np.array(cmd, dtype=np.float64)
        row = {
            "policy": str(policy_path),
            "name": name,
            "cmd": list(cmd_np),
            "mean_vx": _mean_metric(samples, "vx"),
            "mean_vy": _mean_metric(samples, "vy"),
            "mean_yaw": _mean_metric(samples, "yaw"),
            "mae_vx": float(np.mean([abs(s["vx"] - cmd_np[0]) for s in samples])),
            "mae_vy": float(np.mean([abs(s["vy"] - cmd_np[1]) for s in samples])),
            "mae_yaw": float(np.mean([abs(s["yaw"] - cmd_np[2]) for s in samples])),
            "contact_mean": _mean_metric(samples, "contact"),
            "slip_mean": _mean_metric(samples, "slip"),
            "clear_mean": _mean_metric(samples, "clear"),
            "height": _mean_metric(samples, "height"),
            "tilt": _mean_metric(samples, "tilt"),
            "qdev_l2": _mean_metric(samples, "qdev_l2"),
            "qmax": _mean_metric(samples, "qmax"),
            "action_abs": _mean_metric(samples, "action_abs"),
            "action_rate_abs": _mean_metric(samples, "action_rate_abs"),
            "done_count": int(sum(s["done"] > 0.5 for s in samples)),
        }
        rows.append(row)

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("policy", help="Path to policy .pkl")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    rows = evaluate_policy(args.policy, args.steps, args.warmup, args.seed, COMMANDS)
    output = args.output
    if output is None:
        output = str(Path(args.policy).with_name("fixed_command_diagnostics.jsonl"))

    with open(output, "w", encoding="utf-8") as f:
        for row in rows:
            line = json.dumps(row, ensure_ascii=False)
            print(line)
            f.write(line + "\n")

    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
