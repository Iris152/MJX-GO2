#!/usr/bin/env python3
"""Command-line entry point for Go2 training and visualization."""

import os
import sys
import argparse
import subprocess
from pathlib import Path

try:
    import mujoco
except ImportError:
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "mujoco",
            "mujoco_mjx",
            "brax",
            "mediapy",
            "optax",
            "flax",
            "matplotlib",
        ]
    )

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

cache_dir = Path.home() / ".jax_cache"
cache_dir.mkdir(exist_ok=True)
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", str(cache_dir))
python_bin = Path(sys.executable).resolve().parent
os.environ["PATH"] = f"{python_bin}{os.pathsep}{os.environ.get('PATH', '')}"

import jax

jax.config.update("jax_enable_x64", True)


GO2_DEFAULTS = {
    "algorithm": "jave",
    "variant": "blind_nolinvel_nokinref",
    "steps": 2_000_000,
    "seed": 42,
    "unroll_length": 12,
    "num_envs": 256,
    "model_xml": "src/envs/go2/models/scene_mjx.xml",
    "action_scale": 0.5,
    "default_joint_pose": None,
    "default_base_height": None,
    "settled_joint_pose": None,
    "settled_base_height": None,
    "policy_joint_pose": None,
    "prone_joint_pose": None,
    "prone_base_height": 0.057,
    "prone_hold_steps": 25,
    "standup_duration": 3.6,
    "standup_kp_start": 20.0,
    "standup_kp_end": 50.0,
    "standup_kd": 3.5,
    "target_base_height": 0.3,
    "reset_settle_steps": 0,
    "action_noise_std_start": None,
    "action_noise_std_end": None,
    "max_episode_length": 5000,
    "actor_history_len": 10,
    "actor_lr": 5e-3,
    "critic_lr": 5e-4,
    "critic_iterations": 16,
    "lr_decay": False,
    "diagnose": False,
    "resume": None,
    "checkpoint_interval": 100_000,
    "cmd_vel_x_range": None,
    "cmd_vel_y_range": None,
    "cmd_yaw_rate_range": None,
    "track_weights": None,
    "cmd_zero_prob": [
        0.1,
        0.7,
        0.5,
    ],
    "cmd_stand_prob": 0.0,
    "ahac_contact_force_threshold": 4.0,
    "ahac_contact_delta_threshold": 3.0,
    "ahac_penalty_weight": 0.02,
    "ahac_min_gradient_steps": 8,
    "ahac_steps_min": None,
    "ahac_steps_max": None,
    "ahac_initial_horizon": None,
    "ahac_lambd_lr": 5e-4,
    "ahac_horizon_contact_threshold": 1.0,
    "ahac_contact_truncation": False,
    "ahac_reset_optimizer_on_resume": False,
    "ahac_reset_best_on_resume": False,
    "ahac_override_hparams_on_resume": False,
    "sapo_init_alpha": 1.0,
    "sapo_target_entropy_scalar": 0.5,
    "sapo_alpha_lr": 0.005,
    "sapo_alpha_betas": [0.7, 0.95],
    "sapo_log_std_init": -1.0,
    "sapo_min_log_std": -5.0,
    "sapo_max_log_std": 2.0,
    "sapo_actor_hidden": [128, 64, 32],
    "sapo_critic_hidden": [64, 64],
    "sapo_max_grad_norm": 0.5,
    "sapo_per_env_grad_clip": 10.0,
    "sapo_weight_decay": 0.0,
    "zero_difficulty_frac": 0.0,
    "kp_range": [25.0, 45.0],
    "kd_range": [0.3, 0.7],
    "com_offset": [0.05, 0.05, 0.04],
    "terrain": False,
    "terrain_slope": 5.0,
    "no_curriculum": True,
    "curriculum_grace": None,
    "curriculum_steps": None,
    "visualize": None,
    "interactive": None,
    "plot": None,
    "no_post_viz": False,
}

GO2_VARIANTS = {
    "blind_nolinvel_nokinref",
    "blind_linvel_nokinref",
    "blind_linvel_kinref",
    "highspeed_nokinref",
}

_BLIND_CMD_DEFAULTS = {
    "cmd_vel_x_range": (-2.0, 2.0),
    "cmd_vel_y_range": (-1.0, 1.0),
    "cmd_yaw_rate_range": (-1.5, 1.5),
}

# Per-variant command-range defaults, used when not set via CLI or config.
VARIANT_CMD_DEFAULTS = {
    "blind_nolinvel_nokinref": _BLIND_CMD_DEFAULTS,
    "blind_linvel_nokinref": _BLIND_CMD_DEFAULTS,
    "blind_linvel_kinref": _BLIND_CMD_DEFAULTS,
    "highspeed_nokinref": {
        "cmd_vel_x_range": (-3.0, 3.0),
        "cmd_vel_y_range": (0.0, 0.0),
        "cmd_yaw_rate_range": (-1.0, 1.0),
    },
}

def _build_go2_parser(subparsers):
    parser = subparsers.add_parser(
        "go2",
        help="Go2 quadruped locomotion",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=argparse.SUPPRESS,
        help="Path to YAML config file",
    )

    parser.add_argument(
        "--algorithm",
        choices=["ahac", "jave", "sapo", "shac"],
        default=argparse.SUPPRESS,
        help="Training algorithm implementation",
    )
    parser.add_argument(
        "--variant",
        choices=sorted(GO2_VARIANTS),
        default=argparse.SUPPRESS,
        help="Go2 environment variant",
    )
    parser.add_argument(
        "--steps", type=int, default=argparse.SUPPRESS, help="Training steps"
    )
    parser.add_argument(
        "--seed", type=int, default=argparse.SUPPRESS, help="Random seed"
    )
    parser.add_argument(
        "--unroll-length",
        type=int,
        default=argparse.SUPPRESS,
        help="Short analytical rollout horizon h",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=argparse.SUPPRESS,
        help="Number of parallel training environments",
    )

    parser.add_argument(
        "--model-xml",
        type=str,
        default=argparse.SUPPRESS,
        help="Path to MuJoCo XML model file",
    )
    parser.add_argument(
        "--action-scale", type=float, default=argparse.SUPPRESS, help="Action scale"
    )
    parser.add_argument(
        "--default-joint-pose",
        type=float,
        nargs="+",
        default=argparse.SUPPRESS,
        metavar="Q",
        help="Optional default joint pose: either 3 per-leg values or 12 values "
        "in MuJoCo order [FL, FR, RL, RR]",
    )
    parser.add_argument(
        "--default-base-height",
        type=float,
        default=argparse.SUPPRESS,
        help="Optional initial root z height for the configured default pose",
    )
    parser.add_argument(
        "--target-base-height",
        type=float,
        default=argparse.SUPPRESS,
        help="Reward target for root height above ground",
    )
    parser.add_argument(
        "--action-noise-std-start",
        type=float,
        default=argparse.SUPPRESS,
        help="Initial Gaussian action noise std",
    )
    parser.add_argument(
        "--action-noise-std-end",
        type=float,
        default=argparse.SUPPRESS,
        help="Final Gaussian action noise std after linear schedule",
    )
    parser.add_argument(
        "--max-episode-length",
        type=int,
        default=argparse.SUPPRESS,
        help="Max steps per episode before forced reset",
    )
    parser.add_argument(
        "--actor-history-len",
        type=int,
        default=argparse.SUPPRESS,
        help="Number of actor observation frames to stack",
    )

    parser.add_argument(
        "--actor-lr", type=float, default=argparse.SUPPRESS, help="Actor learning rate"
    )
    parser.add_argument(
        "--critic-lr",
        type=float,
        default=argparse.SUPPRESS,
        help="Critic learning rate",
    )
    parser.add_argument(
        "--critic-iterations",
        type=int,
        default=argparse.SUPPRESS,
        help="Critic gradient steps per actor update",
    )

    parser.add_argument(
        "--lr-decay",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Enable learning rate decay",
    )
    parser.add_argument(
        "--diagnose",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Enable detailed diagnostic logging",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=argparse.SUPPRESS,
        help="Resume training from checkpoint (path to .pkl or folder)",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=argparse.SUPPRESS,
        help="Save checkpoint every N steps",
    )

    parser.add_argument(
        "--cmd-vel-x-range",
        type=float,
        nargs=2,
        default=argparse.SUPPRESS,
        metavar=("LO", "HI"),
        help="Forward velocity command range in m/s "
        "(default depends on variant, see VARIANT_CMD_DEFAULTS)",
    )
    parser.add_argument(
        "--cmd-vel-y-range",
        type=float,
        nargs=2,
        default=argparse.SUPPRESS,
        metavar=("LO", "HI"),
        help="Lateral velocity command range in m/s "
        "(default depends on variant, see VARIANT_CMD_DEFAULTS)",
    )
    parser.add_argument(
        "--cmd-yaw-rate-range",
        type=float,
        nargs=2,
        default=argparse.SUPPRESS,
        metavar=("LO", "HI"),
        help="Yaw rate command range in rad/s "
        "(default depends on variant, see VARIANT_CMD_DEFAULTS)",
    )
    parser.add_argument(
        "--track-weights",
        type=float,
        nargs=3,
        default=argparse.SUPPRESS,
        metavar=("VX", "VY", "YAW"),
        help="Optional reward weights for vx, vy, and yaw tracking",
    )
    parser.add_argument(
        "--cmd-zero-prob",
        type=float,
        nargs=3,
        default=argparse.SUPPRESS,
        metavar=("VX", "VY", "YAW"),
        help="Per-component probability of zeroing the velocity command "
        "(vx, vy, yaw) at each random command sample",
    )
    parser.add_argument(
        "--cmd-stand-prob",
        type=float,
        default=argparse.SUPPRESS,
        help="Probability of sampling an exact full-stop command",
    )
    parser.add_argument(
        "--ahac-contact-force-threshold",
        type=float,
        default=argparse.SUPPRESS,
        help="AHAC max foot normal force ratio before truncating actor gradients",
    )
    parser.add_argument(
        "--ahac-contact-delta-threshold",
        type=float,
        default=argparse.SUPPRESS,
        help="AHAC foot normal force change ratio before truncating actor gradients",
    )
    parser.add_argument(
        "--ahac-penalty-weight",
        type=float,
        default=argparse.SUPPRESS,
        help="Small actor penalty on contact-threshold excess",
    )
    parser.add_argument(
        "--ahac-min-gradient-steps",
        type=int,
        default=argparse.SUPPRESS,
        help="Minimum AHAC actor-gradient steps before contact truncation is allowed",
    )
    parser.add_argument(
        "--ahac-steps-min",
        type=int,
        default=argparse.SUPPRESS,
        help="Minimum adaptive AHAC horizon H",
    )
    parser.add_argument(
        "--ahac-steps-max",
        type=int,
        default=argparse.SUPPRESS,
        help="Maximum adaptive AHAC horizon H; defaults to --unroll-length",
    )
    parser.add_argument(
        "--ahac-initial-horizon",
        type=float,
        default=argparse.SUPPRESS,
        help="Initial adaptive AHAC horizon H; defaults to --ahac-steps-min",
    )
    parser.add_argument(
        "--ahac-lambd-lr",
        type=float,
        default=argparse.SUPPRESS,
        help="AHAC horizon dual learning rate",
    )
    parser.add_argument(
        "--ahac-horizon-contact-threshold",
        type=float,
        default=argparse.SUPPRESS,
        help="Normalized contact threshold C used to adapt AHAC horizon",
    )
    parser.add_argument(
        "--ahac-contact-truncation",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Also stop future actor gradients after stiff contacts, AHAC-1 style",
    )
    parser.add_argument(
        "--ahac-reset-optimizer-on-resume",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Reset actor/critic optimizer states when resuming AHAC fine-tuning",
    )
    parser.add_argument(
        "--ahac-reset-best-on-resume",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Start a fresh best-policy selection when resuming AHAC fine-tuning",
    )
    parser.add_argument(
        "--ahac-override-hparams-on-resume",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Use the current configuration when fine-tuning a checkpoint",
    )

    parser.add_argument(
        "--sapo-init-alpha",
        type=float,
        default=argparse.SUPPRESS,
        help="Initial SAPO entropy temperature",
    )
    parser.add_argument(
        "--sapo-target-entropy-scalar",
        type=float,
        default=argparse.SUPPRESS,
        help="SAPO target entropy scalar; target is -action_dim * scalar",
    )
    parser.add_argument(
        "--sapo-alpha-lr",
        type=float,
        default=argparse.SUPPRESS,
        help="SAPO entropy-temperature learning rate",
    )
    parser.add_argument(
        "--sapo-alpha-betas",
        type=float,
        nargs=2,
        default=argparse.SUPPRESS,
        metavar=("B1", "B2"),
        help="AdamW beta values for SAPO temperature optimization",
    )
    parser.add_argument(
        "--sapo-log-std-init",
        type=float,
        default=argparse.SUPPRESS,
        help="Initial SAPO state-dependent log standard deviation",
    )
    parser.add_argument(
        "--sapo-min-log-std",
        type=float,
        default=argparse.SUPPRESS,
        help="Lower bound for SAPO log standard deviation",
    )
    parser.add_argument(
        "--sapo-max-log-std",
        type=float,
        default=argparse.SUPPRESS,
        help="Upper bound for SAPO log standard deviation",
    )
    parser.add_argument(
        "--sapo-actor-hidden",
        type=int,
        nargs="+",
        default=argparse.SUPPRESS,
        metavar="H",
        help="SAPO actor hidden layer widths",
    )
    parser.add_argument(
        "--sapo-critic-hidden",
        type=int,
        nargs="+",
        default=argparse.SUPPRESS,
        metavar="H",
        help="SAPO critic hidden layer widths",
    )
    parser.add_argument(
        "--sapo-max-grad-norm",
        type=float,
        default=argparse.SUPPRESS,
        help="SAPO actor/critic gradient clipping norm",
    )
    parser.add_argument(
        "--sapo-per-env-grad-clip",
        type=float,
        default=argparse.SUPPRESS,
        help="Robust per-environment gradient clip before batch averaging",
    )
    parser.add_argument(
        "--sapo-weight-decay",
        type=float,
        default=argparse.SUPPRESS,
        help="SAPO AdamW weight decay",
    )

    parser.add_argument(
        "--zero-difficulty-frac",
        type=float,
        default=argparse.SUPPRESS,
        help="Fraction of envs held at difficulty=0 each unroll, regardless "
        "of curriculum progress. Prevents forgetting of easy behaviors. "
        "Default: 0.0.",
    )
    parser.add_argument(
        "--kp-range",
        type=float,
        nargs=2,
        default=argparse.SUPPRESS,
        metavar=("LO", "HI"),
        help="Actuator kp randomization range",
    )
    parser.add_argument(
        "--kd-range",
        type=float,
        nargs=2,
        default=argparse.SUPPRESS,
        metavar=("LO", "HI"),
        help="Actuator kd randomization range",
    )
    parser.add_argument(
        "--com-offset",
        type=float,
        nargs=3,
        default=argparse.SUPPRESS,
        metavar=("X", "Y", "Z"),
        help="COM offset half-ranges (m) for x, y, z randomization ",
    )

    parser.add_argument(
        "--terrain",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Enable implicit tilted-gravity terrain randomization",
    )
    parser.add_argument(
        "--terrain-slope",
        type=float,
        default=argparse.SUPPRESS,
        help="Max implicit slope angle in degrees at full difficulty",
    )
    parser.add_argument(
        "--no-curriculum",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Disable curriculum (immediate full difficulty)",
    )
    parser.add_argument(
        "--curriculum-grace",
        type=int,
        default=argparse.SUPPRESS,
        help="Steps at difficulty=0 before ramp starts",
    )
    parser.add_argument(
        "--curriculum-steps",
        type=int,
        default=argparse.SUPPRESS,
        help="Steps over which difficulty ramps 0->1",
    )

    parser.add_argument(
        "--visualize",
        type=str,
        default=argparse.SUPPRESS,
        help="Path to policy.pkl to render video",
    )
    parser.add_argument(
        "--interactive",
        type=str,
        default=argparse.SUPPRESS,
        help="Path to policy.pkl for interactive MuJoCo viewer",
    )
    parser.add_argument(
        "--plot", type=str, default=argparse.SUPPRESS, help="Path to log.npy to plot"
    )
    parser.add_argument(
        "--no-post-viz",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Skip automatic plot/video generation after training",
    )

    return parser


def _load_yaml_config(config_path):
    try:
        import yaml
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "PyYAML"])
        import yaml

    with open(config_path, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}

    if not isinstance(config, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {config_path}")
    return config


def _flatten_config(config):
    flattened = {}
    for key, value in config.items():
        if isinstance(value, dict) and key in {
            "training",
            "environment_params",
            "visualization",
        }:
            flattened.update(value)
        else:
            flattened[key] = value

    return flattened


def _apply_config(args, config):
    raw_values = vars(args)
    config_values = _flatten_config(config)

    # Always pop "embodiment" to avoid it being treated as unknown config key
    config_embodiment = config_values.pop("embodiment", None)
    embodiment = raw_values.get("embodiment") or config_embodiment
    if embodiment is not None:
        embodiment = str(embodiment).lower()

    if embodiment != "go2":
        merged = dict(raw_values)
        merged["embodiment"] = embodiment
        return argparse.Namespace(**merged)

    merged = dict(GO2_DEFAULTS)
    unknown_keys = sorted(key for key in config_values if key not in GO2_DEFAULTS)
    if unknown_keys:
        raise ValueError(f"Unknown config key(s): {', '.join(unknown_keys)}")

    merged.update(config_values)
    for key, value in raw_values.items():
        if key in GO2_DEFAULTS:
            merged[key] = value

    merged.update(
        {
            "config": raw_values.get("config"),
            "gpu": raw_values.get("gpu"),
            "embodiment": embodiment,
        }
    )
    if merged["algorithm"] not in {"ahac", "jave", "sapo", "shac"}:
        raise ValueError(
            "Config value 'algorithm' must be one of: 'ahac', 'jave', 'sapo', 'shac'"
        )
    if merged["variant"] not in GO2_VARIANTS:
        valid = ", ".join(sorted(GO2_VARIANTS))
        raise ValueError(f"Config value 'variant' must be one of: {valid}")
    return argparse.Namespace(**merged)


def _resolve_cmd_ranges(args):
    """Resolve command ranges: CLI/config values win, else per-variant defaults."""
    resolved = {}
    for key, default in VARIANT_CMD_DEFAULTS[args.variant].items():
        value = getattr(args, key)
        resolved[key] = tuple(value) if value is not None else default
    return resolved


def _run_go2(args):
    if args.interactive:
        from src.visualization.go2 import visualize_interactive

        visualize_interactive(args.interactive)
    elif args.visualize:
        from src.visualization.go2 import visualize

        visualize(args.visualize)
    elif args.plot:
        from src.visualization.go2 import plot_training

        plot_training(args.plot)
    else:
        if args.algorithm == "jave":
            from src.algorithms.jave.algorithm import train
        elif args.algorithm == "ahac":
            from src.algorithms.ahac.algorithm import train
        elif args.algorithm == "sapo":
            from src.algorithms.sapo.algorithm import train
        else:
            from src.algorithms.shac.algorithm import train
        from src.visualization.go2 import visualize, plot_training

        if args.no_curriculum:
            curriculum_grace = 0
            curriculum_steps = 1
        else:
            curriculum_grace = args.curriculum_grace
            curriculum_steps = args.curriculum_steps

        cmd_ranges = _resolve_cmd_ranges(args)

        train_kwargs = dict(
            total_steps=args.steps,
            unroll_length=args.unroll_length,
            num_envs=args.num_envs,
            xml_path=args.model_xml,
            actor_lr=args.actor_lr,
            critic_lr=args.critic_lr,
            critic_iterations=args.critic_iterations,
            action_scale=args.action_scale,
            use_lr_decay=args.lr_decay,
            diagnose=args.diagnose,
            seed=args.seed,
            resume_from=args.resume,
            checkpoint_interval=args.checkpoint_interval,
            terrain=args.terrain,
            terrain_slope_max=args.terrain_slope,
            curriculum_grace=curriculum_grace,
            curriculum_steps=curriculum_steps,
            kp_range=tuple(args.kp_range),
            kd_range=tuple(args.kd_range),
            com_offset_range=tuple(args.com_offset),
            cmd_zero_prob=tuple(args.cmd_zero_prob),
            cmd_stand_prob=args.cmd_stand_prob,
            track_weights=tuple(args.track_weights)
            if args.track_weights is not None
            else None,
            zero_difficulty_frac=args.zero_difficulty_frac,
            max_episode_length=args.max_episode_length,
            actor_history_len=args.actor_history_len,
            env_variant=args.variant,
            **cmd_ranges,
        )
        if args.variant == "blind_nolinvel_nokinref":
            for optional_key in [
                "default_joint_pose",
                "default_base_height",
                "settled_joint_pose",
                "settled_base_height",
                "policy_joint_pose",
                "target_base_height",
                "reset_settle_steps",
            ]:
                optional_value = getattr(args, optional_key)
                if optional_value is not None:
                    train_kwargs[optional_key] = optional_value
        if args.algorithm == "shac" and args.prone_joint_pose is not None:
            train_kwargs.update(
                prone_joint_pose=args.prone_joint_pose,
                prone_base_height=args.prone_base_height,
                prone_hold_steps=args.prone_hold_steps,
                standup_duration=args.standup_duration,
                standup_kp_start=args.standup_kp_start,
                standup_kp_end=args.standup_kp_end,
                standup_kd=args.standup_kd,
            )
        if args.action_noise_std_start is not None:
            train_kwargs["action_noise_std_start"] = args.action_noise_std_start
        if args.action_noise_std_end is not None:
            train_kwargs["action_noise_std_end"] = args.action_noise_std_end
        if args.algorithm == "ahac":
            train_kwargs.update(
                ahac_contact_force_threshold=args.ahac_contact_force_threshold,
                ahac_contact_delta_threshold=args.ahac_contact_delta_threshold,
                ahac_penalty_weight=args.ahac_penalty_weight,
                ahac_min_gradient_steps=args.ahac_min_gradient_steps,
                ahac_steps_min=args.ahac_steps_min,
                ahac_steps_max=args.ahac_steps_max,
                ahac_initial_horizon=args.ahac_initial_horizon,
                ahac_lambd_lr=args.ahac_lambd_lr,
                ahac_horizon_contact_threshold=args.ahac_horizon_contact_threshold,
                ahac_contact_truncation=args.ahac_contact_truncation,
                ahac_reset_optimizer_on_resume=args.ahac_reset_optimizer_on_resume,
                ahac_reset_best_on_resume=args.ahac_reset_best_on_resume,
                ahac_override_hparams_on_resume=args.ahac_override_hparams_on_resume,
            )
        elif args.algorithm == "sapo":
            train_kwargs.update(
                sapo_init_alpha=args.sapo_init_alpha,
                sapo_target_entropy_scalar=args.sapo_target_entropy_scalar,
                sapo_alpha_lr=args.sapo_alpha_lr,
                sapo_alpha_betas=tuple(args.sapo_alpha_betas),
                sapo_log_std_init=args.sapo_log_std_init,
                sapo_min_log_std=args.sapo_min_log_std,
                sapo_max_log_std=args.sapo_max_log_std,
                sapo_actor_hidden=tuple(args.sapo_actor_hidden),
                sapo_critic_hidden=tuple(args.sapo_critic_hidden),
                sapo_max_grad_norm=args.sapo_max_grad_norm,
                sapo_per_env_grad_clip=args.sapo_per_env_grad_clip,
                sapo_weight_decay=args.sapo_weight_decay,
            )

        state, folder = train(**train_kwargs)

        if args.no_post_viz:
            print(f"Training artifacts saved to {folder}")
            return

        plot_training(f"{folder}/log.npy")
        for policy_name in [
            "policy_best_tracking.pkl",
            "policy_best.pkl",
            "policy_final.pkl",
        ]:
            policy_path = f"{folder}/{policy_name}"
            if os.path.exists(policy_path):
                visualize(policy_path)
                break


def main():
    parser = argparse.ArgumentParser(
        description="Open-DiffLoco: SHAC/JAVE locomotion training with differentiable simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m src.main --gpu 0 go2 --steps 100000
    python -m src.main --config src/configs/jave_go2.yaml
    python -m src.main --config src/configs/shac_go2.yaml go2 --steps 500000
    python -m src.main --gpu 1 go2 --visualize runs/policy_best.pkl
    python -m src.main --gpu 1 go2 --interactive runs/policy_best.pkl
        """,
    )

    parser.add_argument(
        "--gpu", type=int, default=0, help="GPU index to use (CUDA_VISIBLE_DEVICES)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file",
    )

    subparsers = parser.add_subparsers(dest="embodiment", help="Choose embodiment")
    _build_go2_parser(subparsers)

    args = parser.parse_args()
    config = _load_yaml_config(args.config) if args.config else {}
    args = _apply_config(args, config)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    if args.embodiment is None:
        parser.print_help()
        sys.exit(1)
    elif args.embodiment == "go2":
        _run_go2(args)
    else:
        parser.error(f"Unsupported embodiment: {args.embodiment}")


if __name__ == "__main__":
    main()
