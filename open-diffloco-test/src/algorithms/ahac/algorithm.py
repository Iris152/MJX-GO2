"""Adaptive Horizon Actor-Critic training for Go2 locomotion."""

import os
import time
import pickle
import json
from datetime import datetime

# Set to True to enable per-foot normal force logging
DEBUG_FOOT_CONTACTS = False

import jax
import jax.numpy as jp
import optax
import numpy as np

from src.core.data_structures import (
    Normalizer,
    TrainState,
    slim_checkpoint_state,
)
from src.core.networks import Actor, Critic
from src.envs.go2.environment import Go2Env
from src.envs.go2.terrain import differentiated_ou_foot_forces
from src.core.utils import compute_grad_norm


def _gate_gradient_leaf(x, gate):
    """Keep values unchanged while stopping gradients when gate is zero."""
    if hasattr(x, "dtype") and jp.issubdtype(x.dtype, jp.inexact):
        gate = gate.astype(x.dtype)
        return gate * x + (1.0 - gate) * jax.lax.stop_gradient(x)
    return x


def _gate_gradient_tree(tree, gate):
    return jax.tree_util.tree_map(lambda x: _gate_gradient_leaf(x, gate), tree)


def load_checkpoint(path: str):
    """
    Load a training checkpoint.

    Args:
        path: Path to a .pkl file, or a training folder containing one.
              When given a folder, searches in order:
              checkpoint_latest.pkl, policy_best_tracking.pkl, policy_best.pkl,
              policy_final.pkl

    Returns:
        Tuple of (state, hparams, step) where:
            - state: TrainState object
            - hparams: dict of hyperparameters (or None)
            - step: training step count
    """
    if os.path.isdir(path):
        for name in [
            "checkpoint_latest.pkl",
            "policy_best_tracking.pkl",
            "policy_best.pkl",
            "policy_final.pkl",
        ]:
            candidate = os.path.join(path, name)
            if os.path.exists(candidate):
                path = candidate
                break
        else:
            raise FileNotFoundError(f"No checkpoint found in {path}")

    print(f"Loading checkpoint from {path}")
    with open(path, "rb") as f:
        state = pickle.load(f)

    # Try to load hyperparameters from same directory
    hparams = None
    hparams_path = os.path.join(os.path.dirname(path), "hparams.json")
    if os.path.exists(hparams_path):
        with open(hparams_path) as f:
            hparams = json.load(f)

    return state, hparams, int(jax.device_get(state.step))


def train(
    # General
    total_steps: int = 100_000,
    unroll_length: int = 12,
    num_envs: int = 256,
    actor_lr: float = 5e-3,
    critic_lr: float = 5e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    target_update_rate: float = 0.01,
    critic_iterations: int = 16,
    use_lr_decay: bool = False,
    xml_path: str = "src/envs/go2/models/scene_mjx.xml",
    action_scale: float = 0.5,
    default_joint_pose: tuple = None,
    default_base_height: float = None,
    settled_joint_pose: tuple = None,
    settled_base_height: float = None,
    target_base_height: float = 0.3,
    reset_settle_steps: int = 0,
    # Commands
    cmd_vel_x_range: tuple = (-2.0, 2.0),
    cmd_vel_y_range: tuple = (-1.0, 1.0),
    cmd_yaw_rate_range: tuple = (-1.5, 1.5),
    cmd_zero_prob: tuple = (0.1, 0.7, 0.5),
    cmd_stand_prob: float = 0.0,
    cmd_ctrl_interval_range: tuple = (60, 140),
    track_weights: tuple = None,
    # Randomization
    action_noise_std_start: float = 0.5,
    action_noise_std_end: float = 0.32,
    friction_range: tuple = (0.5, 2.0),
    mass_range: tuple = (0.85, 1.15),
    kp_range: tuple = (25.0, 45.0),
    kd_range: tuple = (0.3, 0.7),
    com_offset_range: tuple = (0.05, 0.05, 0.04),
    push_velocity_range: tuple = (-1.0, 1.0),
    push_interval_s: float = 4.0,
    terrain_flat_prob: float = 0.2,
    terrain_slope_max: float = 5.0,
    terrain_bump_std: float = 0.4,
    terrain_bump_decay: float = 0.4,
    terrain: bool = False,
    # Annealing
    zero_difficulty_frac: float = 0.0,
    curriculum_grace: int = None,
    curriculum_steps: int = None,
    # AHAC
    ahac_contact_force_threshold: float = 4.0,
    ahac_contact_delta_threshold: float = 3.0,
    ahac_penalty_weight: float = 0.02,
    ahac_min_gradient_steps: int = 8,
    ahac_steps_min: int = None,
    ahac_steps_max: int = None,
    ahac_initial_horizon: float = None,
    ahac_lambd_lr: float = 5e-4,
    ahac_horizon_contact_threshold: float = 1.0,
    ahac_contact_truncation: bool = False,
    ahac_reset_optimizer_on_resume: bool = False,
    ahac_reset_best_on_resume: bool = False,
    ahac_override_hparams_on_resume: bool = False,
    # Misc.
    diagnose: bool = False,
    seed: int = 0,
    resume_from: str = None,
    checkpoint_interval: int = 10_000,
    max_episode_length: int = 5000,
    actor_history_len: int = 10,
    env_variant: str = "blind_nolinvel_nokinref",
):
    """
    Train a quadruped locomotion policy using an AHAC-style actor update.

    Args:
        total_steps: Total environment steps to train
        unroll_length: Number of steps per trajectory rollout (short horizon h)
        num_envs: Number of parallel environments (N)
        actor_lr: Actor learning rate
        critic_lr: Critic learning rate
        gamma: Discount factor
        target_update_rate: Soft update rate for target critic (1-alpha)
        critic_iterations: Number of critic gradient steps per actor update
        use_lr_decay: Linear LR decay to 62.5% over training
        action_scale: Scale factor for actions
        default_joint_pose: Optional 3-D per-leg or 12-D full-body default
            joint pose in MuJoCo order [FL, FR, RL, RR].
        default_base_height: Optional initial root z height for default pose.
        target_base_height: Reward target for root height above ground.
        cmd_vel_x_range: (min, max) for forward velocity command (m/s)
        cmd_vel_y_range: (min, max) for lateral velocity command (m/s)
        cmd_yaw_rate_range: (min, max) for yaw rate command (rad/s)
        cmd_zero_prob: Per-component probability of zeroing (vx, vy, yaw)
        cmd_stand_prob: Probability of forcing all command axes to exactly zero.
        cmd_ctrl_interval_range: (min, max) steps between random command samples
        action_noise_std_start: Std dev of Gaussian action noise at step 0
        action_noise_std_end: Std dev of Gaussian action noise at total_steps
        friction_range: (lo, hi) multiplicative factor for geom_friction per episode
        mass_range: (lo, hi) multiplicative factor for body_mass per episode
        kp_range: (lo, hi) absolute range for actuator position gain per episode
        kd_range: (lo, hi) absolute range for actuator velocity gain per episode
        push_velocity_range: Interval root x/y velocity disturbance range.
        push_interval_s: Seconds between velocity pushes.
        terrain_flat_prob: Fraction of terrain episodes that use nominal gravity.
        terrain: Enable implicit tilted-gravity terrain randomization.
        zero_difficulty_frac: Fraction of envs that are held at difficulty=0 each
                              unroll, regardless of curriculum progress. These envs
                              see nominal gravity, nominal gains, and no COM offset.
        curriculum_grace: Steps at difficulty=0 before ramping starts.
        curriculum_steps: Steps over which difficulty ramps 0->1 (after grace).
        ahac_contact_force_threshold: Per-foot normal-force ratio that triggers
            actor-gradient truncation. Values are normalized by nominal body
            weight / 4, so normal stance is near 1.0 per supporting foot.
        ahac_contact_delta_threshold: Per-foot normal-force jump ratio that
            triggers actor-gradient truncation.
        ahac_penalty_weight: Small actor penalty on contact-threshold excess.
        ahac_min_gradient_steps: Minimum rollout steps before AHAC contact
            truncation may cut actor gradients.
        ahac_steps_min: Minimum adaptive horizon H. Defaults to
            ahac_min_gradient_steps.
        ahac_steps_max: Maximum adaptive horizon H. Defaults to unroll_length.
        ahac_initial_horizon: Initial adaptive horizon H. Defaults to
            ahac_steps_min.
        ahac_lambd_lr: Dual learning rate used for AHAC horizon adaptation.
        ahac_horizon_contact_threshold: Contact threshold C used by the AHAC
            horizon update. The contact metric is normalized by the stiff-contact
            force/delta thresholds, so values above 1.0 indicate violations.
        ahac_contact_truncation: If true, additionally stops future actor
            gradients after stiff contacts, matching the older AHAC-1-style local
            implementation. Full AHAC leaves this false and adapts H instead.
        ahac_reset_optimizer_on_resume: Reinitialize optimizer states after
            loading a policy, useful when AHAC fine-tunes a SHAC checkpoint.
        ahac_reset_best_on_resume: Start fresh best-policy selection in the new
            AHAC run instead of inheriting the source run's thresholds.
        ahac_override_hparams_on_resume: Keep the supplied configuration when
            fine-tuning a checkpoint, while restoring its learned state.
        diagnose: Enable detailed diagnostic logging
        seed: Random seed
        resume_from: Path to checkpoint .pkl file or training folder to resume from
        checkpoint_interval: Save checkpoint every N steps

    Returns:
        Tuple of (final_state, save_directory)
    """
    # Handle checkpoint resumption
    resumed_state = None
    resumed_step = 0
    resumed_hparams = None

    if resume_from:
        resumed_state, resumed_hparams, resumed_step = load_checkpoint(resume_from)
        if resumed_hparams and not ahac_override_hparams_on_resume:
            print(f"Resuming from step {resumed_step}")
            print(
                f"  Loaded hparams: action_scale={resumed_hparams.get('action_scale')}"
            )
            action_scale = resumed_hparams.get("action_scale", action_scale)
            if not ahac_reset_optimizer_on_resume:
                action_noise_std_start = resumed_hparams.get(
                    "action_noise_std_start", action_noise_std_start
                )
                action_noise_std_end = resumed_hparams.get(
                    "action_noise_std_end", action_noise_std_end
                )
            xml_path = resumed_hparams.get("xml_path", xml_path)
            env_variant = resumed_hparams.get("env_variant", env_variant)
            if "default_joint_pose" in resumed_hparams:
                default_joint_pose = resumed_hparams["default_joint_pose"]
            if "default_base_height" in resumed_hparams:
                default_base_height = resumed_hparams["default_base_height"]
            if "settled_joint_pose" in resumed_hparams:
                settled_joint_pose = resumed_hparams["settled_joint_pose"]
            if "settled_base_height" in resumed_hparams:
                settled_base_height = resumed_hparams["settled_base_height"]
            if "kp_range" in resumed_hparams:
                kp_range = tuple(resumed_hparams["kp_range"])
            if "kd_range" in resumed_hparams:
                kd_range = tuple(resumed_hparams["kd_range"])
            if "com_offset_range" in resumed_hparams:
                com_offset_range = tuple(resumed_hparams["com_offset_range"])
            if "push_velocity_range" in resumed_hparams:
                push_velocity_range = tuple(resumed_hparams["push_velocity_range"])
            if "push_interval_s" in resumed_hparams:
                push_interval_s = resumed_hparams["push_interval_s"]
            if "terrain_bump_std" in resumed_hparams:
                terrain_bump_std = resumed_hparams["terrain_bump_std"]
            if "terrain_bump_decay" in resumed_hparams:
                terrain_bump_decay = resumed_hparams["terrain_bump_decay"]
            if "cmd_ctrl_interval_range" in resumed_hparams:
                cmd_ctrl_interval_range = tuple(
                    resumed_hparams["cmd_ctrl_interval_range"]
                )
            if "zero_difficulty_frac" in resumed_hparams:
                zero_difficulty_frac = resumed_hparams["zero_difficulty_frac"]
            if "curriculum_grace" in resumed_hparams:
                curriculum_grace = resumed_hparams["curriculum_grace"]
            if "curriculum_steps" in resumed_hparams:
                curriculum_steps = resumed_hparams["curriculum_steps"]
            if "max_episode_length" in resumed_hparams:
                max_episode_length = resumed_hparams["max_episode_length"]
            if "actor_history_len" in resumed_hparams:
                actor_history_len = resumed_hparams["actor_history_len"]
            if "ahac_contact_force_threshold" in resumed_hparams:
                ahac_contact_force_threshold = resumed_hparams[
                    "ahac_contact_force_threshold"
                ]
            if "ahac_contact_delta_threshold" in resumed_hparams:
                ahac_contact_delta_threshold = resumed_hparams[
                    "ahac_contact_delta_threshold"
                ]
            if "ahac_penalty_weight" in resumed_hparams:
                ahac_penalty_weight = resumed_hparams["ahac_penalty_weight"]
            if "ahac_min_gradient_steps" in resumed_hparams:
                ahac_min_gradient_steps = resumed_hparams["ahac_min_gradient_steps"]
            if "ahac_steps_min" in resumed_hparams:
                ahac_steps_min = resumed_hparams["ahac_steps_min"]
            if "ahac_steps_max" in resumed_hparams:
                ahac_steps_max = resumed_hparams["ahac_steps_max"]
            if "ahac_initial_horizon" in resumed_hparams:
                ahac_initial_horizon = resumed_hparams["ahac_initial_horizon"]
            if "ahac_lambd_lr" in resumed_hparams:
                ahac_lambd_lr = resumed_hparams["ahac_lambd_lr"]
            if "ahac_horizon_contact_threshold" in resumed_hparams:
                ahac_horizon_contact_threshold = resumed_hparams[
                    "ahac_horizon_contact_threshold"
                ]
            if "ahac_contact_truncation" in resumed_hparams:
                ahac_contact_truncation = resumed_hparams["ahac_contact_truncation"]
        elif resumed_hparams:
            print("Using current configuration for AHAC fine-tuning")

    # Compute curriculum defaults relative to total_steps
    if curriculum_grace is None:
        curriculum_grace = total_steps // 10  # 10% grace at difficulty=0
    if curriculum_steps is None:
        curriculum_steps = int(total_steps * 0.8)  # ramp over 80%

    # Original AHAC keeps a learnable model-based horizon H in [steps_min,
    # steps_max]. In JAX/MJX the scan length must stay static, so unroll_length
    # is treated as steps_max unless an explicit maximum is provided.
    if ahac_steps_min is None:
        ahac_steps_min = ahac_min_gradient_steps
    if ahac_steps_max is None:
        ahac_steps_max = unroll_length
    if ahac_initial_horizon is None:
        ahac_initial_horizon = float(ahac_steps_min)
    if ahac_steps_min <= 0 or ahac_steps_max <= ahac_steps_min:
        raise ValueError(
            "AHAC requires 0 < ahac_steps_min < ahac_steps_max "
            f"(got {ahac_steps_min}, {ahac_steps_max})"
        )
    if not (ahac_steps_min <= ahac_initial_horizon <= ahac_steps_max):
        raise ValueError(
            "ahac_initial_horizon must lie within [ahac_steps_min, ahac_steps_max]"
        )
    if ahac_lambd_lr <= 0:
        raise ValueError("ahac_lambd_lr must be positive")
    if ahac_horizon_contact_threshold <= 0:
        raise ValueError("ahac_horizon_contact_threshold must be positive")

    unroll_length = int(ahac_steps_max)

    _curriculum_steps = max(curriculum_steps, 1)  # avoid division by zero

    _curriculum_grace_jax = jp.array(curriculum_grace, dtype=jp.int32)
    _curriculum_steps_jax = jp.array(_curriculum_steps, dtype=jp.float32)

    env_kwargs = dict(
        variant=env_variant,
        xml_path=xml_path,
        action_scale=action_scale,
        cmd_vel_x_range=cmd_vel_x_range,
        cmd_vel_y_range=cmd_vel_y_range,
        cmd_yaw_rate_range=cmd_yaw_rate_range,
        cmd_zero_prob=cmd_zero_prob,
        cmd_stand_prob=cmd_stand_prob,
        cmd_ctrl_interval_range=cmd_ctrl_interval_range,
        friction_range=friction_range,
        mass_range=mass_range,
        kp_range=kp_range,
        kd_range=kd_range,
        com_offset_range=com_offset_range,
        terrain_flat_prob=terrain_flat_prob,
        terrain_slope_max=terrain_slope_max if terrain else 0.0,
        max_episode_length=max_episode_length,
        actor_history_len=actor_history_len,
    )
    if env_variant == "blind_nolinvel_nokinref":
        env_kwargs.update(
            default_joint_pose=default_joint_pose,
            default_base_height=default_base_height,
            settled_joint_pose=settled_joint_pose,
            settled_base_height=settled_base_height,
            target_base_height=target_base_height,
            reset_settle_steps=reset_settle_steps,
        )
    if track_weights is not None and env_variant == "blind_nolinvel_nokinref":
        env_kwargs["track_weights"] = tuple(track_weights)
    env = Go2Env(**env_kwargs)
    actor_norm = Normalizer(env.actor_frame_obs_dim)
    critic_norm = Normalizer(env.critic_obs_dim)

    # Create save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"training_runs/ahac_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    print(f"Algorithm: AHAC, Save dir: {save_dir}")
    print(
        "AHAC contact gating: "
        f"force_ratio>{ahac_contact_force_threshold}, "
        f"delta_ratio>{ahac_contact_delta_threshold}, "
        f"min_grad_steps={ahac_min_gradient_steps}, "
        f"penalty_weight={ahac_penalty_weight}"
    )
    print(
        "AHAC adaptive horizon: "
        f"H0={ahac_initial_horizon:.1f}, H=[{ahac_steps_min},{ahac_steps_max}], "
        f"lambda_lr={ahac_lambd_lr}, C={ahac_horizon_contact_threshold}, "
        f"contact_truncation={'ON' if ahac_contact_truncation else 'OFF'}"
    )
    print(
        f"Domain randomization: action_noise={action_noise_std_start}->{action_noise_std_end}, "
        f"friction={friction_range}, mass={mass_range}, "
        f"kp={kp_range}, kd={kd_range}, "
        f"com_offset=+/-{com_offset_range}, "
        f"velocity_push={push_velocity_range} every {push_interval_s}s"
    )
    print(
        f"Curriculum: grace={curriculum_grace}, curriculum={curriculum_steps} steps, "
        f"terrain={'ON' if terrain else 'OFF'}, "
        f"terrain_flat_prob={terrain_flat_prob}, "
        f"terrain_slope_max={terrain_slope_max} deg, "
        f"terrain_bump_std={terrain_bump_std if terrain else 0.0}"
    )

    best_reward = (
        resumed_hparams.get("best_reward", -np.inf) if resumed_hparams else -np.inf
    )
    best_tracking = (
        resumed_hparams.get("best_tracking", np.inf) if resumed_hparams else np.inf
    )
    if resumed_state is not None and ahac_reset_best_on_resume:
        best_reward = -np.inf
        best_tracking = np.inf

    # Save hyperparameters up front so interrupted runs can be resumed.
    hparams = {
        "algorithm": "ahac",
        "total_steps": total_steps,
        "unroll_length": unroll_length,
        "num_envs": num_envs,
        "actor_lr": actor_lr,
        "critic_lr": critic_lr,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "target_update_rate": target_update_rate,
        "critic_iterations": critic_iterations,
        "xml_path": xml_path,
        "action_scale": action_scale,
        "default_joint_pose": list(default_joint_pose)
        if default_joint_pose is not None
        else None,
        "default_base_height": default_base_height,
        "settled_joint_pose": list(settled_joint_pose)
        if settled_joint_pose is not None
        else None,
        "settled_base_height": settled_base_height,
        "target_base_height": target_base_height,
        "cmd_vel_x_range": list(cmd_vel_x_range),
        "cmd_vel_y_range": list(cmd_vel_y_range),
        "cmd_yaw_rate_range": list(cmd_yaw_rate_range),
        "cmd_zero_prob": list(cmd_zero_prob),
        "cmd_stand_prob": cmd_stand_prob,
        "cmd_ctrl_interval_range": list(cmd_ctrl_interval_range),
        "track_weights": list(track_weights) if track_weights is not None else None,
        "action_noise_std_start": action_noise_std_start,
        "action_noise_std_end": action_noise_std_end,
        "friction_range": list(friction_range),
        "mass_range": list(mass_range),
        "kp_range": list(kp_range),
        "kd_range": list(kd_range),
        "com_offset_range": list(com_offset_range),
        "push_velocity_range": list(push_velocity_range),
        "push_interval_s": push_interval_s,
        "terrain_flat_prob": terrain_flat_prob,
        "terrain_slope_max": terrain_slope_max,
        "terrain_bump_std": terrain_bump_std,
        "terrain_bump_decay": terrain_bump_decay,
        "terrain": terrain,
        "zero_difficulty_frac": zero_difficulty_frac,
        "curriculum_grace": curriculum_grace,
        "curriculum_steps": curriculum_steps,
        "ahac_contact_force_threshold": ahac_contact_force_threshold,
        "ahac_contact_delta_threshold": ahac_contact_delta_threshold,
        "ahac_penalty_weight": ahac_penalty_weight,
        "ahac_min_gradient_steps": ahac_min_gradient_steps,
        "ahac_steps_min": ahac_steps_min,
        "ahac_steps_max": ahac_steps_max,
        "ahac_initial_horizon": ahac_initial_horizon,
        "ahac_lambd_lr": ahac_lambd_lr,
        "ahac_horizon_contact_threshold": ahac_horizon_contact_threshold,
        "ahac_contact_truncation": ahac_contact_truncation,
        "ahac_reset_optimizer_on_resume": ahac_reset_optimizer_on_resume,
        "ahac_reset_best_on_resume": ahac_reset_best_on_resume,
        "ahac_override_hparams_on_resume": ahac_override_hparams_on_resume,
        "seed": seed,
        "best_reward": best_reward,
        "best_tracking": best_tracking,
        "max_episode_length": max_episode_length,
        "actor_history_len": actor_history_len,
        "reset_settle_steps": env.reset_settle_steps,
        "env_variant": env_variant,
    }
    with open(f"{save_dir}/hparams.json", "w") as f:
        json.dump(hparams, f, indent=2)

    # Initialize random keys
    key = jax.random.PRNGKey(seed)
    key, k1, k2, k3 = jax.random.split(key, 4)

    # Initialize networks
    actor = Actor(env.action_dim)
    critic = Critic()

    actor_dummy = jp.zeros((1, env.actor_obs_dim), dtype=jp.float32)
    critic_dummy = jp.zeros((1, env.critic_obs_dim), dtype=jp.float32)
    actor_params = actor.init(k1, actor_dummy)
    critic_params = critic.init(k2, critic_dummy)
    target_critic_params = critic_params

    actor_normalizer = actor_norm.init()
    critic_normalizer = critic_norm.init()

    # Linear LR decay
    schedule_iters = max(
        1,
        int(np.ceil(total_steps / (num_envs * float(ahac_steps_max)))),
    )
    if use_lr_decay:
        lr_floor = 0.62
        actor_schedule = optax.linear_schedule(
            init_value=actor_lr,
            end_value=actor_lr * lr_floor,
            transition_steps=schedule_iters,
        )

        critic_schedule = optax.linear_schedule(
            init_value=critic_lr,
            end_value=critic_lr * lr_floor,
            transition_steps=schedule_iters * critic_iterations,
        )
        print(
            f"LR decay: linear over ~{schedule_iters} adaptive-H iters, "
            f"actor {actor_lr:.1e} --> {actor_lr * lr_floor:.1e} ({schedule_iters} steps), "
            f"critic {critic_lr:.1e} --> {critic_lr * lr_floor:.1e} ({schedule_iters * critic_iterations} steps)"
        )
    else:
        actor_schedule = actor_lr
        critic_schedule = critic_lr

    # Initialize optimizers
    actor_opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(actor_schedule))
    critic_opt = optax.chain(
        optax.clip_by_global_norm(1.0), optax.adam(critic_schedule)
    )

    actor_opt_state = actor_opt.init(actor_params)
    critic_opt_state = critic_opt.init(critic_params)

    # Initialize environments at difficulty=0 (flat ground)
    env_keys = jax.random.split(k3, num_envs)
    env_state = jax.vmap(env.reset)(env_keys, jp.zeros(num_envs))

    _push_interval_steps = max(int(round(push_interval_s / env.dt)), 1)
    _push_velocity_lo = jp.array(push_velocity_range[0], dtype=jp.float64)
    _push_velocity_hi = jp.array(push_velocity_range[1], dtype=jp.float64)
    _foot_body_ids = env._foot_body_ids
    _nominal_weight = env.nominal_total_mass * env.base_gravity_mag
    _terrain_bump_std = terrain_bump_std if terrain else 0.0

    def actor_loss(
        actor_params,
        target_critic_params,
        actor_norm_state,
        critic_norm_state,
        env_state,
        randomization,
        current_noise_std,
        ahac_horizon,
        ahac_lambda,
    ):
        """Short-horizon actor objective with sampled perturbations."""
        action_noise, velocity_pushes, terrain_bump_innovations = randomization
        horizon_steps = jp.clip(
            jp.rint(ahac_horizon).astype(jp.int32), ahac_steps_min, ahac_steps_max
        )
        lambda_vec = ahac_lambda.astype(jp.float64)

        def rollout_step(carry, inputs):
            noise_t, velocity_push_t, terrain_bump_innov_t, scan_idx = inputs
            (
                state,
                foot_bump_ou,
                grad_alive,
                prev_foot_forces,
                grad_len,
                horizon_state,
                horizon_foot_bump_ou,
            ) = carry
            horizon_active = (scan_idx < horizon_steps).astype(jp.float32)

            push_due = (state.info["step"] > 0) & (
                (state.info["step"] % _push_interval_steps) == 0
            )
            scaled_push = state.info["difficulty"] * velocity_push_t
            pushed_qvel = state.data.qvel.at[:2].set(scaled_push)
            state = state.replace(
                data=state.data.replace(
                    qvel=jp.where(push_due, pushed_qvel, state.data.qvel)
                )
            )

            foot_bump_ou, terrain_bump_forces = differentiated_ou_foot_forces(
                foot_bump_ou,
                terrain_bump_innov_t,
                jax.lax.stop_gradient(state.info["foot_normal_forces"]),
                difficulty=state.info["difficulty"],
                std=_terrain_bump_std,
                decay=terrain_bump_decay,
                robot_weight=_nominal_weight,
            )
            xfrc = state.data.xfrc_applied
            for i in range(4):
                xfrc = xfrc.at[_foot_body_ids[i], :3].add(terrain_bump_forces[i])
            state = state.replace(data=state.data.replace(xfrc_applied=xfrc))

            # Full AHAC adapts H to avoid stiff contacts. Keeping this gate
            # optional preserves the older AHAC-1-style truncation for ablations.
            state_for_grad = _gate_gradient_tree(state, grad_alive * horizon_active)

            # Actor sees noisy observations; critic/training targets keep raw obs.
            obs_rng, env_rng = jax.random.split(state_for_grad.info["rng"])
            state_for_grad = state_for_grad.replace(
                info={**state_for_grad.info, "rng": env_rng}
            )
            actor_obs = env._apply_obs_noise(state_for_grad.obs, obs_rng)

            obs_norm = env.normalize_actor_obs(
                actor_norm, actor_norm_state, actor_obs
            ).astype(jp.float32)
            action = actor.apply(actor_params, obs_norm).astype(jp.float64)
            noisy_action = action + current_noise_std * noise_t.astype(jp.float64)
            noisy_action = jp.clip(noisy_action, -1.0, 1.0)

            next_state = env.step(state_for_grad, noisy_action)
            foot_bump_ou = jp.where(next_state.done, jp.zeros((4, 3)), foot_bump_ou)

            foot_forces = next_state.info["foot_normal_forces"]
            foot_force_scale = (_nominal_weight / 4.0) + 1e-6
            force_ratio = jp.max(jp.abs(foot_forces)) / foot_force_scale
            delta_ratio = jp.mean(jp.abs(foot_forces - prev_foot_forces)) / foot_force_scale
            force_excess = jp.maximum(force_ratio - ahac_contact_force_threshold, 0.0)
            delta_excess = jp.maximum(delta_ratio - ahac_contact_delta_threshold, 0.0)
            contact_excess = force_excess + delta_excess
            contact_metric = jp.maximum(
                force_ratio / (ahac_contact_force_threshold + 1e-6),
                delta_ratio / (ahac_contact_delta_threshold + 1e-6),
            )
            contact_violation = (
                (contact_excess > 0.0)
                & (scan_idx >= ahac_min_gradient_steps)
                & (next_state.done == 0.0)
                & (horizon_active > 0.0)
                & ahac_contact_truncation
            )
            next_grad_alive = grad_alive * (1.0 - contact_violation.astype(jp.float64))
            grad_len = grad_len + grad_alive * horizon_active
            next_state_for_carry = _gate_gradient_tree(next_state, next_grad_alive)
            next_foot_forces = jp.where(next_state.done, jp.zeros(4), foot_forces)
            horizon_trunc = (scan_idx == (horizon_steps - 1)) & (horizon_active > 0.0)
            horizon_state = jax.tree_util.tree_map(
                lambda old, new: jp.where(horizon_trunc, new, old),
                horizon_state,
                next_state_for_carry,
            )
            horizon_foot_bump_ou = jp.where(
                horizon_trunc, foot_bump_ou, horizon_foot_bump_ou
            )
            done_for_loss = jp.where(
                horizon_trunc, jp.ones_like(next_state.done), next_state.done
            )

            return (
                next_state_for_carry,
                foot_bump_ou,
                next_grad_alive,
                next_foot_forces,
                grad_len,
                horizon_state,
                horizon_foot_bump_ou,
            ), {
                "reward": next_state.reward * horizon_active,
                "done": jp.where(horizon_active > 0.0, done_for_loss, 0.0),
                "terminal": jp.where(
                    horizon_active > 0.0, next_state.info["terminal"], 0.0
                ),
                "actor_obs": state.obs,
                "critic_obs": env._get_critic_obs(state.data, state.info),
                "bootstrap_critic_obs": next_state.info["bootstrap_critic_obs"],
                "ahac_contact_excess": contact_excess * grad_alive * horizon_active,
                "ahac_contact_violation": contact_violation.astype(jp.float32),
                "ahac_contact_metric": contact_metric * horizon_active,
                "ahac_lambda": lambda_vec[scan_idx] * horizon_active,
                "ahac_force_ratio": force_ratio * horizon_active,
                "ahac_delta_ratio": delta_ratio * horizon_active,
                "ahac_grad_alive": grad_alive * horizon_active,
                "ahac_horizon_active": horizon_active,
                "ahac_horizon_trunc": horizon_trunc.astype(jp.float32),
                "vel_x": next_state.metrics["vel_x"] * horizon_active,
                "vel_y": next_state.metrics["vel_y"] * horizon_active,
                "yaw_rate": next_state.metrics["yaw_rate"] * horizon_active,
                "cmd_x": next_state.metrics["cmd_x"] * horizon_active,
                "cmd_y": next_state.metrics["cmd_y"] * horizon_active,
                "cmd_yaw": next_state.metrics["cmd_yaw"] * horizon_active,
                "height": next_state.metrics["height"] * horizon_active,
                "tilt": next_state.metrics["tilt"] * horizon_active,
                "foot_normal_FL": next_state.metrics["foot_normal_FL"] * horizon_active,
                "foot_normal_FR": next_state.metrics["foot_normal_FR"] * horizon_active,
                "foot_normal_RL": next_state.metrics["foot_normal_RL"] * horizon_active,
                "foot_normal_RR": next_state.metrics["foot_normal_RR"] * horizon_active,
            }

        env_state = jax.lax.stop_gradient(env_state)

        (
            _final_scan_state,
            _final_scan_foot_bump_ou,
            _final_grad_alive,
            _final_foot_forces,
            _grad_len,
            horizon_state,
            horizon_foot_bump_ou,
        ), traj = jax.lax.scan(
            rollout_step,
            (
                env_state,
                env_state.info["foot_bump_ou"],
                jp.array(1.0, dtype=jp.float64),
                env_state.info["foot_normal_forces"],
                jp.array(0.0, dtype=jp.float64),
                env_state,
                env_state.info["foot_bump_ou"],
            ),
            (
                action_noise,
                velocity_pushes,
                terrain_bump_innovations,
                jp.arange(unroll_length),
            ),
            length=unroll_length,
        )
        # The loss scan is padded to steps_max for JAX static shape support, but
        # the next rollout must resume from the adaptive horizon H, matching the
        # original AHAC implementation's variable short rollout semantics.
        final_state = jax.lax.stop_gradient(
            horizon_state.replace(
                info={**horizon_state.info, "foot_bump_ou": horizon_foot_bump_ou}
            )
        )

        bootstrap_obs = critic_norm.normalize(
            critic_norm_state, traj["bootstrap_critic_obs"]
        ).astype(jp.float32)
        bootstrap_v = critic.apply(target_critic_params, bootstrap_obs).squeeze()

        # Accumulate discounted returns, handling episode boundaries. Time-limit
        # truncations bootstrap from the pre-reset observation stored by env.step.
        def accum_return(carry, x):
            total, running, discount = carry
            r, done, terminal, v_next = x
            next_discount = discount * gamma
            running = running + discount * r
            trunc_bootstrap = (1.0 - terminal) * next_discount * v_next
            total = total + jp.where(done, running + trunc_bootstrap, 0.0)
            running = jp.where(done, 0.0, running)
            discount = jp.where(done, 1.0, next_discount)
            return (total, running, discount), None

        (total_ret, running, final_discount), _ = jax.lax.scan(
            accum_return,
            (0.0, 0.0, 1.0),
            (traj["reward"], traj["done"], traj["terminal"], bootstrap_v),
        )

        final_obs = critic_norm.normalize(
            critic_norm_state,
            env._get_critic_obs(final_state.data, final_state.info),
        ).astype(jp.float32)
        final_v = critic.apply(target_critic_params, final_obs).squeeze()
        horizon_completed = jp.any(traj["ahac_horizon_trunc"] > 0.0)
        final_bootstrap = jp.where(
            horizon_completed | (traj["done"][-1] > 0.0), 0.0, final_discount * final_v
        )

        total_ret = total_ret + running + final_bootstrap

        active_count = jp.maximum(jp.sum(traj["ahac_horizon_active"]), 1.0)
        ahac_penalty = ahac_penalty_weight * (
            jp.sum(traj["ahac_contact_excess"]) / active_count
        )

        return -total_ret / active_count + ahac_penalty, (traj, final_state)

    def critic_loss_from_data(
        critic_params,
        target_critic_params,
        critic_norm_state,
        traj_obs,
        traj_rewards,
        traj_dones,
        traj_terminals,
        traj_bootstrap_obs,
        final_obs,
        traj_mask,
    ):
        """
        Critic TD(lambda) loss using trajectory data collected by the actor.

        Implements Eq. 7 from the SHAC paper (Xu et al., ICLR 2022).
        All in float32 precision.
        """

        flat_obs = traj_obs.reshape(-1, env.critic_obs_dim)
        flat_bootstrap_obs = traj_bootstrap_obs.reshape(-1, env.critic_obs_dim)
        flat_obs_norm = critic_norm.normalize(critic_norm_state, flat_obs).astype(
            jp.float32
        )
        flat_bootstrap_obs_norm = critic_norm.normalize(
            critic_norm_state, flat_bootstrap_obs
        ).astype(jp.float32)
        final_obs_norm = critic_norm.normalize(critic_norm_state, final_obs).astype(
            jp.float32
        )

        # Predicted values V(s_t)
        values = critic.apply(critic_params, flat_obs_norm).squeeze()  # (H,)

        next_v = critic.apply(target_critic_params, flat_bootstrap_obs_norm).squeeze()
        final_v = critic.apply(target_critic_params, final_obs_norm).squeeze()  # scalar

        rewards = traj_rewards.reshape(-1).astype(jp.float32)  # (H,)
        dones = traj_dones.reshape(-1).astype(jp.float32)  # (H,)
        terminals = traj_terminals.reshape(-1).astype(jp.float32)  # (H,)
        mask = traj_mask.reshape(-1).astype(jp.float32)  # (H,)

        def scan_fn(g_next, inputs):
            r"""TD(lambda) backward scan."""
            r, done, terminal, v_next = inputs
            g_normal = r + gamma * (
                (1.0 - gae_lambda) * v_next + gae_lambda * g_next
            )  # Normal step
            g_trunc = r + gamma * v_next  # Time-limit trunc.
            g_term = r  # true term.
            g = jp.where(terminal, g_term, jp.where(done, g_trunc, g_normal))
            return g, g

        _, targets_reversed = jax.lax.scan(
            scan_fn,
            final_v,  # float32 scalar (determines the carry dtype)
            (rewards[::-1], dones[::-1], terminals[::-1], next_v[::-1]),
        )
        targets = targets_reversed[::-1]

        sqerr = jp.square(values - jax.lax.stop_gradient(targets)) * mask
        return jp.sum(sqerr) / jp.maximum(jp.sum(mask), 1.0)

    def update_normalizer_masked(norm_state, obs, mask):
        """Welford-style normalizer update using only active horizon rows."""
        flat_mask = mask.reshape(-1).astype(obs.dtype)
        obs = jp.where(jp.isfinite(obs), obs, norm_state.mean)
        count = jp.maximum(jp.sum(flat_mask), 1.0)
        weights = flat_mask[:, None]
        batch_mean = jp.sum(obs * weights, axis=0) / count
        centered = obs - batch_mean
        batch_var = jp.sum(jp.square(centered) * weights, axis=0) / count

        delta = batch_mean - norm_state.mean
        total = norm_state.count + count
        m_a = norm_state.var * norm_state.count
        m_b = batch_var * count
        m2 = m_a + m_b + jp.square(delta) * norm_state.count * count / total
        return norm_state.replace(mean=norm_state.mean + delta * count / total, var=m2 / total, count=total)

    @jax.jit
    def train_step(state: TrainState):
        key, noise_key, push_key, bump_key, diff_mask_key, _ = jax.random.split(
            state.key, 6
        )

        # Curriculum: difficulty=0 during grace, then ramp to 1
        difficulty = jp.clip(
            (state.step - _curriculum_grace_jax).astype(jp.float32)
            / _curriculum_steps_jax,
            0.0,
            1.0,
        )

        # Per-env difficulty: a fixed fraction of envs are held at difficulty=0
        # The mask is resampled every unroll
        zero_diff_mask = (
            jax.random.uniform(diff_mask_key, (num_envs,)) < zero_difficulty_frac
        )
        per_env_difficulty = jp.where(
            zero_diff_mask, jp.zeros(num_envs), jp.full((num_envs,), difficulty)
        )

        # Inject per-env difficulty into all non-zeroed-out env states
        updated_env_state = state.env_state.replace(
            info={**state.env_state.info, "difficulty": per_env_difficulty}
        )

        # Pre-sample all stochastic inputs (reparameterization)
        all_action_noise = jax.random.normal(
            noise_key, (num_envs, unroll_length, env.action_dim)
        )
        all_velocity_pushes = jax.random.uniform(
            push_key,
            (num_envs, unroll_length, 2),
            minval=_push_velocity_lo,
            maxval=_push_velocity_hi,
        )
        all_terrain_bump_innovations = jax.random.normal(
            bump_key, (num_envs, unroll_length, 4, 3)
        )
        all_randomization = (
            all_action_noise,
            all_velocity_pushes,
            all_terrain_bump_innovations,
        )

        # Linear noise schedule: start -> end over [0, total_steps]
        progress = jp.clip(state.step / total_steps, 0.0, 1.0)
        current_noise_std = action_noise_std_start + progress * (
            action_noise_std_end - action_noise_std_start
        )
        current_lambd_lr = ahac_lambd_lr
        if use_lr_decay:
            current_lambd_lr = ahac_lambd_lr + progress * (1e-5 - ahac_lambd_lr)
        current_horizon_steps = jp.clip(
            jp.rint(state.ahac_horizon).astype(jp.int32),
            ahac_steps_min,
            ahac_steps_max,
        )

        # Actor update
        actor_grad_fn = jax.value_and_grad(actor_loss, has_aux=True)
        (losses, (trajs, final_states)), grads = jax.vmap(
            actor_grad_fn, in_axes=(None, None, None, None, 0, 0, None, None, None)
        )(
            state.actor_params,
            state.target_critic_params,
            state.normalizer,
            state.critic_normalizer,
            updated_env_state,
            all_randomization,
            current_noise_std,
            state.ahac_horizon,
            state.ahac_lambda,
        )

        grads = jax.tree_util.tree_map(lambda g: jp.nanmean(g, axis=0), grads)
        grads = jax.tree_util.tree_map(
            lambda g: jp.where(jp.isfinite(g), g, 0.0), grads
        )

        actor_grad_norm = compute_grad_norm(grads)

        updates, new_actor_opt = actor_opt.update(grads, state.actor_opt)
        new_actor_params = optax.apply_updates(state.actor_params, updates)

        # Critic updates
        all_obs = trajs["critic_obs"]
        all_rewards = trajs["reward"]
        all_dones = trajs["done"]
        all_terminals = trajs["terminal"]
        all_bootstrap_obs = trajs["bootstrap_critic_obs"]
        all_active = trajs["ahac_horizon_active"]
        all_final_obs = jax.vmap(env._get_critic_obs)(
            final_states.data, final_states.info
        )

        def single_env_critic_loss(
            critic_params,
            target_critic_params,
            norm_state,
            obs,
            rewards,
            dones,
            terminals,
            bootstrap_obs,
            final_obs,
            active_mask,
        ):
            return critic_loss_from_data(
                critic_params,
                target_critic_params,
                norm_state,
                obs,
                rewards,
                dones,
                terminals,
                bootstrap_obs,
                final_obs,
                active_mask,
            )

        def critic_update_step(carry, _):
            c_params, c_opt_state = carry

            c_losses, c_grads = jax.vmap(
                jax.value_and_grad(single_env_critic_loss, argnums=0),
                in_axes=(None, None, None, 0, 0, 0, 0, 0, 0, 0),
            )(
                c_params,
                state.target_critic_params,
                state.critic_normalizer,
                all_obs,
                all_rewards,
                all_dones,
                all_terminals,
                all_bootstrap_obs,
                all_final_obs,
                all_active,
            )

            c_grads = jax.tree_util.tree_map(lambda g: jp.nanmean(g, axis=0), c_grads)
            c_grads = jax.tree_util.tree_map(
                lambda g: jp.where(jp.isfinite(g), g, 0.0), c_grads
            )

            c_updates, new_c_opt = critic_opt.update(c_grads, c_opt_state)
            new_c_params = optax.apply_updates(c_params, c_updates)

            return (new_c_params, new_c_opt), jp.mean(c_losses)

        (new_critic_params, new_critic_opt), critic_losses = jax.lax.scan(
            critic_update_step,
            (state.critic_params, state.critic_opt),
            None,
            length=critic_iterations,
        )

        # Original AHAC-style adaptive horizon update. Contact metrics are
        # averaged per rollout step, and H is adjusted through a dual variable.
        active_by_step = jp.sum(all_active, axis=0)
        active_step_mask = (active_by_step > 0.0).astype(jp.float64)
        step_contact_metric = jp.sum(
            trajs["ahac_contact_metric"] * all_active, axis=0
        ) / jp.maximum(active_by_step, 1.0)
        updated_lambda = state.ahac_lambda - current_lambd_lr * (
            ahac_horizon_contact_threshold - step_contact_metric
        )
        horizon_lambda = jp.where(
            active_step_mask > 0.0, updated_lambda, state.ahac_lambda
        )
        new_horizon = jp.clip(
            state.ahac_horizon
            + current_lambd_lr * jp.sum(horizon_lambda * active_step_mask),
            float(ahac_steps_min),
            float(ahac_steps_max),
        )
        # DiffRL's AHAC rebuilds its buffers after H changes and repeats the
        # first dual value across the new horizon. Keep the static JAX vector,
        # but broadcast the same scalar so the optimizer state follows that
        # original adaptive-horizon update rather than a steps_max-specific one.
        new_lambda = jp.full_like(state.ahac_lambda, horizon_lambda[0])

        # Soft target update
        new_target = optax.incremental_update(
            new_critic_params, state.target_critic_params, target_update_rate
        )

        # Update actor and critic normalizers from their own observation streams.
        flat_actor_obs = trajs["actor_obs"].reshape(-1, env.actor_frame_obs_dim)
        actor_frame_repeats = trajs["actor_obs"].shape[-1] // env.actor_frame_obs_dim
        flat_actor_active = jp.repeat(all_active.reshape(-1), actor_frame_repeats)
        flat_active = all_active.reshape(-1)
        new_actor_norm = update_normalizer_masked(
            state.normalizer, flat_actor_obs, flat_actor_active
        )

        flat_critic_obs = trajs["critic_obs"].reshape(-1, env.critic_obs_dim)
        new_critic_norm = update_normalizer_masked(
            state.critic_normalizer, flat_critic_obs, flat_active
        )

        new_state = state.replace(
            key=key,
            env_state=final_states,
            actor_params=new_actor_params,
            critic_params=new_critic_params,
            target_critic_params=new_target,
            normalizer=new_actor_norm,
            critic_normalizer=new_critic_norm,
            actor_opt=new_actor_opt,
            critic_opt=new_critic_opt,
            step=state.step + num_envs * current_horizon_steps,
            ahac_horizon=new_horizon,
            ahac_lambda=new_lambda,
        )

        active_count = jp.maximum(jp.sum(all_active), 1.0)

        def active_mean(x):
            return jp.sum(x * all_active) / active_count

        # Collect metrics
        metrics = {
            "reward": active_mean(trajs["reward"]),
            "vel_x": active_mean(trajs["vel_x"]),
            "vel_y": active_mean(trajs["vel_y"]),
            "yaw_rate": active_mean(trajs["yaw_rate"]),
            "cmd_x": active_mean(trajs["cmd_x"]),
            "cmd_y": active_mean(trajs["cmd_y"]),
            "cmd_yaw": active_mean(trajs["cmd_yaw"]),
            "contact": jp.mean(final_states.metrics["contact_force"]),
            "actor_grad": actor_grad_norm,
            "critic_loss": critic_losses[-1],
            "actor_loss": jp.mean(losses),
            "action_noise_current": current_noise_std,
            "track_vx": active_mean(jp.abs(trajs["vel_x"] - trajs["cmd_x"])),
            "track_vy": active_mean(jp.abs(trajs["vel_y"] - trajs["cmd_y"])),
            "track_yaw": active_mean(jp.abs(trajs["yaw_rate"] - trajs["cmd_yaw"])),
            "track_vx_sq": active_mean((trajs["vel_x"] - trajs["cmd_x"]) ** 2),
            "track_vy_sq": active_mean((trajs["vel_y"] - trajs["cmd_y"]) ** 2),
            "track_yaw_sq": active_mean(
                (trajs["yaw_rate"] - trajs["cmd_yaw"]) ** 2
            ),
            "rew_vel_x": jp.mean(final_states.metrics["rew_vel_x"]),
            "rew_vel_y": jp.mean(final_states.metrics["rew_vel_y"]),
            "rew_yaw": jp.mean(final_states.metrics["rew_yaw"]),
            "rew_vz": jp.mean(final_states.metrics["rew_vz"]),
            "pen_rate": jp.mean(final_states.metrics["pen_rate"]),
            "height": active_mean(trajs["height"]),
            "tilt": active_mean(trajs["tilt"]),
            "difficulty": difficulty,
            "ahac_horizon": new_horizon,
            "ahac_lambda_mean": jp.sum(new_lambda * active_step_mask)
            / jp.maximum(jp.sum(active_step_mask), 1.0),
            "ahac_contact_metric": active_mean(trajs["ahac_contact_metric"]),
            "ahac_contact_excess": active_mean(trajs["ahac_contact_excess"]),
            "ahac_contact_violation": active_mean(trajs["ahac_contact_violation"]),
            "ahac_force_ratio": active_mean(trajs["ahac_force_ratio"]),
            "ahac_delta_ratio": active_mean(trajs["ahac_delta_ratio"]),
            "ahac_grad_horizon": jp.mean(
                jp.sum(trajs["ahac_grad_alive"] * all_active, axis=1)
            ),
            "ahac_active_horizon": jp.mean(jp.sum(all_active, axis=1)),
            "foot_normal_FL": active_mean(trajs["foot_normal_FL"]),
            "foot_normal_FR": active_mean(trajs["foot_normal_FR"]),
            "foot_normal_RL": active_mean(trajs["foot_normal_RL"]),
            "foot_normal_RR": active_mean(trajs["foot_normal_RR"]),
        }

        return new_state, metrics

    if resumed_state is not None:
        # Restore learned params and optimizer states
        print(
            f"Restoring learned parameters and optimizer states from step {resumed_step}"
        )
        resumed_horizon = getattr(resumed_state, "ahac_horizon", None)
        if resumed_horizon is None:
            resumed_horizon = jp.array(ahac_initial_horizon, dtype=jp.float64)
        resumed_lambda = getattr(resumed_state, "ahac_lambda", None)
        if resumed_lambda is None or tuple(np.shape(resumed_lambda)) != (ahac_steps_max,):
            resumed_lambda = jp.zeros((ahac_steps_max,), dtype=jp.float64)
        state = TrainState(
            key=key,
            env_state=env_state,
            actor_params=resumed_state.actor_params,
            critic_params=resumed_state.critic_params,
            target_critic_params=resumed_state.target_critic_params,
            normalizer=resumed_state.normalizer,
            critic_normalizer=resumed_state.critic_normalizer,
            actor_opt=actor_opt_state
            if ahac_reset_optimizer_on_resume
            else resumed_state.actor_opt,
            critic_opt=critic_opt_state
            if ahac_reset_optimizer_on_resume
            else resumed_state.critic_opt,
            step=resumed_step,
            ahac_horizon=resumed_horizon,
            ahac_lambda=resumed_lambda,
        )
        if ahac_reset_optimizer_on_resume:
            print("Reset optimizer states for AHAC fine-tuning")
    else:
        state = TrainState(
            key=key,
            env_state=env_state,
            actor_params=actor_params,
            critic_params=critic_params,
            target_critic_params=target_critic_params,
            normalizer=actor_normalizer,
            critic_normalizer=critic_normalizer,
            actor_opt=actor_opt_state,
            critic_opt=critic_opt_state,
            step=0,
            ahac_horizon=jp.array(ahac_initial_horizon, dtype=jp.float64),
            ahac_lambda=jp.zeros((ahac_steps_max,), dtype=jp.float64),
        )

    print("Compiling...")
    start_comp_time = time.perf_counter()
    warmup_state, _ = train_step(state)
    compile_time = time.perf_counter() - start_comp_time
    print(f"Compilation took {compile_time:.1f}s")

    # Warm up normalizer from the compilation step.
    state = state.replace(
        normalizer=warmup_state.normalizer,
        critic_normalizer=warmup_state.critic_normalizer,
    )

    print("Training...")

    if diagnose:
        header = (
            f"{'Step':>7} | {'Rew':>7} | {'TrkVx':>7} | {'TrkVy':>7} | "
            f"{'TrkYaw':>7} | {'RewVx':>7} | {'RewVy':>7} | {'RewYaw':>7} | "
            f"{'PenRate':>7} | {'Height':>7} | "
            f"{'Tilt':>7} | {'Diff':>5} | {'AH':>5} | {'CM':>5} | {'CV':>5} | "
            f"{'AGrad':>7} | {'Status'}"
        )
    else:
        header = (
            f"{'Step':>7} | {'Rew':>7} | {'TrkVx':>7} | {'TrkVy':>7} | "
            f"{'TrkYaw':>7} | {'AH':>5} | {'CM':>5} | {'CV':>5} | {'AGrad':>7} | {'CLoss':>7} | "
            f"{'Diff':>5} | {'Status':>8}"
        )
    print("=" * len(header))
    print(header)
    print("=" * len(header))

    start = time.time()
    log = []
    diag_log = []
    last_checkpoint_step = int(resumed_step)

    min_steps_per_iter = num_envs * ahac_steps_min
    remaining_steps = max(int(total_steps) - int(resumed_step), 0)
    total_iters = max(1, int(np.ceil(remaining_steps / min_steps_per_iter)))

    for i in range(total_iters):
        state, metrics = train_step(state)

        if i % 10 == 0:
            jax.block_until_ready(state.step)
            step_value = int(jax.device_get(state.step))

            vel_x = float(metrics["vel_x"])
            vel_y = float(metrics["vel_y"])
            yaw_rate = float(metrics["yaw_rate"])
            reward = float(metrics["reward"])

            # Per-env tracking errors (proper: mean of |vel-cmd| per env)
            cmd_x = float(metrics["cmd_x"])
            cmd_y = float(metrics["cmd_y"])
            cmd_yaw = float(metrics["cmd_yaw"])
            trk_vx = float(metrics["track_vx"])
            trk_vy = float(metrics["track_vy"])
            trk_yaw = float(metrics["track_yaw"])
            diff = float(metrics["difficulty"])
            ahac_horizon = float(metrics["ahac_horizon"])
            ahac_grad_horizon = float(metrics["ahac_grad_horizon"])
            ahac_contact_metric = float(metrics["ahac_contact_metric"])
            ahac_contact_violation = float(metrics["ahac_contact_violation"])
            max_err = max(trk_vx, trk_vy, trk_yaw)

            if max_err < 0.1:
                status = "TRACK :D"
            elif max_err < 0.25:
                status = "CLOSE :)"
            elif max_err < 0.4:
                status = "TRYING"
            else:
                status = "LEARN"

            if diagnose:
                print(
                    f"{step_value:7d} | {reward:7.2f} | {trk_vx:7.3f} | {trk_vy:7.3f} | "
                    f"{trk_yaw:7.3f} | "
                    f"{metrics['rew_vel_x']:7.2f} | "
                    f"{metrics['rew_vel_y']:7.2f} | "
                    f"{metrics['rew_yaw']:7.2f} | "
                    f"{metrics['pen_rate']:7.3f} | {metrics['height']:7.3f} | "
                    f"{metrics['tilt']:7.2f} | {diff:5.2f} | "
                    f"{ahac_horizon:5.1f} | {ahac_contact_metric:5.3f} | {ahac_contact_violation:5.3f} | "
                    f"{metrics['actor_grad']:7.1f} | {status}"
                )

                diag_log.append(
                    {
                        "step": step_value,
                        "reward": reward,
                        "difficulty": diff,
                        "vel_x": vel_x,
                        "vel_y": vel_y,
                        "yaw_rate": yaw_rate,
                        "cmd_x": cmd_x,
                        "cmd_y": cmd_y,
                        "cmd_yaw": cmd_yaw,
                        "track_vx": trk_vx,
                        "track_vy": trk_vy,
                        "track_yaw": trk_yaw,
                        "rew_vel_x": float(metrics["rew_vel_x"]),
                        "rew_vel_y": float(metrics["rew_vel_y"]),
                        "rew_yaw": float(metrics["rew_yaw"]),
                        "pen_rate": float(metrics["pen_rate"]),
                        "height": float(metrics["height"]),
                        "tilt": float(metrics["tilt"]),
                        "ahac_contact_excess": float(metrics["ahac_contact_excess"]),
                        "ahac_contact_violation": ahac_contact_violation,
                        "ahac_force_ratio": float(metrics["ahac_force_ratio"]),
                        "ahac_delta_ratio": float(metrics["ahac_delta_ratio"]),
                        "ahac_contact_metric": float(metrics["ahac_contact_metric"]),
                        "ahac_lambda_mean": float(metrics["ahac_lambda_mean"]),
                        "ahac_horizon": ahac_horizon,
                        "ahac_grad_horizon": ahac_grad_horizon,
                        "ahac_gradient_steps": ahac_grad_horizon,
                        "actor_grad": float(metrics["actor_grad"]),
                        "critic_loss": float(metrics["critic_loss"]),
                    }
                )
            else:
                print(
                    f"{step_value:7d} | {reward:7.3f} | {trk_vx:7.3f} | {trk_vy:7.3f} | "
                    f"{trk_yaw:7.3f} | "
                    f"{ahac_horizon:5.1f} | {ahac_contact_metric:5.3f} | {ahac_contact_violation:5.3f} | "
                    f"{metrics['actor_grad']:7.2f} | {metrics['critic_loss']:7.4f} | "
                    f"{diff:5.2f} | {status}"
                )

            if DEBUG_FOOT_CONTACTS:
                print(
                    f"         foot GRF (N):  "
                    f"FL={float(metrics['foot_normal_FL']):7.2f}  "
                    f"FR={float(metrics['foot_normal_FR']):7.2f}  "
                    f"RL={float(metrics['foot_normal_RL']):7.2f}  "
                    f"RR={float(metrics['foot_normal_RR']):7.2f}"
                )

            log.append(
                [
                    step_value,
                    reward,
                    vel_x,
                    vel_y,
                    yaw_rate,
                    cmd_x,
                    cmd_y,
                    cmd_yaw,
                    float(metrics["actor_loss"]),
                    float(metrics["contact"]),
                    float(metrics["actor_grad"]),
                    float(metrics["critic_loss"]),
                    float(metrics["track_vx_sq"]),
                    float(metrics["track_vy_sq"]),
                    float(metrics["track_yaw_sq"]),
                    diff,
                    ahac_horizon,
                    float(metrics["ahac_grad_horizon"]),
                    float(metrics["ahac_lambda_mean"]),
                    float(metrics["ahac_contact_metric"]),
                ]
            )

            # Save best policy
            if reward > best_reward and step_value > 5000:
                best_reward = reward
                with open(f"{save_dir}/policy_best.pkl", "wb") as f:
                    pickle.dump(slim_checkpoint_state(state), f)
                print(f"  >> New best! Reward: {best_reward:.3f}")

            tracking_score = float(
                metrics["track_vx_sq"] + metrics["track_vy_sq"] + metrics["track_yaw_sq"]
            )
            if tracking_score < best_tracking and step_value > 5000:
                best_tracking = tracking_score
                with open(f"{save_dir}/policy_best_tracking.pkl", "wb") as f:
                    pickle.dump(slim_checkpoint_state(state), f)
                print(f"  >> New best tracking! Error sum: {best_tracking:.3f}")

            # Periodic checkpoint
            if step_value - last_checkpoint_step >= checkpoint_interval:
                ckpt_path = os.path.join(save_dir, "checkpoint_latest.pkl")
                with open(ckpt_path, "wb") as f:
                    pickle.dump(slim_checkpoint_state(state), f)
                last_checkpoint_step = step_value
                print(f"  >> Checkpoint saved at step {step_value}")

            if step_value >= total_steps:
                break

    # Save final state and logs
    with open(f"{save_dir}/policy_final.pkl", "wb") as f:
        pickle.dump(slim_checkpoint_state(state), f)
    np.save(f"{save_dir}/log.npy", np.array(log))

    if diagnose and diag_log:
        with open(f"{save_dir}/diag_log.json", "w") as f:
            json.dump(diag_log, f, indent=2)
        print(f"Diagnostic log saved to {save_dir}/diag_log.json")

    elapsed = time.time() - start
    cmd_str = (
        f"vx=[{cmd_vel_x_range[0]:.2f},{cmd_vel_x_range[1]:.2f}], "
        f"vy=[{cmd_vel_y_range[0]:.2f},{cmd_vel_y_range[1]:.2f}], "
        f"yaw=[{cmd_yaw_rate_range[0]:.2f},{cmd_yaw_rate_range[1]:.2f}] "
        f"| zero_prob={cmd_zero_prob} stand_prob={cmd_stand_prob} "
        f"interval={cmd_ctrl_interval_range}"
    )
    print("=" * (160 if diagnose else 120))
    print(f"Training complete in {elapsed:.1f}s (compilation: {compile_time:.1f}s)")
    print(f"Command ranges: {cmd_str}")
    print("=" * 100)
    print(f"Training complete in {elapsed:.1f}s (compile: {compile_time:.1f}s)")
    print(
        f"Curriculum: grace={curriculum_grace}, curriculum_steps={curriculum_steps}, "
        f"terrain={'ON' if terrain else 'OFF'}, terrain_flat_prob={terrain_flat_prob}, "
        f"terrain_slope_max={terrain_slope_max} deg, "
        f"terrain_bump_std={terrain_bump_std if terrain else 0.0}"
    )
    print(f"Best reward: {best_reward:.3f}")
    print(f"Best tracking error sum: {best_tracking:.3f}")

    # Update hyperparameters with the final result.
    hparams["best_reward"] = best_reward
    hparams["best_tracking"] = best_tracking
    hparams["final_ahac_horizon"] = float(jax.device_get(state.ahac_horizon))
    hparams["final_ahac_lambda_mean"] = float(
        jax.device_get(jp.mean(state.ahac_lambda))
    )
    with open(f"{save_dir}/hparams.json", "w") as f:
        json.dump(hparams, f, indent=2)

    return state, save_dir
