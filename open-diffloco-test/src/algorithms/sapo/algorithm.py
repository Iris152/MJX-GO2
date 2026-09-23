"""Soft Analytic Policy Optimization (SAPO) for MJX Go2 locomotion.

This is an on-policy, short-horizon actor-critic implementation. It follows
the SAPO design choices from the paper and the Rewarped/Mineral reference:

* reparameterized state-dependent squashed-Normal actor;
* entropy-augmented differentiable actor return;
* entropy-augmented TD(lambda) critic targets;
* two independent critics, with mean values for the actor and min values for
  critic targets;
* no target critic network;
* automatic entropy temperature.

The Go2 environment, curriculum, domain randomization, and MJX scan are
shared with the existing SHAC implementation.
"""

import json
import os
import pickle
import time
from datetime import datetime

import jax
import jax.numpy as jp
import numpy as np
import optax

from src.core.data_structures import Normalizer, TrainState, slim_checkpoint_state
from src.core.networks import SAPOActor, SAPOCritic
from src.core.utils import compute_grad_norm
from src.envs.go2.environment import Go2Env
from src.envs.go2.terrain import differentiated_ou_foot_forces


def load_checkpoint(path: str):
    """Load a SAPO checkpoint and its run hyperparameters."""
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

    hparams = None
    hparams_path = os.path.join(os.path.dirname(path), "hparams.json")
    if os.path.exists(hparams_path):
        with open(hparams_path) as f:
            hparams = json.load(f)

    return state, hparams, int(state.step)


def _finite_tree(tree):
    """Replace invalid gradient leaves without changing pytree structure."""
    return jax.tree_util.tree_map(
        lambda x: jp.where(jp.isfinite(x), x, jp.zeros_like(x)), tree
    )


def _stop_gradient_tree(tree):
    return jax.tree_util.tree_map(jax.lax.stop_gradient, tree)


def _clip_batched_tree_by_norm(tree, max_norm: float):
    """Clip each environment's gradient before the batch reduction.

    Differentiable contact dynamics can produce a rare, very large gradient
    for one environment. The reference algorithm clips the optimizer update
    after reduction; this additional robustification prevents one contact
    outlier from determining the direction of the reduced gradient.
    """
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return tree
    batch_size = leaves[0].shape[0]
    norm_sq = jp.zeros((batch_size,), dtype=jp.float32)
    for leaf in leaves:
        finite_leaf = jp.where(jp.isfinite(leaf), leaf, jp.zeros_like(leaf))
        axes = tuple(range(1, finite_leaf.ndim))
        norm_sq = norm_sq + jp.sum(jp.square(finite_leaf), axis=axes)
    norms = jp.sqrt(norm_sq + 1e-12)
    scales = jp.minimum(1.0, max_norm / norms)

    def scale_leaf(leaf):
        shape = (batch_size,) + (1,) * (leaf.ndim - 1)
        return leaf * scales.reshape(shape)

    return jax.tree_util.tree_map(scale_leaf, tree)


def _soft_td_lambda_targets(
    rewards,
    done_mask,
    terminals,
    next_values,
    gamma,
    gae_lambda,
):
    """Compute Mineral/SAPO soft TD(lambda) target values.

    This matches the reference implementation's Ai/Bi recurrence. ``done_mask``
    marks return segment boundaries, including the artificial boundary at the
    end of the H-step actor rollout. True terminals are represented by zeroing
    the corresponding bootstrap value; time-limit and horizon truncations keep
    their value bootstrap.
    """
    rewards = rewards.astype(jp.float32)
    done_mask = done_mask.astype(jp.float32)
    terminals = terminals.astype(jp.float32)
    next_values = jp.where(
        terminals.astype(bool),
        jp.zeros_like(next_values),
        next_values.astype(jp.float32),
    )

    def scan_fn(carry, inputs):
        ai, bi, lam_trace = carry
        reward_t, done_t, next_value_t = inputs

        lam_trace = lam_trace * gae_lambda * (1.0 - done_t) + done_t
        adjusted_reward = (1.0 - lam_trace) / (1.0 - gae_lambda) * reward_t
        ai = (1.0 - done_t) * (
            gae_lambda * gamma * ai + gamma * next_value_t + adjusted_reward
        )
        bi = gamma * (
            next_value_t * done_t + bi * (1.0 - done_t)
        ) + reward_t
        target = (1.0 - gae_lambda) * ai + lam_trace * bi
        return (ai, bi, lam_trace), target

    init = (
        jp.zeros_like(rewards[0]),
        jp.zeros_like(rewards[0]),
        jp.ones_like(rewards[0]),
    )
    _, targets_reversed = jax.lax.scan(
        scan_fn,
        init,
        (rewards[::-1], done_mask[::-1], next_values[::-1]),
    )
    return jax.lax.stop_gradient(targets_reversed[::-1])


def _sapo_sample(
    actor,
    params,
    obs,
    epsilon,
    min_log_std,
    max_log_std,
    entropy_epsilon=None,
):
    """Sample tanh(N(mu, sigma)) and return its stable log probability.

    ``epsilon`` is supplied by the caller, so ``action`` remains fully
    reparameterized through the MJX dynamics and reward computation.
    """
    mu, raw_log_std = actor.apply(params, obs)
    log_std = jp.clip(raw_log_std, min_log_std, max_log_std)
    std = jp.exp(log_std)
    action_z = mu + std * epsilon
    action = jp.tanh(action_z)
    if entropy_epsilon is None:
        entropy_epsilon = epsilon

    log_two_pi = jp.log(2.0 * jp.pi)
    entropy_z = mu + std * entropy_epsilon
    base_log_prob = -0.5 * (
        jp.square((entropy_z - mu) / (std + 1e-8))
        + 2.0 * log_std
        + log_two_pi
    )
    # Stable log(1 - tanh(z)^2), avoiding an inverse tanh and boundary NaNs.
    log_tanh_det = 2.0 * (
        jp.log(2.0) - entropy_z - jax.nn.softplus(-2.0 * entropy_z)
    )
    log_prob = jp.sum(base_log_prob - log_tanh_det, axis=-1)
    raw_entropy = -log_prob
    return action, raw_entropy, mu, log_std


def train(
    # General
    total_steps: int = 100_000,
    unroll_length: int = 32,
    num_envs: int = 256,
    actor_lr: float = 2e-3,
    critic_lr: float = 5e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    critic_iterations: int = 16,
    use_lr_decay: bool = False,
    xml_path: str = "src/envs/go2/models/scene_mjx.xml",
    action_scale: float = 0.5,
    default_joint_pose: tuple = None,
    default_base_height: float = None,
    target_base_height: float = 0.3,
    # Commands
    cmd_vel_x_range: tuple = (-1.5, 1.5),
    cmd_vel_y_range: tuple = (-1.0, 1.0),
    cmd_yaw_rate_range: tuple = (-1.5, 1.5),
    cmd_zero_prob: tuple = (0.1, 0.7, 0.5),
    cmd_stand_prob: float = 0.0,
    cmd_ctrl_interval_range: tuple = (60, 140),
    track_weights: tuple = None,
    # Randomization
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
    # Curriculum
    zero_difficulty_frac: float = 0.0,
    curriculum_grace: int = None,
    curriculum_steps: int = None,
    # SAPO
    sapo_init_alpha: float = 1.0,
    sapo_target_entropy_scalar: float = 0.5,
    sapo_alpha_lr: float = 5e-3,
    sapo_alpha_betas: tuple = (0.7, 0.95),
    sapo_log_std_init: float = -1.0,
    sapo_min_log_std: float = -5.0,
    sapo_max_log_std: float = 2.0,
    sapo_actor_hidden: tuple = (128, 64, 32),
    sapo_critic_hidden: tuple = (64, 64),
    sapo_max_grad_norm: float = 0.5,
    sapo_per_env_grad_clip: float = 10.0,
    sapo_weight_decay: float = 0.0,
    # Misc.
    diagnose: bool = False,
    seed: int = 0,
    resume_from: str = None,
    checkpoint_interval: int = 100_000,
    max_episode_length: int = 5000,
    actor_history_len: int = 10,
    env_variant: str = "blind_nolinvel_nokinref",
):
    """Train Go2 with maximum-entropy first-order model-based RL."""
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if unroll_length <= 0 or num_envs <= 0:
        raise ValueError("unroll_length and num_envs must be positive")
    if not (0.0 < sapo_target_entropy_scalar):
        raise ValueError("sapo_target_entropy_scalar must be positive")
    if sapo_init_alpha <= 0.0:
        raise ValueError("sapo_init_alpha must be positive")
    if sapo_min_log_std >= sapo_max_log_std:
        raise ValueError("sapo_min_log_std must be smaller than sapo_max_log_std")
    if sapo_per_env_grad_clip <= 0.0:
        raise ValueError("sapo_per_env_grad_clip must be positive")

    resumed_state = None
    resumed_hparams = None
    resumed_step = 0
    if resume_from:
        resumed_state, resumed_hparams, resumed_step = load_checkpoint(resume_from)
        if resumed_hparams and resumed_hparams.get("algorithm") not in (None, "sapo"):
            raise ValueError("SAPO can only resume from a SAPO checkpoint")
        if resumed_hparams:
            print(f"Resuming SAPO from step {resumed_step}")
            action_scale = resumed_hparams.get("action_scale", action_scale)
            xml_path = resumed_hparams.get("xml_path", xml_path)
            env_variant = resumed_hparams.get(
                "env_variant", "blind_nolinvel_nokinref"
            )
            if "default_joint_pose" in resumed_hparams:
                default_joint_pose = resumed_hparams["default_joint_pose"]
            if "default_base_height" in resumed_hparams:
                default_base_height = resumed_hparams["default_base_height"]
            if "target_base_height" in resumed_hparams:
                target_base_height = resumed_hparams["target_base_height"]
            for name in [
                "cmd_vel_x_range",
                "cmd_vel_y_range",
                "cmd_yaw_rate_range",
                "cmd_zero_prob",
                "cmd_ctrl_interval_range",
                "kp_range",
                "kd_range",
                "com_offset_range",
                "push_velocity_range",
                "sapo_alpha_betas",
                "sapo_actor_hidden",
                "sapo_critic_hidden",
            ]:
                if name in resumed_hparams:
                    value = resumed_hparams[name]
                    if name.endswith("_range") or name.endswith("_prob"):
                        value = tuple(value)
                    elif name.endswith("_betas") or name.endswith("_hidden"):
                        value = tuple(value)
                    locals()[name] = value
            # Explicit assignments are clearer than relying on locals() for
            # values that are used below.
            cmd_vel_x_range = tuple(
                resumed_hparams.get("cmd_vel_x_range", cmd_vel_x_range)
            )
            cmd_vel_y_range = tuple(
                resumed_hparams.get("cmd_vel_y_range", cmd_vel_y_range)
            )
            cmd_yaw_rate_range = tuple(
                resumed_hparams.get("cmd_yaw_rate_range", cmd_yaw_rate_range)
            )
            cmd_zero_prob = tuple(
                resumed_hparams.get("cmd_zero_prob", cmd_zero_prob)
            )
            cmd_ctrl_interval_range = tuple(
                resumed_hparams.get(
                    "cmd_ctrl_interval_range", cmd_ctrl_interval_range
                )
            )
            kp_range = tuple(resumed_hparams.get("kp_range", kp_range))
            kd_range = tuple(resumed_hparams.get("kd_range", kd_range))
            com_offset_range = tuple(
                resumed_hparams.get("com_offset_range", com_offset_range)
            )
            push_velocity_range = tuple(
                resumed_hparams.get("push_velocity_range", push_velocity_range)
            )
            max_episode_length = resumed_hparams.get(
                "max_episode_length", max_episode_length
            )
            actor_history_len = resumed_hparams.get(
                "actor_history_len", actor_history_len
            )
            sapo_init_alpha = resumed_hparams.get(
                "sapo_init_alpha", sapo_init_alpha
            )
            sapo_target_entropy_scalar = resumed_hparams.get(
                "sapo_target_entropy_scalar", sapo_target_entropy_scalar
            )
            sapo_alpha_lr = resumed_hparams.get("sapo_alpha_lr", sapo_alpha_lr)
            sapo_alpha_betas = tuple(
                resumed_hparams.get("sapo_alpha_betas", sapo_alpha_betas)
            )
            sapo_log_std_init = resumed_hparams.get(
                "sapo_log_std_init", sapo_log_std_init
            )
            sapo_min_log_std = resumed_hparams.get(
                "sapo_min_log_std", sapo_min_log_std
            )
            sapo_max_log_std = resumed_hparams.get(
                "sapo_max_log_std", sapo_max_log_std
            )
            sapo_actor_hidden = tuple(
                resumed_hparams.get("sapo_actor_hidden", sapo_actor_hidden)
            )
            sapo_critic_hidden = tuple(
                resumed_hparams.get("sapo_critic_hidden", sapo_critic_hidden)
            )
            sapo_max_grad_norm = resumed_hparams.get(
                "sapo_max_grad_norm", sapo_max_grad_norm
            )
            sapo_per_env_grad_clip = resumed_hparams.get(
                "sapo_per_env_grad_clip", sapo_per_env_grad_clip
            )
            sapo_weight_decay = resumed_hparams.get(
                "sapo_weight_decay", sapo_weight_decay
            )
        else:
            env_variant = "blind_nolinvel_nokinref"

    if curriculum_grace is None:
        curriculum_grace = total_steps // 10
    if curriculum_steps is None:
        curriculum_steps = int(total_steps * 0.8)
    curriculum_steps = max(int(curriculum_steps), 1)
    curriculum_grace_jax = jp.array(curriculum_grace, dtype=jp.int32)
    curriculum_steps_jax = jp.array(curriculum_steps, dtype=jp.float32)

    env_kwargs = dict(
        variant=env_variant,
        xml_path=xml_path,
        action_scale=action_scale,
        default_joint_pose=default_joint_pose,
        default_base_height=default_base_height,
        target_base_height=target_base_height,
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
    if track_weights is not None and env_variant == "blind_nolinvel_nokinref":
        env_kwargs["track_weights"] = tuple(track_weights)
    env = Go2Env(**env_kwargs)
    actor_norm = Normalizer(env.actor_frame_obs_dim)
    critic_norm = Normalizer(env.critic_obs_dim)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"training_runs/sapo_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)

    target_entropy = -float(env.action_dim) * float(sapo_target_entropy_scalar)
    target_entropy_abs = abs(target_entropy)
    print(f"Algorithm: SAPO, Save dir: {save_dir}")
    print(
        f"  Actor: hidden={tuple(sapo_actor_hidden)}, "
        f"log_std=[{sapo_min_log_std:.1f},{sapo_max_log_std:.1f}], "
        f"init={sapo_log_std_init:.2f}"
    )
    print(
        f"  Soft value: double critic hidden={tuple(sapo_critic_hidden)}, "
        f"no target critic, H={unroll_length}"
    )
    print(
        f"  Entropy: alpha0={sapo_init_alpha:.4f}, "
        f"target_entropy={target_entropy:.2f} "
        f"(Shannon magnitude {target_entropy_abs:.2f}), "
        f"alpha_lr={sapo_alpha_lr:.2e}"
    )
    print(
        f"Domain randomization: friction={friction_range}, mass={mass_range}, "
        f"kp={kp_range}, kd={kd_range}, com_offset=+/-{com_offset_range}, "
        f"velocity_push={push_velocity_range} every {push_interval_s}s"
    )
    print(
        f"Curriculum: grace={curriculum_grace}, curriculum={curriculum_steps} "
        f"steps, terrain={'ON' if terrain else 'OFF'}"
    )

    best_reward = (
        resumed_hparams.get("best_reward", -np.inf)
        if resumed_hparams
        else -np.inf
    )
    best_tracking = (
        resumed_hparams.get("best_tracking", np.inf)
        if resumed_hparams
        else np.inf
    )

    hparams = {
        "algorithm": "sapo",
        "total_steps": total_steps,
        "unroll_length": unroll_length,
        "num_envs": num_envs,
        "actor_lr": actor_lr,
        "critic_lr": critic_lr,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "critic_iterations": critic_iterations,
        "xml_path": xml_path,
        "action_scale": action_scale,
        "default_joint_pose": list(default_joint_pose)
        if default_joint_pose is not None
        else None,
        "default_base_height": default_base_height,
        "target_base_height": target_base_height,
        "cmd_vel_x_range": list(cmd_vel_x_range),
        "cmd_vel_y_range": list(cmd_vel_y_range),
        "cmd_yaw_rate_range": list(cmd_yaw_rate_range),
        "cmd_zero_prob": list(cmd_zero_prob),
        "cmd_stand_prob": cmd_stand_prob,
        "cmd_ctrl_interval_range": list(cmd_ctrl_interval_range),
        "track_weights": list(track_weights) if track_weights is not None else None,
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
        "seed": seed,
        "best_reward": best_reward,
        "best_tracking": best_tracking,
        "max_episode_length": max_episode_length,
        "actor_history_len": actor_history_len,
        "env_variant": env_variant,
        "sapo_init_alpha": sapo_init_alpha,
        "sapo_target_entropy_scalar": sapo_target_entropy_scalar,
        "sapo_target_entropy": target_entropy,
        "sapo_alpha_lr": sapo_alpha_lr,
        "sapo_alpha_betas": list(sapo_alpha_betas),
        "sapo_log_std_init": sapo_log_std_init,
        "sapo_min_log_std": sapo_min_log_std,
        "sapo_max_log_std": sapo_max_log_std,
        "sapo_actor_hidden": list(sapo_actor_hidden),
        "sapo_critic_hidden": list(sapo_critic_hidden),
        "sapo_max_grad_norm": sapo_max_grad_norm,
        "sapo_per_env_grad_clip": sapo_per_env_grad_clip,
        "sapo_weight_decay": sapo_weight_decay,
        "sapo_entropy_normalization": "(H + abs(target)) / (2 * abs(target))",
        "sapo_distribution_entropy_sample": True,
        "sapo_td_lambda": "mineral_aibi",
        "sapo_external_action_noise": False,
        "sapo_critic_target_network": False,
    }
    with open(f"{save_dir}/hparams.json", "w") as f:
        json.dump(hparams, f, indent=2)

    key = jax.random.PRNGKey(seed)
    key, k1, k2, k3 = jax.random.split(key, 4)

    actor = SAPOActor(
        env.action_dim,
        hidden=tuple(sapo_actor_hidden),
        log_std_init=sapo_log_std_init,
    )
    critic = SAPOCritic(hidden=tuple(sapo_critic_hidden))
    actor_dummy = jp.zeros((1, env.actor_obs_dim), dtype=jp.float32)
    critic_dummy = jp.zeros((1, env.critic_obs_dim), dtype=jp.float32)

    actor_params = actor.init(k1, actor_dummy)
    critic_params_1 = critic.init(k2, critic_dummy)
    critic_params_2 = critic.init(k3, critic_dummy)
    critic_params = (critic_params_1, critic_params_2)

    actor_normalizer = actor_norm.init()
    critic_normalizer = critic_norm.init()

    if use_lr_decay:
        total_iters = max(total_steps // (num_envs * unroll_length), 1)
        lr_floor = 0.62
        actor_schedule = optax.linear_schedule(
            init_value=actor_lr,
            end_value=actor_lr * lr_floor,
            transition_steps=total_iters,
        )
        critic_schedule = optax.linear_schedule(
            init_value=critic_lr,
            end_value=critic_lr * lr_floor,
            transition_steps=total_iters * critic_iterations,
        )
        print(
            f"LR decay: actor {actor_lr:.2e}->{actor_lr * lr_floor:.2e}, "
            f"critic {critic_lr:.2e}->{critic_lr * lr_floor:.2e}"
        )
    else:
        actor_schedule = actor_lr
        critic_schedule = critic_lr

    actor_opt = optax.chain(
        optax.clip_by_global_norm(sapo_max_grad_norm),
        optax.adamw(
            actor_schedule,
            b1=sapo_alpha_betas[0],
            b2=sapo_alpha_betas[1],
            weight_decay=sapo_weight_decay,
        ),
    )
    critic_opt = optax.chain(
        optax.clip_by_global_norm(sapo_max_grad_norm),
        optax.adamw(
            critic_schedule,
            b1=sapo_alpha_betas[0],
            b2=sapo_alpha_betas[1],
            weight_decay=sapo_weight_decay,
        ),
    )
    alpha_opt = optax.chain(
        optax.clip_by_global_norm(sapo_max_grad_norm),
        optax.adamw(
            sapo_alpha_lr,
            b1=sapo_alpha_betas[0],
            b2=sapo_alpha_betas[1],
            weight_decay=0.0,
        ),
    )

    actor_opt_state = actor_opt.init(actor_params)
    critic_opt_state = critic_opt.init(critic_params)
    initial_log_alpha = jp.array(np.log(sapo_init_alpha), dtype=jp.float32)
    alpha_opt_state = alpha_opt.init(initial_log_alpha)

    env_keys = jax.random.split(k3, num_envs)
    env_state = jax.vmap(env.reset)(env_keys, jp.zeros(num_envs))

    push_interval_steps = max(int(round(push_interval_s / env.dt)), 1)
    push_velocity_lo = jp.array(push_velocity_range[0], dtype=jp.float64)
    push_velocity_hi = jp.array(push_velocity_range[1], dtype=jp.float64)
    foot_body_ids = env._foot_body_ids
    nominal_weight = env.nominal_total_mass * env.base_gravity_mag
    terrain_bump_std = terrain_bump_std if terrain else 0.0

    def actor_loss(
        actor_params_,
        critic_params_,
        actor_norm_state,
        critic_norm_state,
        env_state_,
        randomization,
        log_alpha_,
    ):
        """Differentiate a soft H-step return through MJX."""
        (
            action_eps,
            entropy_eps,
            velocity_pushes,
            terrain_bump_innovations,
        ) = randomization
        alpha = jp.exp(jax.lax.stop_gradient(log_alpha_))
        critic_params_sg = _stop_gradient_tree(critic_params_)

        def rollout_step(carry, inputs):
            state, foot_bump_ou = carry
            (
                epsilon_t,
                entropy_epsilon_t,
                velocity_push_t,
                terrain_bump_innov_t,
            ) = inputs

            push_due = (state.info["step"] > 0) & (
                (state.info["step"] % push_interval_steps) == 0
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
                std=terrain_bump_std,
                decay=terrain_bump_decay,
                robot_weight=nominal_weight,
            )
            xfrc = state.data.xfrc_applied
            for i in range(4):
                xfrc = xfrc.at[foot_body_ids[i], :3].add(
                    terrain_bump_forces[i]
                )
            state = state.replace(data=state.data.replace(xfrc_applied=xfrc))

            obs_rng, env_rng = jax.random.split(state.info["rng"])
            state = state.replace(info={**state.info, "rng": env_rng})
            actor_obs = env._apply_obs_noise(state.obs, obs_rng)
            obs_norm = env.normalize_actor_obs(
                actor_norm, actor_norm_state, actor_obs
            ).astype(jp.float32)

            action, raw_entropy, mu, log_std = _sapo_sample(
                actor,
                actor_params_,
                obs_norm,
                epsilon_t.astype(jp.float32),
                sapo_min_log_std,
                sapo_max_log_std,
                entropy_epsilon=entropy_epsilon_t.astype(jp.float32),
            )
            action = action.astype(jp.float64)

            next_state = env.step(state, action)
            foot_bump_ou = jp.where(
                next_state.done, jp.zeros((4, 3)), foot_bump_ou
            )
            entropy_norm = (raw_entropy + target_entropy_abs) / (
                2.0 * target_entropy_abs
            )

            return (next_state, foot_bump_ou), {
                "reward": next_state.reward,
                "done": next_state.done,
                "terminal": next_state.info["terminal"],
                "raw_entropy": raw_entropy,
                "entropy_norm": entropy_norm,
                "actor_obs": state.obs,
                "critic_obs": env._get_critic_obs(state.data, state.info),
                "bootstrap_critic_obs": next_state.info[
                    "bootstrap_critic_obs"
                ],
                "vel_x": next_state.metrics["vel_x"],
                "vel_y": next_state.metrics["vel_y"],
                "yaw_rate": next_state.metrics["yaw_rate"],
                "cmd_x": next_state.metrics["cmd_x"],
                "cmd_y": next_state.metrics["cmd_y"],
                "cmd_yaw": next_state.metrics["cmd_yaw"],
                "height": next_state.metrics["height"],
                "tilt": next_state.metrics["tilt"],
                "foot_normal_FL": next_state.metrics["foot_normal_FL"],
                "foot_normal_FR": next_state.metrics["foot_normal_FR"],
                "foot_normal_RL": next_state.metrics["foot_normal_RL"],
                "foot_normal_RR": next_state.metrics["foot_normal_RR"],
            }

        env_state_ = jax.lax.stop_gradient(env_state_)
        (final_state, final_foot_bump_ou), traj = jax.lax.scan(
            rollout_step,
            (env_state_, env_state_.info["foot_bump_ou"]),
            (
                action_eps,
                entropy_eps,
                velocity_pushes,
                terrain_bump_innovations,
            ),
            length=unroll_length,
        )
        final_state = final_state.replace(
            info={**final_state.info, "foot_bump_ou": final_foot_bump_ou}
        )

        bootstrap_obs = critic_norm.normalize(
            critic_norm_state, traj["bootstrap_critic_obs"]
        ).astype(jp.float32)
        v1 = critic.apply(critic_params_sg[0], bootstrap_obs).squeeze(-1)
        v2 = critic.apply(critic_params_sg[1], bootstrap_obs).squeeze(-1)
        bootstrap_v = 0.5 * (v1 + v2)

        soft_rewards = traj["reward"] + alpha * traj["entropy_norm"]

        def accum_return(carry, inputs):
            total, running, discount = carry
            reward_t, done_t, terminal_t, v_next = inputs
            next_discount = discount * gamma
            running_value = running + discount * reward_t
            trunc_bootstrap = (1.0 - terminal_t) * next_discount * v_next
            completed = jp.where(
                done_t, running_value + trunc_bootstrap, jp.array(0.0)
            )
            running = jp.where(done_t, jp.array(0.0), running_value)
            discount = jp.where(done_t, jp.array(1.0), next_discount)
            return (total + completed, running, discount), None

        (total_ret, running, final_discount), _ = jax.lax.scan(
            accum_return,
            (jp.array(0.0), jp.array(0.0), jp.array(1.0)),
            (
                soft_rewards,
                traj["done"],
                traj["terminal"],
                bootstrap_v,
            ),
        )

        final_obs = critic_norm.normalize(
            critic_norm_state,
            env._get_critic_obs(final_state.data, final_state.info),
        ).astype(jp.float32)
        final_v1 = critic.apply(critic_params_sg[0], final_obs).squeeze()
        final_v2 = critic.apply(critic_params_sg[1], final_obs).squeeze()
        final_v = 0.5 * (final_v1 + final_v2)
        final_bootstrap = jp.where(
            traj["done"][-1], jp.array(0.0), final_discount * final_v
        )
        total_ret = total_ret + running + final_bootstrap

        return -total_ret / float(unroll_length), (traj, final_state)

    def critic_loss_from_targets(
        critic_params_,
        critic_norm_state,
        traj_obs,
        target_values,
    ):
        """Fit both critics to a fixed, detached soft TD(lambda) target."""
        flat_obs = traj_obs.reshape(-1, env.critic_obs_dim)
        flat_obs_norm = critic_norm.normalize(
            critic_norm_state, flat_obs
        ).astype(jp.float32)

        values_1 = critic.apply(critic_params_[0], flat_obs_norm).squeeze(-1)
        values_2 = critic.apply(critic_params_[1], flat_obs_norm).squeeze(-1)
        targets = jax.lax.stop_gradient(target_values.reshape(-1))
        return 0.5 * (
            jp.mean(jp.square(values_1 - targets))
            + jp.mean(jp.square(values_2 - targets))
        )

    @jax.jit
    def train_step(state: TrainState):
        (
            key,
            noise_key,
            entropy_noise_key,
            push_key,
            bump_key,
            diff_mask_key,
        ) = jax.random.split(
            state.key, 6
        )

        difficulty = jp.clip(
            (state.step - curriculum_grace_jax).astype(jp.float32)
            / curriculum_steps_jax,
            0.0,
            1.0,
        )
        zero_diff_mask = jax.random.uniform(
            diff_mask_key, (num_envs,)
        ) < zero_difficulty_frac
        per_env_difficulty = jp.where(
            zero_diff_mask,
            jp.zeros(num_envs),
            jp.full((num_envs,), difficulty),
        )
        updated_env_state = state.env_state.replace(
            info={
                **state.env_state.info,
                "difficulty": per_env_difficulty,
            }
        )

        all_action_eps = jax.random.normal(
            noise_key, (num_envs, unroll_length, env.action_dim)
        )
        all_entropy_eps = jax.random.normal(
            entropy_noise_key, (num_envs, unroll_length, env.action_dim)
        )
        all_velocity_pushes = jax.random.uniform(
            push_key,
            (num_envs, unroll_length, 2),
            minval=push_velocity_lo,
            maxval=push_velocity_hi,
        )
        all_terrain_bump_innovations = jax.random.normal(
            bump_key, (num_envs, unroll_length, 4, 3)
        )
        all_randomization = (
            all_action_eps,
            all_entropy_eps,
            all_velocity_pushes,
            all_terrain_bump_innovations,
        )

        current_alpha = jp.exp(jax.lax.stop_gradient(state.sapo_log_alpha))
        actor_grad_fn = jax.value_and_grad(actor_loss, has_aux=True)
        (actor_losses, (trajs, final_states)), actor_grads = jax.vmap(
            actor_grad_fn,
            in_axes=(None, None, None, None, 0, 0, None),
        )(
            state.actor_params,
            state.critic_params,
            state.normalizer,
            state.critic_normalizer,
            updated_env_state,
            all_randomization,
            state.sapo_log_alpha,
        )

        actor_grads = _clip_batched_tree_by_norm(
            actor_grads, sapo_per_env_grad_clip
        )
        actor_grads = jax.tree_util.tree_map(
            lambda g: jp.nanmean(g, axis=0), actor_grads
        )
        actor_grads = _finite_tree(actor_grads)
        actor_grad_norm = compute_grad_norm(actor_grads)
        actor_updates, new_actor_opt = actor_opt.update(
            actor_grads, state.actor_opt, state.actor_params
        )
        new_actor_params = optax.apply_updates(
            state.actor_params, actor_updates
        )

        # Temperature update uses the signed differential entropy, matching
        # Mineral's squashed-Normal implementation. The target is negative
        # because a bounded continuous distribution can have negative
        # differential entropy.
        raw_entropy = jax.lax.stop_gradient(trajs["raw_entropy"])

        def alpha_loss(log_alpha, entropy):
            alpha = jp.exp(log_alpha)
            # Match Mineral's ``unscale_entropy_alpha`` path: actor/critic
            # use entropy normalized by |H_target|, while the temperature
            # dual uses the signed, unnormalized entropy and target.
            alpha_unscaled = alpha * target_entropy_abs
            return alpha_unscaled * jp.mean(entropy - target_entropy)

        alpha_value, alpha_grad = jax.value_and_grad(alpha_loss)(
            state.sapo_log_alpha, raw_entropy
        )
        alpha_grad = jp.where(
            jp.isfinite(alpha_grad), alpha_grad, jp.array(0.0)
        )
        alpha_updates, new_alpha_opt = alpha_opt.update(
            alpha_grad, state.sapo_alpha_opt, state.sapo_log_alpha
        )
        new_log_alpha = optax.apply_updates(
            state.sapo_log_alpha, alpha_updates
        )
        new_log_alpha = jp.clip(new_log_alpha, -8.0, 4.0)

        all_obs = trajs["critic_obs"]
        all_rewards = trajs["reward"]
        all_entropy_norm = trajs["entropy_norm"]
        all_dones = trajs["done"]
        all_terminals = trajs["terminal"]
        all_bootstrap_obs = trajs["bootstrap_critic_obs"]

        # SAPO computes the soft TD(lambda) targets once, before the critic
        # mini-epochs. The min of the two pre-update critics is used for
        # bootstrapping, while the actor above used their mean.
        bootstrap_norm = critic_norm.normalize(
            state.critic_normalizer,
            all_bootstrap_obs.reshape(-1, env.critic_obs_dim),
        ).astype(jp.float32)
        next_v1 = critic.apply(
            state.critic_params[0], bootstrap_norm
        ).squeeze(-1).reshape(num_envs, unroll_length)
        next_v2 = critic.apply(
            state.critic_params[1], bootstrap_norm
        ).squeeze(-1).reshape(num_envs, unroll_length)
        next_values = jp.minimum(next_v1, next_v2)
        horizon_boundary = jp.zeros_like(all_dones).at[:, -1].set(1.0)
        done_mask = jp.maximum(all_dones, horizon_boundary)

        target_alpha = jp.exp(jax.lax.stop_gradient(new_log_alpha))
        soft_rewards = jax.lax.stop_gradient(
            all_rewards + target_alpha * all_entropy_norm
        )
        fixed_targets = jax.vmap(
            _soft_td_lambda_targets,
            in_axes=(0, 0, 0, 0, None, None),
        )(
            soft_rewards,
            done_mask,
            all_terminals,
            next_values,
            gamma,
            gae_lambda,
        )

        def single_env_critic_loss(
            critic_params_,
            norm_state,
            obs,
            targets,
        ):
            return critic_loss_from_targets(
                critic_params_,
                norm_state,
                obs,
                targets,
            )

        def critic_update_step(carry, _):
            c_params, c_opt_state = carry
            c_losses, c_grads = jax.vmap(
                jax.value_and_grad(single_env_critic_loss),
                in_axes=(None, None, 0, 0),
            )(
                c_params,
                state.critic_normalizer,
                all_obs,
                fixed_targets,
            )
            c_grads = _clip_batched_tree_by_norm(
                c_grads, sapo_per_env_grad_clip
            )
            c_grads = jax.tree_util.tree_map(
                lambda g: jp.nanmean(g, axis=0), c_grads
            )
            c_grads = _finite_tree(c_grads)
            c_updates, new_c_opt = critic_opt.update(
                c_grads, c_opt_state, c_params
            )
            new_c_params = optax.apply_updates(c_params, c_updates)
            return (new_c_params, new_c_opt), jp.mean(c_losses)

        (new_critic_params, new_critic_opt), critic_losses = jax.lax.scan(
            critic_update_step,
            (state.critic_params, state.critic_opt),
            None,
            length=critic_iterations,
        )

        flat_actor_obs = trajs["actor_obs"].reshape(
            -1, env.actor_frame_obs_dim
        )
        safe_actor_obs = jp.where(
            jp.isfinite(flat_actor_obs),
            flat_actor_obs,
            state.normalizer.mean,
        )
        new_actor_norm = actor_norm.update(
            state.normalizer, safe_actor_obs
        )

        flat_critic_obs = trajs["critic_obs"].reshape(
            -1, env.critic_obs_dim
        )
        safe_critic_obs = jp.where(
            jp.isfinite(flat_critic_obs),
            flat_critic_obs,
            state.critic_normalizer.mean,
        )
        new_critic_norm = critic_norm.update(
            state.critic_normalizer, safe_critic_obs
        )

        new_state = state.replace(
            key=key,
            env_state=final_states,
            actor_params=new_actor_params,
            critic_params=new_critic_params,
            target_critic_params=None,
            normalizer=new_actor_norm,
            critic_normalizer=new_critic_norm,
            actor_opt=new_actor_opt,
            critic_opt=new_critic_opt,
            sapo_log_alpha=new_log_alpha,
            sapo_alpha_opt=new_alpha_opt,
            step=state.step + num_envs * unroll_length,
        )

        metrics = {
            "reward": jp.mean(trajs["reward"]),
            "vel_x": jp.mean(trajs["vel_x"]),
            "vel_y": jp.mean(trajs["vel_y"]),
            "yaw_rate": jp.mean(trajs["yaw_rate"]),
            "cmd_x": jp.mean(trajs["cmd_x"]),
            "cmd_y": jp.mean(trajs["cmd_y"]),
            "cmd_yaw": jp.mean(trajs["cmd_yaw"]),
            "contact": jp.mean(final_states.metrics["contact_force"]),
            "actor_grad": actor_grad_norm,
            "critic_loss": critic_losses[-1],
            "actor_loss": jp.mean(actor_losses),
            "track_vx": jp.mean(
                jp.abs(trajs["vel_x"] - trajs["cmd_x"])
            ),
            "track_vy": jp.mean(
                jp.abs(trajs["vel_y"] - trajs["cmd_y"])
            ),
            "track_yaw": jp.mean(
                jp.abs(trajs["yaw_rate"] - trajs["cmd_yaw"])
            ),
            "track_vx_sq": jp.mean(
                jp.square(trajs["vel_x"] - trajs["cmd_x"])
            ),
            "track_vy_sq": jp.mean(
                jp.square(trajs["vel_y"] - trajs["cmd_y"])
            ),
            "track_yaw_sq": jp.mean(
                jp.square(trajs["yaw_rate"] - trajs["cmd_yaw"])
            ),
            "rew_vel_x": jp.mean(final_states.metrics["rew_vel_x"]),
            "rew_vel_y": jp.mean(final_states.metrics["rew_vel_y"]),
            "rew_yaw": jp.mean(final_states.metrics["rew_yaw"]),
            "rew_vz": jp.mean(final_states.metrics["rew_vz"]),
            "pen_rate": jp.mean(final_states.metrics["pen_rate"]),
            "height": jp.mean(trajs["height"]),
            "tilt": jp.mean(trajs["tilt"]),
            "difficulty": difficulty,
            "alpha": current_alpha,
            "alpha_loss": alpha_value,
            "raw_entropy": jp.mean(raw_entropy),
            "entropy_norm": jp.mean(trajs["entropy_norm"]),
            "log_std": jp.mean(
                jax.vmap(
                    lambda obs: _sapo_sample(
                        actor,
                        state.actor_params,
                        obs,
                        jp.zeros(env.action_dim),
                        sapo_min_log_std,
                        sapo_max_log_std,
                    )[3]
                )(
                    env.normalize_actor_obs(
                        actor_norm,
                        state.normalizer,
                        trajs["actor_obs"][0, 0],
                    ).reshape(1, -1)
                )
            ),
            "foot_normal_FL": jp.mean(trajs["foot_normal_FL"]),
            "foot_normal_FR": jp.mean(trajs["foot_normal_FR"]),
            "foot_normal_RL": jp.mean(trajs["foot_normal_RL"]),
            "foot_normal_RR": jp.mean(trajs["foot_normal_RR"]),
        }
        return new_state, metrics

    if resumed_state is not None:
        if resumed_state.sapo_log_alpha is None:
            raise ValueError("Checkpoint does not contain SAPO temperature state")
        state = TrainState(
            key=key,
            env_state=env_state,
            actor_params=resumed_state.actor_params,
            critic_params=resumed_state.critic_params,
            target_critic_params=None,
            normalizer=resumed_state.normalizer,
            critic_normalizer=resumed_state.critic_normalizer,
            actor_opt=resumed_state.actor_opt,
            critic_opt=resumed_state.critic_opt,
            step=resumed_step,
            sapo_log_alpha=resumed_state.sapo_log_alpha,
            sapo_alpha_opt=resumed_state.sapo_alpha_opt,
        )
    else:
        state = TrainState(
            key=key,
            env_state=env_state,
            actor_params=actor_params,
            critic_params=critic_params,
            target_critic_params=None,
            normalizer=actor_normalizer,
            critic_normalizer=critic_normalizer,
            actor_opt=actor_opt_state,
            critic_opt=critic_opt_state,
            step=0,
            sapo_log_alpha=initial_log_alpha,
            sapo_alpha_opt=alpha_opt_state,
        )

    print("Compiling SAPO...")
    start_comp_time = time.perf_counter()
    warmup_state, _ = train_step(state)
    jax.block_until_ready(warmup_state.step)
    compile_time = time.perf_counter() - start_comp_time
    print(f"Compilation took {compile_time:.1f}s")

    # Use the compilation rollout to warm up normalizer statistics, but do not
    # consume its optimizer update or advance the reported training step.
    state = state.replace(
        normalizer=warmup_state.normalizer,
        critic_normalizer=warmup_state.critic_normalizer,
    )

    if diagnose:
        header = (
            f"{'Step':>8} | {'Rew':>7} | {'TrkVx':>7} | {'TrkVy':>7} | "
            f"{'TrkYaw':>7} | {'Alpha':>7} | {'Ent':>7} | "
            f"{'AGrad':>7} | {'CLoss':>8} | {'Diff':>5} | {'Status'}"
        )
    else:
        header = (
            f"{'Step':>8} | {'Rew':>7} | {'TrkVx':>7} | {'TrkVy':>7} | "
            f"{'TrkYaw':>7} | {'Alpha':>7} | {'Ent':>7} | "
            f"{'CLoss':>8} | {'Diff':>5} | {'Status':>8}"
        )
    print("=" * len(header))
    print(header)
    print("=" * len(header))

    start = time.time()
    log = []
    diag_log = []
    last_checkpoint_step = state.step
    steps_per_iter = num_envs * unroll_length
    start_iter = resumed_step // steps_per_iter
    total_iters = total_steps // steps_per_iter

    for i in range(start_iter, total_iters):
        state, metrics = train_step(state)
        if i % 10 == 0:
            jax.block_until_ready(state.step)
            reward = float(metrics["reward"])
            trk_vx = float(metrics["track_vx"])
            trk_vy = float(metrics["track_vy"])
            trk_yaw = float(metrics["track_yaw"])
            alpha = float(metrics["alpha"])
            entropy = float(metrics["raw_entropy"])
            diff = float(metrics["difficulty"])
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
                    f"{state.step:8d} | {reward:7.3f} | {trk_vx:7.3f} | "
                    f"{trk_vy:7.3f} | {trk_yaw:7.3f} | {alpha:7.3f} | "
                    f"{entropy:7.3f} | {float(metrics['actor_grad']):7.2f} | "
                    f"{float(metrics['critic_loss']):8.4f} | {diff:5.2f} | "
                    f"{status}"
                )
            else:
                print(
                    f"{state.step:8d} | {reward:7.3f} | {trk_vx:7.3f} | "
                    f"{trk_vy:7.3f} | {trk_yaw:7.3f} | {alpha:7.3f} | "
                    f"{entropy:7.3f} | {float(metrics['critic_loss']):8.4f} | "
                    f"{diff:5.2f} | {status}"
                )

            vel_x = float(metrics["vel_x"])
            vel_y = float(metrics["vel_y"])
            yaw_rate = float(metrics["yaw_rate"])
            cmd_x = float(metrics["cmd_x"])
            cmd_y = float(metrics["cmd_y"])
            cmd_yaw = float(metrics["cmd_yaw"])
            log.append(
                [
                    int(state.step),
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
                    alpha,
                    entropy,
                    float(metrics["entropy_norm"]),
                ]
            )
            if diagnose:
                diag_log.append(
                    {
                        "step": int(state.step),
                        "reward": reward,
                        "difficulty": diff,
                        "track_vx": trk_vx,
                        "track_vy": trk_vy,
                        "track_yaw": trk_yaw,
                        "alpha": alpha,
                        "raw_entropy": entropy,
                        "entropy_norm": float(metrics["entropy_norm"]),
                        "actor_grad": float(metrics["actor_grad"]),
                        "critic_loss": float(metrics["critic_loss"]),
                    }
                )

            if reward > best_reward and state.step > 5000:
                best_reward = reward
                with open(f"{save_dir}/policy_best.pkl", "wb") as f:
                    pickle.dump(slim_checkpoint_state(state), f)
                print(f"  >> New best reward: {best_reward:.3f}")

            tracking_score = float(
                metrics["track_vx_sq"]
                + metrics["track_vy_sq"]
                + metrics["track_yaw_sq"]
            )
            if tracking_score < best_tracking and state.step > 5000:
                best_tracking = tracking_score
                with open(
                    f"{save_dir}/policy_best_tracking.pkl", "wb"
                ) as f:
                    pickle.dump(slim_checkpoint_state(state), f)
                print(
                    f"  >> New best tracking error sum: {best_tracking:.3f}"
                )

            if state.step - last_checkpoint_step >= checkpoint_interval:
                with open(
                    os.path.join(save_dir, "checkpoint_latest.pkl"), "wb"
                ) as f:
                    pickle.dump(slim_checkpoint_state(state), f)
                last_checkpoint_step = state.step
                print(f"  >> Checkpoint saved at step {state.step}")

    with open(f"{save_dir}/policy_final.pkl", "wb") as f:
        pickle.dump(slim_checkpoint_state(state), f)
    np.save(f"{save_dir}/log.npy", np.array(log))
    if diagnose and diag_log:
        with open(f"{save_dir}/diag_log.json", "w") as f:
            json.dump(diag_log, f, indent=2)

    elapsed = time.time() - start
    print("=" * 110)
    print(
        f"SAPO training complete in {elapsed:.1f}s "
        f"(compile: {compile_time:.1f}s)"
    )
    print(f"Best reward: {best_reward:.3f}")
    print(f"Best tracking error sum: {best_tracking:.3f}")
    print(
        f"Final alpha: {float(jp.exp(state.sapo_log_alpha)):.5f}, "
        f"target entropy: {target_entropy:.2f}, "
        f"final raw entropy: {float(metrics['raw_entropy']) if total_iters else float('nan'):.3f}"
    )

    hparams["best_reward"] = best_reward
    hparams["best_tracking"] = best_tracking
    hparams["final_alpha"] = float(jp.exp(state.sapo_log_alpha))
    hparams["training_elapsed_s"] = elapsed
    hparams["compile_time_s"] = compile_time
    with open(f"{save_dir}/hparams.json", "w") as f:
        json.dump(hparams, f, indent=2)

    return state, save_dir
