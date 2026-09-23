"""Shared neural network modules."""

from typing import Sequence

import jax.numpy as jp
import flax.linen as nn


class Actor(nn.Module):
    """Policy MLP with zero-initialized action head."""

    action_dim: int
    hidden: Sequence[int] = (512, 256, 128)

    @nn.compact
    def __call__(self, x):
        for h in self.hidden:
            x = nn.Dense(h)(x)
            x = nn.LayerNorm()(x)
            x = nn.elu(x)

        x = nn.Dense(
            self.action_dim,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
        )(x)

        return nn.tanh(x)


class Critic(nn.Module):
    """Value MLP."""

    hidden: Sequence[int] = (512, 256, 128)
    activation: str = "elu"

    @nn.compact
    def __call__(self, x):
        for h in self.hidden:
            x = nn.Dense(h)(x)
            x = nn.LayerNorm()(x)
            if self.activation.lower() == "silu":
                x = nn.silu(x)
            elif self.activation.lower() == "relu":
                x = nn.relu(x)
            else:
                x = nn.elu(x)

        return nn.Dense(1)(x)


class SAPOActor(nn.Module):
    """State-dependent squashed-Normal actor used by SAPO.

    The module returns the parameters of the unsquashed Normal distribution.
    Sampling and the tanh Jacobian correction live in the algorithm so that
    the reparameterized sample remains in the MJX differentiation graph.
    """

    action_dim: int
    hidden: Sequence[int] = (128, 64, 32)
    activation: str = "silu"
    log_std_init: float = -1.0

    @nn.compact
    def __call__(self, x):
        # Use Mineral's orthogonal initialization, with small distribution
        # heads so the initial Go2 action stays close to the nominal pose.
        kernel_init = nn.initializers.orthogonal(scale=1.0)
        for h in self.hidden:
            x = nn.Dense(h, kernel_init=kernel_init)(x)
            x = nn.LayerNorm()(x)
            if self.activation.lower() == "silu":
                x = nn.silu(x)
            elif self.activation.lower() == "relu":
                x = nn.relu(x)
            else:
                x = nn.elu(x)

        mu = nn.Dense(
            self.action_dim,
            kernel_init=nn.initializers.orthogonal(scale=0.01),
            bias_init=nn.initializers.zeros,
            name="mu",
        )(x)
        log_std = nn.Dense(
            self.action_dim,
            kernel_init=nn.initializers.orthogonal(scale=0.01),
            bias_init=nn.initializers.constant(self.log_std_init),
            name="log_std",
        )(x)
        return mu, log_std


class SAPOCritic(nn.Module):
    """SAPO value network.

    SAPO trains two independent instances of this module. Keeping the
    instances separate is important: the clipped-double-critic target only
    works when their approximation errors are not identical.
    """

    hidden: Sequence[int] = (64, 64)
    activation: str = "silu"

    @nn.compact
    def __call__(self, x):
        # Match the orthogonalg1 initialization used by Mineral's SAPO critic.
        kernel_init = nn.initializers.orthogonal(scale=1.0)
        for h in self.hidden:
            x = nn.Dense(h, kernel_init=kernel_init)(x)
            x = nn.LayerNorm()(x)
            if self.activation.lower() == "silu":
                x = nn.silu(x)
            elif self.activation.lower() == "relu":
                x = nn.relu(x)
            else:
                x = nn.elu(x)

        return nn.Dense(1, kernel_init=nn.initializers.orthogonal(scale=1.0))(x)


class LearnedDynamicsModel(nn.Module):
    """Predict normalized observation residual from obs/action."""

    obs_dim: int
    hidden: Sequence[int] = (256, 256)

    @nn.compact
    def __call__(self, obs_norm, action):
        x = jp.concatenate([obs_norm, action], axis=-1)
        for h in self.hidden:
            x = nn.Dense(h)(x)
            x = nn.LayerNorm()(x)
            x = nn.elu(x)

        return nn.Dense(self.obs_dim)(x)
