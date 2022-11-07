"""Probability distributions."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import gym
import numpy as np
from gym import spaces

import mindspore as ms
from mindspore import ops, nn

from mindspore_baselines.common.preprocessing import get_action_dim


class Distribution(ABC):
    """Abstract base class for distributions."""

    def __init__(self):
        super().__init__()
        self.distribution = None

    @abstractmethod
    def proba_distribution_net(self, *args, **kwargs) -> Union[nn.Cell, Tuple[nn.Cell, ms.Parameter]]:
        """Create the layers and parameters that represent the distribution.

        Subclasses must define this, but the arguments and return type vary between
        concrete classes."""

    @abstractmethod
    def proba_distribution(self, *args, **kwargs) -> "Distribution":
        """Set parameters of the distribution.

        :return: self
        """

    @abstractmethod
    def log_prob(self, x: ms.Tensor) -> ms.Tensor:
        """
        Returns the log likelihood

        :param x: the taken action
        :return: The log likelihood of the distribution
        """

    @abstractmethod
    def entropy(self) -> Optional[ms.Tensor]:
        """
        Returns Shannon's entropy of the probability

        :return: the entropy, or None if no analytical form is known
        """

    @abstractmethod
    def sample(self) -> ms.Tensor:
        """
        Returns a sample from the probability distribution

        :return: the stochastic action
        """

    @abstractmethod
    def mode(self) -> ms.Tensor:
        """
        Returns the most likely action (deterministic output)
        from the probability distribution

        :return: the stochastic action
        """

    def get_actions(self, deterministic: bool = False) -> ms.Tensor:
        """
        Return actions according to the probability distribution.

        :param deterministic:
        :return:
        """
        if deterministic:
            return self.mode()
        return self.sample()

    @abstractmethod
    def actions_from_params(self, *args, **kwargs) -> ms.Tensor:
        """
        Returns samples from the probability distribution
        given its parameters.

        :return: actions
        """

    @abstractmethod
    def log_prob_from_params(self, *args, **kwargs) -> Tuple[ms.Tensor, ms.Tensor]:
        """
        Returns samples and the associated log probabilities
        from the probability distribution given its parameters.

        :return: actions and log prob
        """


def sum_independent_dims(tensor: ms.Tensor) -> ms.Tensor:
    """
    Continuous actions are usually considered to be independent,
    so we can sum components of the ``log_prob`` or the entropy.

    :param tensor: shape: (n_batch, n_actions) or (n_batch,)
    :return: shape: (n_batch,)
    """
    if len(tensor.shape) > 1:
        tensor = tensor.sum(axis=1)
    else:
        tensor = tensor.sum()
    return tensor


class DiagGaussianDistribution(Distribution):
    """
    Gaussian distribution with diagonal covariance matrix, for continuous actions.

    :param action_dim:  Dimension of the action space.
    """

    def __init__(self, action_dim: int):
        super().__init__()
        self.action_dim = action_dim
        self.mean_actions = None
        self.log_std = None

    def proba_distribution_net(self, latent_dim: int, log_std_init: float = 0.0) -> Tuple[nn.Cell, ms.Parameter]:
        """
        Create the layers and parameter that represent the distribution:
        one output will be the mean of the Gaussian, the other parameter will be the
        standard deviation (log std in fact to allow negative values)

        :param latent_dim: Dimension of the last layer of the policy (before the action layer)
        :param log_std_init: Initial value for the log standard deviation
        :return:
        """
        mean_actions = nn.Dense(latent_dim, self.action_dim)
        # TODO: allow action dependent std
        log_std = ms.Parameter(ops.ones(self.action_dim, type=ms.float32) * log_std_init, requires_grad=True)
        return mean_actions, log_std

    def proba_distribution(self, mean_actions: ms.Tensor, log_std: ms.Tensor) -> "DiagGaussianDistribution":
        """
        Create the distribution given its parameters (mean, std)

        :param mean_actions:
        :param log_std:
        :return:
        """
        action_std = ops.ones_like(mean_actions) * log_std.exp()
        self.distribution = nn.probability.distribution.Normal(mean_actions, action_std)
        return self

    def log_prob(self, actions: ms.Tensor) -> ms.Tensor:
        """
        Get the log probabilities of actions according to the distribution.
        Note that you must first call the ``proba_distribution()`` method.

        :param actions:
        :return:
        """
        log_prob = self.distribution.log_prob(actions)
        return sum_independent_dims(log_prob)

    def entropy(self) -> ms.Tensor:
        return sum_independent_dims(self.distribution.entropy())

    def sample(self) -> ms.Tensor:
        # Reparametrization trick to pass gradients
        eps = ops.standard_normal(self.distribution.mean().shape)
        return self.distribution.mean() + eps * self.distribution.sd()

    def mode(self) -> ms.Tensor:
        return self.distribution.mean()

    def actions_from_params(self, mean_actions: ms.Tensor, log_std: ms.Tensor,
                            deterministic: bool = False) -> ms.Tensor:
        # Update the proba distribution
        self.proba_distribution(mean_actions, log_std)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(self, mean_actions: ms.Tensor, log_std: ms.Tensor) -> Tuple[ms.Tensor, ms.Tensor]:
        """
        Compute the log probability of taking an action
        given the distribution parameters.

        :param mean_actions:
        :param log_std:
        :return:
        """
        actions = self.actions_from_params(mean_actions, log_std)
        log_prob = self.log_prob(actions)
        return actions, log_prob


class SquashedDiagGaussianDistribution(DiagGaussianDistribution):
    """
    Gaussian distribution with diagonal covariance matrix, followed by a squashing function (tanh) to ensure bounds.

    :param action_dim: Dimension of the action space.
    :param epsilon: small value to avoid NaN due to numerical imprecision.
    """

    def __init__(self, action_dim: int, epsilon: float = 1e-6):
        super().__init__(action_dim)
        # Avoid NaN (prevents division by zero or log of zero)
        self.epsilon = epsilon
        self.gaussian_actions = None

    def proba_distribution(self, mean_actions: ms.Tensor, log_std: ms.Tensor) -> "SquashedDiagGaussianDistribution":
        super().proba_distribution(mean_actions, log_std)
        return self

    def log_prob(self, actions: ms.Tensor, gaussian_actions: Optional[ms.Tensor] = None) -> ms.Tensor:
        # Inverse tanh
        # Naive implementation (not stable): 0.5 * torch.log((1 + x) / (1 - x))
        # We use numpy to avoid numerical instability
        if gaussian_actions is None:
            # It will be clipped to avoid NaN when inversing tanh
            gaussian_actions = TanhBijector.inverse(actions)

        # Log likelihood for a Gaussian distribution
        log_prob = super().log_prob(gaussian_actions)
        # Squash correction (from original SAC implementation)
        # this comes from the fact that tanh is bijective and differentiable
        log_prob -= ops.ReduceSum()(ops.log(1 - actions ** 2 + self.epsilon), axis=1)
        return log_prob

    def entropy(self) -> Optional[ms.Tensor]:
        # No analytical form,
        # entropy needs to be estimated using -log_prob.mean()
        return None

    def sample(self) -> ms.Tensor:
        # Reparametrization trick to pass gradients
        self.gaussian_actions = super().sample()
        return ops.tanh(self.gaussian_actions)

    def mode(self) -> ms.Tensor:
        self.gaussian_actions = super().mode()
        # Squash the output
        return ops.tanh(self.gaussian_actions)

    def log_prob_from_params(self, mean_actions: ms.Tensor, log_std: ms.Tensor) -> Tuple[ms.Tensor, ms.Tensor]:
        action = self.actions_from_params(mean_actions, log_std)
        log_prob = self.log_prob(action, self.gaussian_actions)
        return action, log_prob


class CategoricalDistribution(Distribution):
    """
    Categorical distribution for discrete actions.

    :param action_dim: Number of discrete actions
    """

    def __init__(self, action_dim: int):
        super().__init__()
        self.action_dim = action_dim

    def proba_distribution_net(self, latent_dim: int) -> nn.Cell:
        """
        Create the layer that represents the distribution:
        it will be the logits of the Categorical distribution.
        You can then get probabilities using a softmax.

        :param latent_dim: Dimension of the last layer
            of the policy network (before the action layer)
        :return:
        """
        action_logits = nn.Dense(latent_dim, self.action_dim)
        return action_logits

    def proba_distribution(self, action_logits: ms.Tensor) -> "CategoricalDistribution":
        self.distribution = nn.probability.distribution.Categorical(probs=ops.softmax(action_logits,axis=-1))
        return self

    def log_prob(self, actions: ms.Tensor) -> ms.Tensor:
        return self.distribution.log_prob(actions)

    def entropy(self) -> ms.Tensor:
        return self.distribution.entropy()

    def sample(self) -> ms.Tensor:
        return self.distribution.sample()

    def mode(self) -> ms.Tensor:
        return ops.argmax(self.distribution.probs, axis=1)

    def actions_from_params(self, action_logits: ms.Tensor, deterministic: bool = False) -> ms.Tensor:
        # Update the proba distribution
        self.proba_distribution(action_logits)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(self, action_logits: ms.Tensor) -> Tuple[ms.Tensor, ms.Tensor]:
        actions = self.actions_from_params(action_logits)
        log_prob = self.log_prob(actions)
        return actions, log_prob


class MultiCategoricalDistribution(Distribution):
    """
    MultiCategorical distribution for multi discrete actions.

    :param action_dims: List of sizes of discrete action spaces
    """

    def __init__(self, action_dims: List[int]):
        super().__init__()
        self.action_dims = action_dims

    def proba_distribution_net(self, latent_dim: int) -> nn.Cell:
        """
        Create the layer that represents the distribution:
        it will be the logits (flattened) of the MultiCategorical distribution.
        You can then get probabilities using a softmax on each sub-space.

        :param latent_dim: Dimension of the last layer
            of the policy network (before the action layer)
        :return:
        """

        action_logits = nn.Dense(latent_dim, sum(self.action_dims))
        return action_logits

    def proba_distribution(self, action_logits: ms.Tensor) -> "MultiCategoricalDistribution":
        split_indices = []
        split_index =0
        for action_dim in self.action_dims:
            split_index +=action_dim
            split_indices.append(split_index)
        split_indices.pop(1)
        self.distribution = [nn.probability.distribution.Categorical(probs=ops.softmax(split,axis=-1)) for split in
                             ms.numpy.split(action_logits, indices_or_sections=split_indices, axis=1)]
        return self

    def log_prob(self, actions: ms.Tensor) -> ms.Tensor:
        # Extract each discrete action and compute log prob for their respective distributions
        return ops.stack(
            [dist.log_prob(action) for dist, action in zip(self.distribution, ops.unstack(actions, axis=1))], axis=1
        ).sum(axis=1)

    def entropy(self) -> ms.Tensor:
        return ops.stack([dist.entropy() for dist in self.distribution], axis=1).sum(axis=1)

    def sample(self) -> ms.Tensor:
        return ops.stack([dist.sample() for dist in self.distribution], axis=1)

    def mode(self) -> ms.Tensor:
        return ops.stack([ops.argmax(dist.probs, axis=1) for dist in self.distribution], axis=1)

    def actions_from_params(self, action_logits: ms.Tensor, deterministic: bool = False) -> ms.Tensor:
        # Update the proba distribution
        self.proba_distribution(action_logits)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(self, action_logits: ms.Tensor) -> Tuple[ms.Tensor, ms.Tensor]:
        actions = self.actions_from_params(action_logits)
        log_prob = self.log_prob(actions)
        return actions, log_prob


class BernoulliDistribution(Distribution):
    """
    Bernoulli distribution for MultiBinary action spaces.

    :param action_dim: Number of binary actions
    """

    def __init__(self, action_dims: int):
        super().__init__()
        self.action_dims = action_dims

    def proba_distribution_net(self, latent_dim: int) -> nn.Cell:
        """
        Create the layer that represents the distribution:
        it will be the logits of the Bernoulli distribution.

        :param latent_dim: Dimension of the last layer
            of the policy network (before the action layer)
        :return:
        """
        action_logits = nn.Dense(latent_dim, self.action_dims)
        return action_logits
    def proba_distribution(self, action_logits: ms.Tensor) -> "BernoulliDistribution":
        self.distribution = nn.probability.distribution.Bernoulli(probs=ops.softmax(action_logits,axis=-1))
        return self

    def log_prob(self, actions: ms.Tensor) -> ms.Tensor:
        return self.distribution.log_prob(actions).sum(axis=1)

    def entropy(self) -> ms.Tensor:
        return self.distribution.entropy().sum(axis=1)

    def sample(self) -> ms.Tensor:
        return self.distribution.sample()

    def mode(self) -> ms.Tensor:
        return ops.round(self.distribution.probs)

    def actions_from_params(self, action_logits: ms.Tensor, deterministic: bool = False) -> ms.Tensor:
        # Update the proba distribution
        self.proba_distribution(action_logits)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(self, action_logits: ms.Tensor) -> Tuple[ms.Tensor, ms.Tensor]:
        actions = self.actions_from_params(action_logits)
        log_prob = self.log_prob(actions)
        return actions, log_prob


class StateDependentNoiseDistribution(Distribution):
    """
    Distribution class for using generalized State Dependent Exploration (gSDE).
    Paper: https://arxiv.org/abs/2005.05719

    It is used to create the noise exploration matrix and
    compute the log probability of an action with that noise.

    :param action_dim: Dimension of the action space.
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,)
    :param use_expln: Use ``expln()`` function instead of ``exp()`` to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param squash_output: Whether to squash the output using a tanh function,
        this ensures bounds are satisfied.
    :param learn_features: Whether to learn features for gSDE or not.
        This will enable gradients to be backpropagated through the features
        ``latent_sde`` in the code.
    :param epsilon: small value to avoid NaN due to numerical imprecision.
    """

    def __init__(
            self,
            action_dim: int,
            full_std: bool = True,
            use_expln: bool = False,
            squash_output: bool = False,
            learn_features: bool = False,
            epsilon: float = 1e-6,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.latent_sde_dim = None
        self.mean_actions = None
        self.log_std = None
        self.weights_dist = None
        self.exploration_mat = None
        self.exploration_matrices = None
        self._latent_sde = None
        self.use_expln = use_expln
        self.full_std = full_std
        self.epsilon = epsilon
        self.learn_features = learn_features
        if squash_output:
            self.bijector = TanhBijector(epsilon)
        else:
            self.bijector = None

    def get_std(self, log_std: ms.Tensor) -> ms.Tensor:
        """
        Get the standard deviation from the learned parameter
        (log of it by default). This ensures that the std is positive.

        :param log_std:
        :return:
        """
        if self.use_expln:
            # From gSDE paper, it allows to keep variance
            # above zero and prevent it from growing too fast
            below_threshold = ops.exp(log_std) * (log_std <= 0)
            # Avoid NaN: zeros values that are below zero
            safe_log_std = log_std * (log_std > 0) + self.epsilon
            above_threshold = (ops.log1p(safe_log_std) + 1.0) * (log_std > 0)
            std = below_threshold + above_threshold
        else:
            # Use normal exponential
            std = ops.exp(log_std)

        if self.full_std:
            return std
        # Reduce the number of parameters:
        return ops.ones((self.latent_sde_dim, self.action_dim)) * std

    def sample_weights(self, log_std: ms.Tensor, batch_size: int = 1) -> None:
        """
        Sample weights for the noise exploration matrix,
        using a centered Gaussian distribution.

        :param log_std:
        :param batch_size:
        """
        std = self.get_std(log_std)
        self.weights_dist = nn.probability.distribution.Normal(ops.zeros_like(std), std)
        # Reparametrization trick to pass gradients
        eps = ops.standard_normal(self.weights_dist.mean().shape)
        self.exploration_mat = self.weights_dist.mean() + eps * self.weights_dist.sd()
        # Pre-compute matrices in case of parallel exploration
        eps = ops.standard_normal(self.weights_dist.sample((batch_size,)).shape)
        self.exploration_matrices = self.weights_dist.mean() + eps * self.weights_dist.sd()

    def proba_distribution_net(
            self, latent_dim: int, log_std_init: float = -2.0, latent_sde_dim: Optional[int] = None
    ) -> Tuple[nn.Cell, ms.Parameter]:
        """
        Create the layers and parameter that represent the distribution:
        one output will be the deterministic action, the other parameter will be the
        standard deviation of the distribution that control the weights of the noise matrix.

        :param latent_dim: Dimension of the last layer of the policy (before the action layer)
        :param log_std_init: Initial value for the log standard deviation
        :param latent_sde_dim: Dimension of the last layer of the features extractor
            for gSDE. By default, it is shared with the policy network.
        :return:
        """
        # Network for the deterministic action, it represents the mean of the distribution
        mean_actions_net = nn.Dense(latent_dim, self.action_dim)
        # When we learn features for the noise, the feature dimension
        # can be different between the policy and the noise network
        self.latent_sde_dim = latent_dim if latent_sde_dim is None else latent_sde_dim
        # Reduce the number of parameters if needed
        log_std = ops.ones((self.latent_sde_dim, self.action_dim),type=ms.float32) if self.full_std else ops.ones((self.latent_sde_dim, 1),type=ms.float32)
        # Transform it to a parameter so it can be optimized
        log_std = ms.Parameter(log_std * log_std_init, requires_grad=True)
        # Sample an exploration matrix
        self.sample_weights(log_std)
        return mean_actions_net, log_std

    def proba_distribution(
            self, mean_actions: ms.Tensor, log_std: ms.Tensor, latent_sde: ms.Tensor
    ) -> "StateDependentNoiseDistribution":
        """
        Create the distribution given its parameters (mean, std)

        :param mean_actions:
        :param log_std:
        :param latent_sde:
        :return:
        """
        # Stop gradient if we don't want to influence the features
        self._latent_sde = latent_sde if self.learn_features else latent_sde.detach()
        variance = ops.matmul(self._latent_sde ** 2, self.get_std(log_std) ** 2)
        self.distribution = nn.probability.distribution.Normal(mean_actions, ops.sqrt(variance + self.epsilon))
        return self

    def log_prob(self, actions: ms.Tensor) -> ms.Tensor:
        if self.bijector is not None:
            gaussian_actions = self.bijector.inverse(actions)
        else:
            gaussian_actions = actions
        # log likelihood for a gaussian
        log_prob = self.distribution.log_prob(gaussian_actions)
        # Sum along action dim
        log_prob = sum_independent_dims(log_prob)

        if self.bijector is not None:
            # Squash correction (from original SAC implementation)
            log_prob -= ops.reduce_sum(self.bijector.log_prob_correction(gaussian_actions), axis=1)
        return log_prob

    def entropy(self) -> Optional[ms.Tensor]:
        if self.bijector is not None:
            # No analytical form,
            # entropy needs to be estimated using -log_prob.mean()
            return None
        return sum_independent_dims(self.distribution.entropy())

    def sample(self) -> ms.Tensor:
        noise = self.get_noise(self._latent_sde)
        actions = self.distribution.mean() + noise
        if self.bijector is not None:
            return self.bijector.construct(actions)
        return actions

    def mode(self) -> ms.Tensor:
        actions = self.distribution.mean()
        if self.bijector is not None:
            return self.bijector.construct(actions)
        return actions

    def get_noise(self, latent_sde: ms.Tensor) -> ms.Tensor:
        latent_sde = latent_sde if self.learn_features else latent_sde
        # Default case: only one exploration matrix
        if len(latent_sde) == 1 or len(latent_sde) != len(self.exploration_matrices):
            return ops.matmul(latent_sde, self.exploration_mat)
        # Use batch matrix multiplication for efficient computation
        # (batch_size, n_features) -> (batch_size, 1, n_features)
        latent_sde = latent_sde.expand_dims(axis=1)
        # (batch_size, 1, n_actions)
        noise = ops.bmm(latent_sde, self.exploration_matrices)
        return noise.squeeze(axis=1)

    def actions_from_params(
            self, mean_actions: ms.Tensor, log_std: ms.Tensor, latent_sde: ms.Tensor, deterministic: bool = False
    ) -> ms.Tensor:
        # Update the proba distribution
        self.proba_distribution(mean_actions, log_std, latent_sde)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(
            self, mean_actions: ms.Tensor, log_std: ms.Tensor, latent_sde: ms.Tensor
    ) -> Tuple[ms.Tensor, ms.Tensor]:
        actions = self.actions_from_params(mean_actions, log_std, latent_sde)
        log_prob = self.log_prob(actions)
        return actions, log_prob


class TanhBijector:
    """
    Bijective transformation of a probability distribution
    using a squashing function (tanh)
    TODO: use Pyro instead (https://pyro.ai/)

    :param epsilon: small value to avoid NaN due to numerical imprecision.
    """

    def __init__(self, epsilon: float = 1e-6):
        super().__init__()
        self.epsilon = epsilon

    @staticmethod
    def construct(x: ms.Tensor) -> ms.Tensor:
        return ops.tanh(x)

    @staticmethod
    def atanh(x: ms.Tensor) -> ms.Tensor:
        """
        Inverse of Tanh

        Taken from Pyro: https://github.com/pyro-ppl/pyro
        0.5 * torch.log((1 + x ) / (1 - x))
        """
        return 0.5 * (x.log1p() - (-x).log1p())

    @staticmethod
    def inverse(y: ms.Tensor) -> ms.Tensor:
        """
        Inverse tanh.

        :param y:
        :return:
        """
        eps = ms.Tensor(np.finfo(np.float32).eps)
        # Clip the action to avoid NaN
        return TanhBijector.atanh(y.clip(xmin=-1.0 + eps, xmax=1.0 - eps))

    def log_prob_correction(self, x: ms.Tensor) -> ms.Tensor:
        # Squash correction (from original SAC implementation)
        return ops.log(1.0 - ops.tanh(x) ** 2 + self.epsilon)


def make_proba_distribution(
        action_space: gym.spaces.Space, use_sde: bool = False, dist_kwargs: Optional[Dict[str, Any]] = None
) -> Distribution:
    """
    Return an instance of Distribution for the correct type of action space

    :param action_space: the input action space
    :param use_sde: Force the use of StateDependentNoiseDistribution
        instead of DiagGaussianDistribution
    :param dist_kwargs: Keyword arguments to pass to the probability distribution
    :return: the appropriate Distribution object
    """
    if dist_kwargs is None:
        dist_kwargs = {}

    if isinstance(action_space, spaces.Box):
        cls = StateDependentNoiseDistribution if use_sde else DiagGaussianDistribution
        return cls(get_action_dim(action_space), **dist_kwargs)
    elif isinstance(action_space, spaces.Discrete):
        return CategoricalDistribution(action_space.n, **dist_kwargs)
    elif isinstance(action_space, spaces.MultiDiscrete):
        return MultiCategoricalDistribution(action_space.nvec, **dist_kwargs)
    elif isinstance(action_space, spaces.MultiBinary):
        return BernoulliDistribution(action_space.n, **dist_kwargs)
    else:
        raise NotImplementedError(
            "Error: probability distribution, not implemented for action space"
            f"of type {type(action_space)}."
            " Must be of type Gym Spaces: Box, Discrete, MultiDiscrete or MultiBinary."
        )


def kl_divergence(dist_true: Distribution, dist_pred: Distribution) -> ms.Tensor:
    """
    Wrapper for the PyTorch implementation of the full form KL Divergence

    :param dist_true: the p distribution
    :param dist_pred: the q distribution
    :return: KL(dist_true||dist_pred)
    """
    # KL Divergence for different distribution types is out of scope
    assert dist_true.__class__ == dist_pred.__class__, "Error: input distributions should be the same type"

    # MultiCategoricalDistribution is not a MindSpore Distribution subclass
    # so we need to implement it ourselves!
    if isinstance(dist_pred, MultiCategoricalDistribution):
        assert np.allclose(dist_pred.action_dims,
                           dist_true.action_dims), "Error: distributions must have the same input space"
        return ops.stack(
            [ops.kl_div(p, q, reduction='none') for p, q in zip(dist_true.distribution, dist_pred.distribution)],
            axis=1,
        ).sum(axis=1)

    # Use the MindSpore kl_divergence implementation
    else:
        return ops.kl_div(dist_true.distribution, dist_pred.distribution,reduction='none')
