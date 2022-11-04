"""Common aliases for type hints"""

import sys
from enum import Enum
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple, Union

import gym
import numpy as np
import mindspore as ms

if sys.version_info >= (3, 8):
    from typing import Protocol
else:
    from typing_extensions import Protocol

from mindspore_baselines.common import callbacks, vec_env

GymEnv = Union[gym.Env, vec_env.VecEnv]
GymObs = Union[Tuple, Dict[str, Any], np.ndarray, int]
GymStepReturn = Tuple[GymObs, float, bool, Dict]
TensorDict = Dict[Union[str, int], ms.Tensor]
OptimizerStateDict = Dict[str, Any]
MaybeCallback = Union[None, Callable, List[callbacks.BaseCallback], callbacks.BaseCallback]

# A schedule takes the remaining progress as input
# and ouputs a scalar (e.g. learning rate, clip range, ...)
Schedule = Callable[[float], float]


class RolloutBufferSamples(NamedTuple):
    observations: ms.Tensor
    actions: ms.Tensor
    old_values: ms.Tensor
    old_log_prob: ms.Tensor
    advantages: ms.Tensor
    returns: ms.Tensor


class DictRolloutBufferSamples(RolloutBufferSamples):
    observations: TensorDict
    actions: ms.Tensor
    old_values: ms.Tensor
    old_log_prob: ms.Tensor
    advantages: ms.Tensor
    returns: ms.Tensor


class ReplayBufferSamples(NamedTuple):
    observations: ms.Tensor
    actions: ms.Tensor
    next_observations: ms.Tensor
    dones: ms.Tensor
    rewards: ms.Tensor


class DictReplayBufferSamples(ReplayBufferSamples):
    observations: TensorDict
    actions: ms.Tensor
    next_observations: TensorDict
    dones: ms.Tensor
    rewards: ms.Tensor


class RolloutReturn(NamedTuple):
    episode_timesteps: int
    n_episodes: int
    continue_training: bool


class TrainFrequencyUnit(Enum):
    STEP = "step"
    EPISODE = "episode"


class TrainFreq(NamedTuple):
    frequency: int
    unit: TrainFrequencyUnit  # either "step" or "episode"

class PolicyPredictor(Protocol):
    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Get the policy action from an observation (and optional hidden state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).
        :param observation: the input observation
        :param state: The last hidden states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
            this correspond to beginning of episodes,
            where the hidden states of the RNN must be reset.
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next hidden state
            (used in recurrent policies)
        """