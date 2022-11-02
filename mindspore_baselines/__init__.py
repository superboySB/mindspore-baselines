import os

from mindspore_baselines.a2c import A2C
from mindspore_baselines.common.utils import get_system_info
from mindspore_baselines.ddpg import DDPG
from mindspore_baselines.dqn import DQN
from mindspore_baselines.her.her_replay_buffer import HerReplayBuffer
from mindspore_baselines.ppo import PPO
from mindspore_baselines.sac import SAC
from mindspore_baselines.td3 import TD3

# Read version from file
version_file = os.path.join(os.path.dirname(__file__), "version.txt")
with open(version_file) as file_handler:
    __version__ = file_handler.read().strip()


def HER(*args, **kwargs):
    raise ImportError(
        "Since Stable Baselines 2.1.0, `HER` is now a replay buffer class `HerReplayBuffer`.\n "
        "Please check the documentation for more information: https://stable-baselines3.readthedocs.io/"
    )
