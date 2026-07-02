import os
from dataclasses import dataclass
from envs import legal_envs
from impls.agents import default_config
from ml_collections import FrozenConfigDict
from typing import Literal

SRC_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class ExpConfig:
    # wandb logging
    name: str
    project: str = "crl_subgoal"
    wandb_mode: str = "online"
    entity: str | None = None

    # Replay buffer and batch size and seed
    num_envs: int = 1024
    batch_size: int = 1024
    seed: int = 0
    max_replay_size: int = 10000

    # Number of updates etc
    epochs: int = 10
    intervals_per_epoch: int = 100
    updates_per_rollout: int = 1000

    # Miscellaneous
    use_targets: bool = False
    gamma: float = 0.99
    use_future_and_random_goals: bool = False  # noqa: E501 Whether to use environment goals, default - geometric sampling of future states

    # Evaluation settings
    eval_different_box_numbers: bool = False
    eval_special: bool = False
    """In addition to standard evaluation also evaluate in 'special' mode. The specifics of special mode depend on level generator used."""

    # Filtering settings
    filtering: Literal["horizontal", "vertical", "quarter"] | None = None
    """Type of filtering used during TRAINING.'horizontal' and 'vertical' doesn't allow to cross respctive board symmetry. 'quarter' only allows the agent to be in the quarter with boxes or targets"""

    # Gifs and
    num_gifs: int = 1
    save_dir: str | None = None
    gif_every: int = 10


@dataclass
class Config:
    exp: ExpConfig
    env: legal_envs
    agent: FrozenConfigDict = default_config
