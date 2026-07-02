import jax
from flax import struct

import re
from typing import Tuple, Any, Dict

@struct.dataclass
class State:
    """Environment state for training and inference."""
    data: Any                                   
    obs: jax.Array                                   
    reward: jax.Array                               
    done: jax.Array                                 
    metrics: Dict[str, jax.Array]                   
    info: Dict[str, Any]                            

def make_env(args):
    # Initialize environment
    if re.fullmatch(r"lightsout-\d+x\d+", args.env_id):
        dims = re.search(r"lightsout-(\d+)x(\d+)", args.env_id)
        m = int(dims.group(1))
        n = int(dims.group(2))

        from envs.lightsout_env import LightsOutEnv, default_config
        env_class = LightsOutEnv
        default_config = default_config()
        default_config.m = m
        default_config.n = n
        default_config.episode_length = (m * n)
        if args.difficulty_threshold is not None:
            default_config.difficulty_threshold = args.difficulty_threshold
    
    elif re.fullmatch(r"dense-lightsout-\d+x\d+", args.env_id):
        dims = re.search(r"dense-lightsout-(\d+)x(\d+)", args.env_id)
        m = int(dims.group(1))
        n = int(dims.group(2))

        from envs.lightsout_env import DenseLightsOutEnv, default_config
        env_class = DenseLightsOutEnv
        default_config = default_config()
        default_config.m = m
        default_config.n = n
        default_config.episode_length = (m * n)
        if args.difficulty_threshold is not None:
            default_config.difficulty_threshold = args.difficulty_threshold
    else:
        raise ValueError(f"Environment {args.env_id} not supported")

    return env_class, default_config