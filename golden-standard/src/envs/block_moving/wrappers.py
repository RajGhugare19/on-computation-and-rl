from .block_moving_env import BoxMovingEnv, BoxMovingState
import jax
import jax.numpy as jnp
from typing import Tuple, Dict, Any


class Wrapper(BoxMovingEnv):
    def __init__(self, env: BoxMovingEnv):
        self._env = env
        # Copy attributes from wrapped environment
        for attr in [
            "grid_size",
            "episode_length",
            "number_of_boxes_min",
            "number_of_boxes_max",
            "number_of_moving_boxes_max",
            "action_space",
            "level_generator",
        ]:
            if hasattr(env, attr):
                setattr(self, attr, getattr(env, attr))

    def reset(self, key: jax.Array) -> Tuple[BoxMovingState, Dict[str, Any]]:
        return self._env.reset(key)

    def step(self, state: BoxMovingState, action: int) -> Tuple[BoxMovingState, float, bool, Dict[str, Any]]:
        return self._env.step(state, action)


class AutoResetWrapper(Wrapper):
    def __init__(self, env: BoxMovingEnv):
        super().__init__(env)

    def reset_function(self, key):
        state, info = self._env.reset(key)
        extras_new = {**state.extras, "reset": jnp.bool_(False)}
        state = state.replace(extras=extras_new)
        return state, info

    def reset(self, key):
        state, info = self.reset_function(key)
        return state, info

    def step(self, state: BoxMovingState, action: int) -> Tuple[BoxMovingState, float, bool, Dict[str, Any]]:
        state, reward, done, info = self._env.step(state, action)
        key_new, _ = jax.random.split(state.key, 2)

        def reset_fn(key):
            reset_state, reset_info = self.reset_function(key)
            return reset_state, jnp.array(0.0).astype(jnp.float32), False, reset_info

        state, reward, done, info = jax.lax.cond(
            state.extras["reset"], lambda: reset_fn(key_new), lambda: (state, reward, done, info)
        )
        reset = jnp.logical_or(info["truncated"], done)
        extras_new = {**state.extras, "reset": reset}
        state = state.replace(extras=extras_new)
        return state, reward, done, info

    def get_dummy_timestep(self, key):
        default_dummy_timestep = super().get_dummy_timestep(key)
        return default_dummy_timestep.replace(extras={**default_dummy_timestep.extras, "reset": jnp.bool_(False)})


class SymmetryFilter(Wrapper):
    def __init__(self, env: BoxMovingEnv, axis="horizontal"):
        assert env.grid_size % 2 == 0  # For clarity and convenience
        self.axis = axis
        super().__init__(env)

    def check_symmetry_crossing(self, old_state: BoxMovingState, new_state: BoxMovingState):
        middle = self._env.grid_size // 2

        if self.axis == "horizontal":
            old_pos = old_state.agent_pos[0]
            new_pos = new_state.agent_pos[0]
        else:
            old_pos = old_state.agent_pos[1]
            new_pos = new_state.agent_pos[1]

        # If old_pos and new_pos are on different sides of the middle it means we've crossed the boundary,
        # and so we reset the environment
        return jnp.logical_xor(old_pos < middle, new_pos < middle)

    def step(self, state: BoxMovingState, action: int) -> Tuple[BoxMovingState, float, bool, Dict[str, Any]]:
        new_state, reward, done, info = self._env.step(state, action)
        is_truncated = jnp.logical_or(info["truncated"], self.check_symmetry_crossing(state, new_state))

        new_info = {**info, "truncated": is_truncated}

        return new_state, reward, done, new_info


class QuarterFilter(Wrapper):
    def __init__(self, env: BoxMovingEnv):
        assert env.grid_size % 2 == 0  # For clarity and convenience
        super().__init__(env)

    def check_wrong_quarter_crossing(self, new_state: BoxMovingState):
        fields_allowed = new_state.extras["fields_allowed"]
        agent_row, agent_col = new_state.agent_pos[0], new_state.agent_pos[1]

        return jnp.logical_not(fields_allowed[agent_row, agent_col])

    def step(self, state: BoxMovingState, action: int) -> Tuple[BoxMovingState, float, bool, Dict[str, Any]]:
        new_state, reward, done, info = self._env.step(state, action)
        is_truncated = jnp.logical_or(info["truncated"], self.check_wrong_quarter_crossing(new_state))

        new_info = {**info, "truncated": is_truncated}

        return new_state, reward, done, new_info


def wrap_for_training(config, env):
    if config.exp.filtering in ["horizontal", "vertical"]:
        env = SymmetryFilter(env, axis=config.exp.filtering)
    elif config.exp.filtering == "quarter":
        env = QuarterFilter(env)
    elif config.exp.filtering is None:
        pass
    else:
        raise ValueError(f"Unknown filtering type: {config.exp.filtering}")

    env = AutoResetWrapper(env)

    return env


def wrap_for_eval(env):
    env = AutoResetWrapper(env)

    return env
