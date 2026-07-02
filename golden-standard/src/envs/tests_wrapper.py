# src/envs/test_autoreset_minimal.py
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from .block_moving.wrappers import AutoResetWrapper
from .block_moving.env_types import BoxMovingState


def make_state(key, extras=None, grid_size=4):
    if extras is None:
        extras = {}
    return BoxMovingState(
        key=key,
        grid=jnp.zeros((grid_size, grid_size), dtype=jnp.int8),
        agent_pos=jnp.array([0, 0], dtype=jnp.int32),
        agent_has_box=jnp.array(False),
        steps=jnp.array(0),
        number_of_boxes=jnp.array(1),
        goal=jnp.zeros((grid_size, grid_size), dtype=jnp.int8),
        reward=jnp.array(0.0, dtype=jnp.float32),
        success=jnp.array(0),
        extras=extras,
    )


class DummyInnerEnv:
    """Minimal inner env used to assert that reset() was called or not called."""

    def __init__(self):
        # attributes copied by Wrapper.__init__
        self.grid_size = 4
        self.episode_length = 100
        self.number_of_boxes_min = 1
        self.number_of_boxes_max = 1
        self.number_of_moving_boxes_max = 1
        self.action_space = 6
        self.level_generator = None

        self.reset_called_with = []
        self.step_called_with = []

        # control what reset returns (state, info) - keep extras empty to avoid cond-pytree mismatch
        self.reset_return = (make_state(jax.random.PRNGKey(1234), extras={}), {"truncated": jnp.bool_(False)})
        # by default step echoes the input state and returns small reward, not done, not truncated
        self.step_return = None

    def reset(self, key):
        self.reset_called_with.append(key)
        return self.reset_return

    def step(self, state, action):
        self.step_called_with.append((state, action))
        if self.step_return is not None:
            return self.step_return
        # echo state so wrapper sees same extras
        return state, jnp.array(0.1, dtype=jnp.float32), False, {"truncated": jnp.bool_(False)}


def jnp_bool(v: bool) -> jnp.bool_:
    return jnp.array(v).astype(jnp.bool_)


def test_autoreset_triggers_inner_reset():
    """
    Minimal check: when incoming state's extras['reset'] is True, AutoResetWrapper.step()
    should call inner.reset(...) and return the reset state (with reward 0.0 and done False).
    """
    pk = jax.random.PRNGKey(0)

    # incoming state has reset flag True and no other extras keys (so pytree shapes match)
    incoming_state = make_state(pk, extras={"reset": jnp_bool(True)})

    inner = DummyInnerEnv()
    wrapper = AutoResetWrapper(inner)

    out_state, out_reward, out_done, out_info = wrapper.step(incoming_state, action=0)

    # inner.reset must have been called exactly once with the split key from incoming_state.key
    expected_split_key, _ = jax.random.split(incoming_state.key, 2)
    assert len(inner.reset_called_with) == 1
    np.testing.assert_array_equal(np.array(inner.reset_called_with[0]), np.array(expected_split_key))

    # wrapper returns reward 0.0 and done False for reset events
    assert float(out_reward) == 0.0
    assert out_done == jnp_bool(False)

    # wrapper.reset_function sets extras["reset"] = False on the reset state
    assert "reset" in out_state.extras
    assert out_state.extras["reset"] == jnp_bool(False)


def test_autoreset_does_not_trigger_inner_reset():
    """
    If incoming state's extras['reset'] is False and inner.step returns not truncated and not done,
    AutoResetWrapper.step() should NOT apply inner.reset(...) result and should return inner.step's outputs.
    (We avoid asserting Python-side reset call counts because JAX tracing may call reset() while tracing.)
    """
    pk = jax.random.PRNGKey(1)

    # incoming state has reset flag False and includes a 'marker' key so null-reset branch has same pytree shape
    incoming_state = make_state(pk, extras={"reset": jnp_bool(False), "marker": jnp.int32(0)})

    inner = DummyInnerEnv()
    # Make inner.reset return a state that would be easy to detect if applied
    inner.reset_return = (
        make_state(jax.random.PRNGKey(999), extras={"marker": jnp.int32(999)}),
        {"truncated": jnp.bool_(False)},
    )

    # make inner.step return a distinct reward and done False, truncated False (the expected no-reset outcome)
    inner.step_return = (incoming_state, jnp.array(0.55, dtype=jnp.float32), False, {"truncated": jnp.bool_(False)})

    wrapper = AutoResetWrapper(inner)

    out_state, out_reward, out_done, out_info = wrapper.step(incoming_state, action=2)

    # Do NOT assert on inner.reset_called_with (unreliable due to JAX tracing).
    # Instead assert the wrapper returned the inner.step output (reward/done).
    assert float(out_reward) == pytest.approx(0.55, rel=1e-6)
    assert out_done == jnp_bool(False)

    # Crucially: the returned state's 'marker' was NOT replaced with reset state's marker (999).
    # If reset had been applied as the runtime result, we'd see 999 here.
    assert "marker" in out_state.extras
    assert int(out_state.extras["marker"]) == 0

    # wrapper will still maintain an extras['reset'] boolean (False in this case)
    assert "reset" in out_state.extras
    assert out_state.extras["reset"] == jnp_bool(False)
