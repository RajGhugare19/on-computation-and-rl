import jax
import jax.numpy as jnp
import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# adjust the import to where your module actually lives
from .block_moving_env import BoxMovingEnv
from .generators import DefaultLevelGenerator, create_solved_state, VariableQuarterGenerator
from .env_types import calculate_number_of_boxes, GridStatesEnum, BoxMovingState, remove_targets
from .wrappers import AutoResetWrapper, QuarterFilter


def test_remove_targets_mapping_all_values():
    """Ensure remove_targets maps every enum to the expected no-target state."""
    inp = jnp.arange(12, dtype=jnp.int8)  # values 0..11
    out = remove_targets(inp)
    expected = jnp.array([0, 1, 0, 3, 4, 5, 3, 4, 5, 11, 1, 11], dtype=jnp.int8)
    assert jnp.array_equal(out, expected), f"got {out}, expected {expected}"


def _count_boxes(grid: jnp.ndarray) -> int:
    """Return number of cells that contain a box (in any form: box, box on target, agent-on-box, etc.)."""
    box_states = jnp.array(
        [
            GridStatesEnum.BOX,
            GridStatesEnum.BOX_ON_TARGET,
            GridStatesEnum.AGENT_ON_BOX,
            GridStatesEnum.AGENT_ON_BOX_CARRYING_BOX,
            GridStatesEnum.AGENT_ON_TARGET_WITH_BOX,
            GridStatesEnum.AGENT_ON_TARGET_WITH_BOX_CARRYING_BOX,
        ],
        dtype=jnp.int8,
    )
    mask = jnp.zeros_like(grid, dtype=bool)
    for s in box_states:
        mask = mask | (grid == s)
    return int(jnp.sum(mask))


def _count_targets(grid: jnp.ndarray) -> int:
    """Return number of target cells (targets, box-on-target, or agent-on-target variants)."""
    target_states = jnp.array(
        [
            GridStatesEnum.TARGET,
            GridStatesEnum.BOX_ON_TARGET,
            GridStatesEnum.AGENT_ON_TARGET,
            GridStatesEnum.AGENT_ON_TARGET_CARRYING_BOX,
            GridStatesEnum.AGENT_ON_TARGET_WITH_BOX,
            GridStatesEnum.AGENT_ON_TARGET_WITH_BOX_CARRYING_BOX,
        ],
        dtype=jnp.int8,
    )
    mask = jnp.zeros_like(grid, dtype=bool)
    for s in target_states:
        mask = mask | (grid == s)
    return int(jnp.sum(mask))


@pytest.mark.parametrize("seed", [0, 7, 123])
def test_default_generator_box_and_target_counts(seed):
    """DefaultLevelGenerator should create exactly `number_of_boxes` boxes (in any state)
    and exactly `number_of_boxes` target cells (some targets may host boxes)."""
    gen = DefaultLevelGenerator(grid_size=5, number_of_boxes_min=2, number_of_boxes_max=4, number_of_moving_boxes_max=1)
    key = jax.random.PRNGKey(seed)
    state = gen.generate(key)
    grid = state.grid
    nboxes = int(state.number_of_boxes)
    assert _count_boxes(grid) == nboxes, f"boxes found {_count_boxes(grid)} != number_of_boxes {nboxes}"
    assert _count_targets(grid) == nboxes, f"targets found {_count_targets(grid)} != number_of_boxes {nboxes}"


@pytest.mark.parametrize("seed", [1, 11])
def test_quarter_generator_box_and_target_counts(seed):
    """QuarterGenerator should also produce matching counts for boxes and targets."""
    gen = VariableQuarterGenerator(
        grid_size=4,
        number_of_boxes_min=3,
        number_of_boxes_max=3,
        quarter_size=2,
        number_of_moving_boxes_max=1,
        special=False,
    )
    key = jax.random.PRNGKey(seed)
    state = gen.generate(key)
    grid = state.grid
    nboxes = int(state.number_of_boxes)
    assert _count_boxes(grid) == nboxes, f"boxes found {_count_boxes(grid)} != number_of_boxes {nboxes}"
    assert _count_targets(grid) == nboxes, f"targets found {_count_targets(grid)} != number_of_boxes {nboxes}"


def test_create_solved_state_transforms_targets_and_boxes_and_agent_cell1():
    """A small hand-crafted grid: TARGET at (0,0), BOX at (0,1), agent at (0,0).
    After create_solved_state:
      - the target cell should become BOX_ON_TARGET with the agent on it -> AGENT_ON_TARGET_WITH_BOX
      - BOX at (0,1) should be cleared to EMPTY
      - agent_has_box must be False
    """
    grid = jnp.array(
        [
            [GridStatesEnum.AGENT_ON_TARGET, GridStatesEnum.BOX, GridStatesEnum.TARGET],
            [GridStatesEnum.BOX_ON_TARGET, GridStatesEnum.EMPTY, GridStatesEnum.EMPTY],
            [GridStatesEnum.EMPTY, GridStatesEnum.BOX, GridStatesEnum.EMPTY],
        ],
        dtype=jnp.int8,
    )

    agent_pos = jnp.array([0, 0], dtype=jnp.int32)
    key = jax.random.PRNGKey(42)
    state = BoxMovingState(
        key=key,
        grid=grid,
        agent_pos=agent_pos,
        agent_has_box=jnp.array(True),
        steps=jnp.array(0),
        number_of_boxes=jnp.array(1),
        goal=jnp.zeros_like(grid),
        reward=jnp.array(0),
        success=jnp.array(0),
        extras={},
    )

    solved = create_solved_state(state)

    assert int(solved.grid[0, 0]) == int(GridStatesEnum.AGENT_ON_TARGET_WITH_BOX)
    assert int(solved.grid[0, 1]) == int(GridStatesEnum.EMPTY)
    assert int(solved.grid[0, 2]) == int(GridStatesEnum.BOX_ON_TARGET)
    assert int(solved.grid[1, 0]) == int(GridStatesEnum.BOX_ON_TARGET)
    assert calculate_number_of_boxes(state.grid) == 3
    assert calculate_number_of_boxes(solved.grid) == 3
    # agent_has_box should be cleared to False
    assert bool(solved.agent_has_box) is False


def test_create_solved_state_transforms_targets_and_boxes_and_agent_cell2():
    """A small hand-crafted grid: TARGET at (0,0), BOX at (0,1), agent at (0,0).
    After create_solved_state:
      - the target cell should become BOX_ON_TARGET with the agent on it -> AGENT_ON_TARGET_WITH_BOX
      - BOX at (0,1) should be cleared to EMPTY
      - agent_has_box must be False
    """
    grid = jnp.array(
        [
            [GridStatesEnum.AGENT_ON_TARGET_WITH_BOX, GridStatesEnum.BOX, GridStatesEnum.TARGET],
            [GridStatesEnum.BOX_ON_TARGET, GridStatesEnum.EMPTY, GridStatesEnum.EMPTY],
            [GridStatesEnum.AGENT, GridStatesEnum.EMPTY, GridStatesEnum.EMPTY],
        ],
        dtype=jnp.int8,
    )

    agent_pos = jnp.array([0, 0], dtype=jnp.int32)
    key = jax.random.PRNGKey(42)
    state = BoxMovingState(
        key=key,
        grid=grid,
        agent_pos=agent_pos,
        agent_has_box=jnp.array(True),
        steps=jnp.array(0),
        number_of_boxes=jnp.array(1),
        goal=jnp.zeros_like(grid),
        reward=jnp.array(0),
        success=jnp.array(0),
        extras={},
    )

    solved = create_solved_state(state)

    assert int(solved.grid[0, 0]) == int(GridStatesEnum.AGENT_ON_TARGET_WITH_BOX)
    assert int(solved.grid[0, 1]) == int(GridStatesEnum.EMPTY)
    assert int(solved.grid[0, 2]) == int(GridStatesEnum.BOX_ON_TARGET)
    assert int(solved.grid[1, 0]) == int(GridStatesEnum.BOX_ON_TARGET)
    assert calculate_number_of_boxes(state.grid) == 3
    assert calculate_number_of_boxes(solved.grid) == 3
    # agent_has_box should be cleared to False
    assert bool(solved.agent_has_box) is False


def test_create_solved_state_transforms_targets_and_boxes_and_agent_cell3():
    """A small hand-crafted grid: TARGET at (0,0), BOX at (0,1), agent at (0,0).
    After create_solved_state:
      - the target cell should become BOX_ON_TARGET with the agent on it -> AGENT_ON_TARGET_WITH_BOX
      - BOX at (0,1) should be cleared to EMPTY
      - agent_has_box must be False
    """
    grid = jnp.array(
        [
            [GridStatesEnum.AGENT_ON_BOX, GridStatesEnum.BOX, GridStatesEnum.TARGET],
            [GridStatesEnum.BOX_ON_TARGET, GridStatesEnum.EMPTY, GridStatesEnum.EMPTY],
            [GridStatesEnum.AGENT, GridStatesEnum.TARGET, GridStatesEnum.EMPTY],
        ],
        dtype=jnp.int8,
    )

    agent_pos = jnp.array([0, 0], dtype=jnp.int32)
    key = jax.random.PRNGKey(42)
    state = BoxMovingState(
        key=key,
        grid=grid,
        agent_pos=agent_pos,
        agent_has_box=jnp.array(True),
        steps=jnp.array(0),
        number_of_boxes=jnp.array(1),
        goal=jnp.zeros_like(grid),
        reward=jnp.array(0),
        success=jnp.array(0),
        extras={},
    )

    solved = create_solved_state(state)

    assert int(solved.grid[0, 0]) == int(GridStatesEnum.AGENT)
    assert int(solved.grid[0, 1]) == int(GridStatesEnum.EMPTY)
    assert int(solved.grid[0, 2]) == int(GridStatesEnum.BOX_ON_TARGET)
    assert int(solved.grid[1, 0]) == int(GridStatesEnum.BOX_ON_TARGET)
    assert calculate_number_of_boxes(state.grid) == 3
    assert calculate_number_of_boxes(solved.grid) == 3
    # agent_has_box should be cleared to False
    assert bool(solved.agent_has_box) is False


def test_create_solved_state_transforms_targets_and_boxes_and_agent_cell4():
    """A small hand-crafted grid: TARGET at (0,0), BOX at (0,1), agent at (0,0).
    After create_solved_state:
      - the target cell should become BOX_ON_TARGET with the agent on it -> AGENT_ON_TARGET_WITH_BOX
      - BOX at (0,1) should be cleared to EMPTY
      - agent_has_box must be False
    """
    grid = jnp.array(
        [
            [GridStatesEnum.AGENT, GridStatesEnum.BOX, GridStatesEnum.TARGET],
            [GridStatesEnum.BOX_ON_TARGET, GridStatesEnum.EMPTY, GridStatesEnum.EMPTY],
            [GridStatesEnum.AGENT, GridStatesEnum.EMPTY, GridStatesEnum.EMPTY],
        ],
        dtype=jnp.int8,
    )

    agent_pos = jnp.array([0, 0], dtype=jnp.int32)
    key = jax.random.PRNGKey(42)
    state = BoxMovingState(
        key=key,
        grid=grid,
        agent_pos=agent_pos,
        agent_has_box=jnp.array(True),
        steps=jnp.array(0),
        number_of_boxes=jnp.array(1),
        goal=jnp.zeros_like(grid),
        reward=jnp.array(0),
        success=jnp.array(0),
        extras={},
    )

    solved = create_solved_state(state)

    assert int(solved.grid[0, 0]) == int(GridStatesEnum.AGENT)
    assert int(solved.grid[0, 1]) == int(GridStatesEnum.EMPTY)
    assert int(solved.grid[0, 2]) == int(GridStatesEnum.BOX_ON_TARGET)
    assert int(solved.grid[1, 0]) == int(GridStatesEnum.BOX_ON_TARGET)
    assert calculate_number_of_boxes(state.grid) == 2
    assert calculate_number_of_boxes(solved.grid) == 2
    # agent_has_box should be cleared to False
    assert bool(solved.agent_has_box) is False


if __name__ == "__main__":
    env = BoxMovingEnv(
        grid_size=6,
        number_of_boxes_max=1,
        number_of_boxes_min=1,
        number_of_moving_boxes_max=1,
        level_generator="variable",
        generator_special=False,
        dense_rewards=False,
        terminate_when_success=True,
        episode_length=10,
        quarter_size=1,
    )
    env = QuarterFilter(env)
    env = AutoResetWrapper(env)
    key = jax.random.PRNGKey(0)
    env.play_game(key)
