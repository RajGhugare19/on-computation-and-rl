from .env_types import BoxMovingState, GridStatesEnum, TimeStep
import jax
import jax.numpy as jnp
from jax import random


def create_solved_state(state: BoxMovingState) -> BoxMovingState:
    """Create a solved state."""
    # Change all target cells to box on target
    target_mask = (state.grid == GridStatesEnum.TARGET) | (state.grid == GridStatesEnum.BOX_ON_TARGET)
    state = state.replace(grid=jnp.where(target_mask, GridStatesEnum.BOX_ON_TARGET, state.grid))
    # Change all boxes to empty - use where to avoid boolean indexing issue
    box_mask = state.grid == GridStatesEnum.BOX
    state = state.replace(grid=jnp.where(box_mask, GridStatesEnum.EMPTY, state.grid))

    # Check what cell the agent is currently on and update accordingly
    agent_row, agent_col = state.agent_pos[0], state.agent_pos[1]
    current_cell = state.grid[agent_row, agent_col]

    # Update grid based on current cell type
    new_cell_value = jax.lax.cond(
        current_cell == GridStatesEnum.AGENT_ON_TARGET,
        lambda: GridStatesEnum.AGENT_ON_TARGET_WITH_BOX,
        lambda: jax.lax.cond(
            current_cell == GridStatesEnum.AGENT_ON_BOX,
            lambda: GridStatesEnum.AGENT,
            lambda: current_cell,  # noqa: E501 Here goes: AGENT_ON_TARGET_WITH_BOX -> AGENT_ON_TARGET_WITH_BOX and Agent -> Agent
        ),
    )
    state = state.replace(grid=state.grid.at[agent_row, agent_col].set(new_cell_value), agent_has_box=jnp.array(False))
    return state


def find_agent_position(grid) -> jnp.array:
    actor_states = jnp.isin(
        grid,
        jnp.array(
            [
                GridStatesEnum.AGENT,
                GridStatesEnum.AGENT_ON_BOX,
                GridStatesEnum.AGENT_ON_TARGET,
                GridStatesEnum.AGENT_ON_TARGET_WITH_BOX,
            ]
        ),
    )
    idx = jnp.argmax(actor_states)

    row_idx = idx // grid.shape[1]
    col_idx = idx % grid.shape[1]

    return jnp.array([row_idx, col_idx])


class DefaultLevelGenerator:
    def __init__(self, grid_size, number_of_boxes_min, number_of_boxes_max, number_of_moving_boxes_max):
        self.grid_size = grid_size
        self.number_of_boxes_min = number_of_boxes_min
        self.number_of_boxes_max = number_of_boxes_max
        self.number_of_moving_boxes_max = number_of_moving_boxes_max

    def place_agent(self, grid, agent_key):
        agent_pos = random.randint(agent_key, (2,), 0, grid.shape[0])
        current_cell = grid[agent_pos[0], agent_pos[1]]

        # TODO: This is ugly, maybe it can be refactored somehow ?
        agent_state = jax.lax.cond(
            current_cell == GridStatesEnum.EMPTY,
            lambda: GridStatesEnum.AGENT,
            lambda: jax.lax.cond(
                current_cell == GridStatesEnum.BOX,
                lambda: GridStatesEnum.AGENT_ON_BOX,
                lambda: jax.lax.cond(
                    current_cell == GridStatesEnum.TARGET,
                    lambda: GridStatesEnum.AGENT_ON_TARGET,
                    lambda: jax.lax.cond(
                        current_cell == GridStatesEnum.BOX_ON_TARGET,
                        lambda: GridStatesEnum.AGENT_ON_TARGET_WITH_BOX,
                        lambda: GridStatesEnum.AGENT_ON_TARGET_WITH_BOX,  # This should never happen
                    ),
                ),
            ),
        )

        new_grid = grid.at[agent_pos[0], agent_pos[1]].set(agent_state)
        return agent_pos, new_grid

    def generate(self, key) -> BoxMovingState:
        permutation_key, number_of_boxes_key, agent_key, state_key = random.split(key, 4)

        idxs = jnp.arange(self.grid_size * self.grid_size)

        number_of_boxes = jax.random.randint(
            number_of_boxes_key, (), self.number_of_boxes_min, self.number_of_boxes_max + 1
        )
        number_of_boxes_on_target = jnp.maximum(0, number_of_boxes - self.number_of_moving_boxes_max)
        number_of_targets_without_boxes = number_of_boxes - number_of_boxes_on_target

        is_fixed = idxs < number_of_boxes_on_target
        is_box = (idxs >= number_of_boxes_on_target) & (idxs < number_of_boxes)
        is_target = (idxs >= number_of_boxes) & (idxs < number_of_boxes + number_of_targets_without_boxes)

        # TODO: this could be made more efficient by only looping over indices up to 2 * number_of_boxes_max, and then concatenating with empty rest of the board
        # this shouldn't matter with board sizes we work now, but worth remembering in the future
        grid = jnp.piecewise(
            idxs,
            [is_fixed, is_box, is_target],
            [GridStatesEnum.BOX_ON_TARGET, GridStatesEnum.BOX, GridStatesEnum.TARGET, GridStatesEnum.EMPTY],
        ).astype(jnp.int8)

        grid = jax.random.permutation(permutation_key, grid)
        grid = grid.reshape((self.grid_size, self.grid_size))

        # Agent is placed at any field randomly
        agent_pos, grid = self.place_agent(grid, agent_key)

        state = BoxMovingState(
            key=state_key,
            grid=grid,
            agent_pos=agent_pos,
            agent_has_box=False,
            steps=0,
            number_of_boxes=number_of_boxes,
            goal=jnp.zeros_like(grid),
            reward=0.0,
            success=0,
            extras={},
        )

        goal = create_solved_state(state)
        state = state.replace(goal=goal.grid)

        return state

    def get_dummy_timestep(self, key):
        return TimeStep(
            key=key,
            grid=jnp.zeros((self.grid_size, self.grid_size), dtype=jnp.int8),
            number_of_boxes=jnp.zeros((1,), dtype=jnp.int8),
            agent_pos=jnp.zeros((2,), dtype=jnp.int8),
            agent_has_box=jnp.zeros((1,), dtype=jnp.int8),
            steps=jnp.zeros((1,), dtype=jnp.int8),
            action=jnp.zeros((1,), dtype=jnp.int8),
            goal=jnp.zeros((self.grid_size, self.grid_size), dtype=jnp.int8),
            reward=jnp.zeros((1,), dtype=jnp.float32),
            success=jnp.zeros((1,), dtype=jnp.int8),
            done=jnp.zeros((1,), dtype=jnp.int8),
            truncated=jnp.zeros((1,), dtype=jnp.int8),
            extras={},
        )


class VariableQuarterGenerator(DefaultLevelGenerator):
    def __init__(
        self,
        grid_size,
        number_of_boxes_min,
        number_of_boxes_max,
        number_of_moving_boxes_max,
        quarter_size,
        special=False,
    ):
        # This is mostly for convenience, without it there would have to be a lot of if statements
        assert number_of_boxes_max <= quarter_size * quarter_size
        assert number_of_boxes_max == number_of_moving_boxes_max, "In this generator we assume all boxes always move"
        assert number_of_boxes_min == number_of_boxes_max, (
            "In this generator we assume there is only one possible number of boxes"
        )
        self.special = special
        self.quarter_size = quarter_size

        super().__init__(grid_size, number_of_boxes_min, number_of_boxes_max, number_of_moving_boxes_max)

    def generate_box_quarter(self, number_of_boxes, key):
        quarter = jnp.full(self.quarter_size * self.quarter_size, GridStatesEnum.EMPTY)
        idxs = jnp.arange(self.quarter_size * self.quarter_size)

        is_box = idxs < number_of_boxes

        quarter = jnp.piecewise(idxs, [is_box], [GridStatesEnum.BOX, GridStatesEnum.EMPTY]).astype(jnp.int8)

        quarter = jax.random.permutation(key, quarter)
        quarter = quarter.reshape((self.quarter_size, self.quarter_size))

        return quarter

    def place_targets_in_slice(self, grid, number_of_targets, key):
        available_indices = jnp.nonzero(grid == GridStatesEnum.EMPTY, size=grid.shape[0] * grid.shape[1], fill_value=-1)
        available_indices = jnp.stack(available_indices, axis=1)
        available_indices = jax.random.permutation(key, available_indices)

        # This is ugly, but I don't see how to achieve the same without for loop
        def f(carry, index):
            _, current_num_targets = carry

            new_carry = jax.lax.cond(
                jnp.logical_and(current_num_targets > 0, index[0] >= 0),
                lambda _grid, _curr_num_targets: (
                    _grid.at[index[0], index[1]].set(GridStatesEnum.TARGET),
                    _curr_num_targets - 1,
                ),
                lambda x, y: (x, y),
                *carry,
            )
            return new_carry, None

        final_carry, _ = jax.lax.scan(f, (grid, number_of_targets), xs=available_indices)

        return final_carry[0]

    def generate(self, key):
        box_quarter_key, target_quarter_key, permutation_3_key, number_of_boxes_key, agent_key, state_key = (
            jax.random.split(key, 6)
        )

        number_of_boxes = self.number_of_boxes_max

        if self.special:
            corners = jnp.array(
                [
                    [0, 3],
                    [3, 0],
                    [1, 2],
                    [2, 1],
                ]
            )
        else:
            corners = jnp.array(
                [
                    [0, 1],
                    [1, 0],
                    [0, 2],
                    [2, 0],
                    [1, 3],
                    [3, 1],
                    [2, 3],
                    [3, 2],
                ]
            )

        corners = jax.random.choice(permutation_3_key, corners)
        corners_left_upper = jnp.array(
            [
                [0, 0],
                [0, self.grid_size - self.quarter_size],
                [self.grid_size - self.quarter_size, 0],
                [self.grid_size - self.quarter_size, self.grid_size - self.quarter_size],
            ]
        )

        box_corner = corners_left_upper[corners[0]]
        target_corner = corners_left_upper[corners[1]]

        grid = jnp.full(self.grid_size * self.grid_size, GridStatesEnum.EMPTY).reshape(self.grid_size, self.grid_size)
        box_slice = self.generate_box_quarter(self.number_of_boxes_max, box_quarter_key)

        grid = jax.lax.dynamic_update_slice(grid, box_slice, box_corner)
        target_slice = jax.lax.dynamic_slice(grid, target_corner, (self.quarter_size, self.quarter_size))
        target_slice = self.place_targets_in_slice(target_slice, self.number_of_boxes_max, target_quarter_key)

        grid = jax.lax.dynamic_update_slice(grid, target_slice, target_corner)

        possible_agent_placements = jax.lax.dynamic_slice(grid, box_corner, (self.quarter_size, self.quarter_size))
        agent_pos, updated_grid_slice = self.place_agent(possible_agent_placements, agent_key)
        grid = jax.lax.dynamic_update_slice(grid, updated_grid_slice, box_corner)

        agent_pos = agent_pos + box_corner

        up = jnp.minimum(box_corner[0], target_corner[0])
        down = jnp.maximum(box_corner[0], target_corner[0]) + self.quarter_size
        left = jnp.minimum(box_corner[1], target_corner[1])
        right = jnp.maximum(box_corner[1], target_corner[1]) + self.quarter_size

        coords = jnp.indices((self.grid_size, self.grid_size))
        fields_allowed = jnp.logical_and(
            jnp.logical_and(coords[0] < down, coords[0] >= up), jnp.logical_and(coords[1] < right, coords[1] >= left)
        )

        state = BoxMovingState(
            key=state_key,
            grid=grid,
            agent_pos=agent_pos,
            agent_has_box=False,
            steps=0,
            number_of_boxes=number_of_boxes,
            goal=jnp.zeros_like(grid),
            reward=0.0,
            success=0,
            extras={"fields_allowed": fields_allowed},
        )

        goal = create_solved_state(state)
        state = state.replace(goal=goal.grid)

        return state

    def get_dummy_timestep(self, key):
        default_dummy_timestep = super().get_dummy_timestep(key)
        return default_dummy_timestep.replace(
            extras={"fields_allowed": jnp.zeros((self.grid_size, self.grid_size), dtype=jnp.bool)}
        )
