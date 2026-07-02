import jax
import jax.numpy as jnp
from flax import struct
from typing import Any, Dict, Optional, Tuple, Union
from ml_collections import config_dict
import numpy as np

from envs.utils import State

@struct.dataclass
class LightsOutData:
    grid: jax.Array
    goal: jax.Array

def default_config() -> config_dict.ConfigDict:
    config = config_dict.create(
        m=3,
        n=3,
        episode_length=6,
        difficulty_threshold=0.5, # <--- Goals with < 4 toggles are "Easy"
    )
    return config

class LightsOutEnv:
    def __init__(self, config: config_dict.ConfigDict = default_config()):
        self._config = config
        self.m, self.n = config.m, config.n
        self.grid_size = self.m * self.n
        self.threshold = int(self.grid_size * config.difficulty_threshold)
        
        # Pre-compute action matrix: matrix[i, j] = 1 if button i toggles button j
        self.action_matrix = self._create_action_matrix()

    def _create_action_matrix(self) -> jnp.ndarray:
        matrix = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        for i in range(self.grid_size):
            r, c = i // self.n, i % self.n
            matrix[i, i] = 1  # Toggle self
            if r > 0: matrix[i, (r-1)*self.n + c] = 1 # Up
            if r < self.m-1: matrix[i, (r+1)*self.n + c] = 1 # Down
            if c > 0: matrix[i, r*self.n + (c-1)] = 1 # Left
            if c < self.n-1: matrix[i, r*self.n + (c+1)] = 1 # Right
        return jnp.array(matrix)

    @property
    def action_size(self): return self.grid_size

    @property
    def observation_size(self): return self.grid_size

    @property
    def goal_size(self): return self.grid_size

    def reset(self, rng: jax.Array) -> State:
        # 1. Split key for distance sampling
        rng_dist, rng_reset = jax.random.split(rng)
        
        # 2. Sample an "Easy" distance: [1, threshold)
        dist = jax.random.randint(
            rng_dist, (), 1, self.threshold
        )
        
        return self._reset_with_dist(rng_reset, dist)

    def eval_reset(self, rng: jax.Array) -> State:
        # 1. Split key for distance sampling
        rng_dist, rng_reset = jax.random.split(rng)
        
        # 2. Sample a "Hard" distance: [threshold, grid_size)
        dist = jax.random.randint(
            rng_dist, (), self.threshold, self.grid_size
        )
        
        return self._reset_with_dist(rng_reset, dist)

    def _reset_with_dist(self, rng: jax.Array, dist: int) -> State:
        key_init, key_actions = jax.random.split(rng)
        
        initial = jax.random.bernoulli(key_init, shape=(self.grid_size,)).astype(jnp.int32)
        
        # Create action coefficients with exactly 'dist' number of ones
        indices = jax.random.permutation(key_actions, self.grid_size)
        mask = jnp.arange(self.grid_size) < dist
        action_coeffs = jnp.zeros(self.grid_size, dtype=jnp.int32).at[indices].set(mask)
        
        goal = (initial + jnp.dot(self.action_matrix.T, action_coeffs)) % 2
        
        data = LightsOutData(grid=initial.astype(jnp.float32), goal=goal.astype(jnp.float32))
        obs = data.grid
        
        info = {
            "rng": rng,
            "target_goal": data.goal,
        }
        
        return State(
            data=data, obs=obs,
            reward=jnp.array(0.0, jnp.float32),
            done=jnp.array(0.0, jnp.float32),
            metrics={"success": 0.0, "reward": 0.0},
            info=info
        )
    
    def step(self, state: State, action: jax.Array) -> State:
        data = state.data
        
        # Apply toggle pattern (action is an integer index)
        toggle_pattern = self.action_matrix[action]
        new_grid = (data.grid.astype(jnp.int32) + toggle_pattern) % 2
        
        new_data = LightsOutData(grid=new_grid.astype(jnp.float32), goal=data.goal)
        
        # Reward logic: solved if grids match
        success = jnp.all(new_grid == data.goal.astype(jnp.int32)).astype(jnp.float32)
        
        # reward = 1.0 if solved, else a small penalty to encourage speed
        reward = success - 0.1
        done = success
        
        new_info = {**state.info, "target_goal": data.goal}
        new_metrics = {**state.metrics, "success": success, "reward": reward}
        
        return State(
            data=new_data,
            obs=new_data.grid,
            reward=reward,
            done=done,
            metrics=new_metrics,
            info=new_info
        )
    
class DenseLightsOutEnv(LightsOutEnv):
    def __init__(self, config: config_dict.ConfigDict = default_config()):
        super().__init__(config)
        self.inv_action_matrix = self._get_inv_matrix_gf2(self.action_matrix)

    def _get_inv_matrix_gf2(self, A):
        """Computes the inverse of a square matrix over GF(2)."""
        import numpy as np
        n = A.shape[0]
        # Augment with identity: [A | I]
        A_aug = np.hstack([np.array(A), np.eye(n, dtype=int)])
        
        # Gaussian elimination
        for i in range(n):
            # Find pivot
            pivot = np.where(A_aug[i:, i] == 1)[0]
            if len(pivot) == 0:
                continue # Singular matrix handling
            pivot = pivot[0] + i
            # Swap rows
            A_aug[[i, pivot]] = A_aug[[pivot, i]]
            # Eliminate other rows
            for j in range(n):
                if i != j and A_aug[j, i] == 1:
                    A_aug[j] = (A_aug[j] + A_aug[i]) % 2
        
        # The right half is the inverse
        return jnp.array(A_aug[:, n:], dtype=jnp.int32)

    def _get_dist_to_goal(self, grid, goal):
        # b = (grid + goal) % 2
        b = (grid.astype(jnp.int32) + goal.astype(jnp.int32)) % 2
        # Solve A*x = b mod 2 -> x = inv(A)*b mod 2
        # x represents the required button presses
        required_presses = jnp.dot(self.inv_action_matrix, b) % 2
        return jnp.sum(required_presses)

    def step(self, state: State, action: jax.Array) -> State:
        data = state.data
        
        # Calculate current distance (potential)
        curr_dist = self._get_dist_to_goal(data.grid, data.goal)
        
        # Apply toggle
        toggle_pattern = self.action_matrix[action]
        new_grid = (data.grid.astype(jnp.int32) + toggle_pattern) % 2
        
        # Calculate new distance
        new_dist = self._get_dist_to_goal(new_grid, data.goal)
        
        # Dense Progress Reward
        progress_reward = (curr_dist - new_dist).astype(jnp.float32)
        
        new_data = LightsOutData(grid=new_grid.astype(jnp.float32), goal=data.goal)
        success = jnp.all(new_grid == data.goal.astype(jnp.int32)).astype(jnp.float32)
        
        # reward = progress (+1 or -1) + success bonus
        reward = progress_reward + (success * 1.0)
        done = success
        
        return State(
            data=new_data,
            obs=new_data.grid,
            reward=reward,
            done=done,
            metrics={**state.metrics, "success": success, "reward": reward},
            info={**state.info, "target_goal": data.goal}
        )
