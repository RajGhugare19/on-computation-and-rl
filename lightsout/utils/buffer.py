import jax
import flax
import functools
import jax.numpy as jnp

from jax import flatten_util

# Adapted from https://github.com/MichalBortkiewicz/JaxGCRL/blob/master/jaxgcrl/utils/replay_buffer.py

@flax.struct.dataclass
class ReplayBufferState:
    data: jnp.ndarray
    insert_position: jnp.ndarray
    sample_position: jnp.ndarray
    key: jax.random.PRNGKey

class TrajectoryUniformSamplingQueue():
    def __init__(
        self,
        max_replay_size: int,
        dummy_data_sample,
        sample_batch_size: int,
        num_envs: int,
        sequence_length: int,
    ):

        self._flatten_fn = jax.vmap(jax.vmap(lambda x: flatten_util.ravel_pytree(x)[0]))
        dummy_flatten, self._unflatten_fn = flatten_util.ravel_pytree(dummy_data_sample)
        self._unflatten_fn = jax.vmap(jax.vmap(self._unflatten_fn))
        data_size = len(dummy_flatten)

        self._data_shape = (max_replay_size, num_envs, data_size)
        self._data_dtype = dummy_flatten.dtype
        self._sample_batch_size = sample_batch_size
        self.num_envs = num_envs
        self.sequence_length = sequence_length

    def init(self, key):
        return ReplayBufferState(
            data=jnp.zeros(self._data_shape, self._data_dtype),
            sample_position=jnp.zeros((), jnp.int32),
            insert_position=jnp.zeros((), jnp.int32),
            key=key,
        )

    def insert(self, buffer_state, samples):
        if buffer_state.data.shape != self._data_shape:
            raise ValueError(
                f"buffer_state.data.shape ({buffer_state.data.shape}) "
                f"doesn't match the expected value ({self._data_shape})"
            )

        update = self._flatten_fn(samples) # updates has shape (unroll_len, num_envs, self._data_shape[-1])
        data = buffer_state.data # shape = (max_replay_size, num_envs, data_size)

        position = buffer_state.insert_position
        roll = jnp.minimum(0, len(data) - position - len(update))
        data = jax.lax.cond(roll, lambda: jnp.roll(data, roll, axis=0), lambda: data)
        position = position + roll

        # Update the buffer and the control numbers.
        data = jax.lax.dynamic_update_slice_in_dim(data, update, position, axis=0)
        position = (position + len(update)) % (len(data) + 1)
        sample_position = jnp.maximum(0, buffer_state.sample_position + roll)

        return buffer_state.replace(
            data=data,
            insert_position=position,
            sample_position=sample_position,
        )

    def sample(self, buffer_state):
        """Sample a batch of data."""
        if buffer_state.data.shape != self._data_shape:
            raise ValueError(
                f"Data shape expected by the replay buffer ({self._data_shape}) does "
                f"not match the shape of the buffer state ({buffer_state.data.shape})"
            )
        key, sample_key, shuffle_key = jax.random.split(buffer_state.key, 3)
        shape = self.num_envs

        envs_idxs = jax.random.choice(sample_key, jnp.arange(self.num_envs), shape=(shape,), replace=False)

        @functools.partial(jax.jit, static_argnames=("rows", "cols"))
        def create_matrix(rows, cols, min_val, max_val, rng_key):
            rng_key, subkey = jax.random.split(rng_key)
            start_values = jax.random.randint(subkey, shape=(rows,), minval=min_val, maxval=max_val)
            row_indices = jnp.arange(cols)
            matrix = start_values[:, jnp.newaxis] + row_indices
            return matrix

        @jax.jit
        def create_batch(arr_2d, indices):
            return jnp.take(arr_2d, indices, axis=0, mode="wrap")

        create_batch_vmaped = jax.vmap(create_batch, in_axes=(1, 0))

        matrix = create_matrix(
            shape,
            self.sequence_length,
            buffer_state.sample_position,
            buffer_state.insert_position - self.sequence_length,
            sample_key,
        )

        batch = create_batch_vmaped(buffer_state.data[:, envs_idxs, :], matrix)
        transitions = self._unflatten_fn(batch)
        return buffer_state.replace(key=key), transitions

    @staticmethod
    @functools.partial(jax.jit, static_argnames=("buffer_config"))
    def flatten_crl_fn(buffer_config, transition, sample_key):

        gamma, = buffer_config

        seq_len = transition.observation.shape[0]
        arrangement = jnp.arange(seq_len)
        is_future_mask = jnp.array(arrangement[:, None] < arrangement[None], dtype=jnp.float32)
        discount = gamma ** jnp.array(arrangement[None] - arrangement[:, None], dtype=jnp.float32)        
        probs = is_future_mask * discount  
        
        single_trajectories = jnp.concatenate(
            [transition.extras["state_extras"]["traj_id"][:, jnp.newaxis].T] * seq_len, axis=0
        )

        probs = probs * jnp.equal(single_trajectories, single_trajectories.T) + jnp.eye(seq_len) * 1e-5
        goal_index = jax.random.categorical(sample_key, jnp.log(probs))

        goal_obs = jnp.take(transition.achieved_goal, goal_index[:-1], axis=0) # the last goal_index cannot be considered as there is no future.  

        extras = {
            "state_extras": {
                "traj_id": jnp.squeeze(transition.extras["state_extras"]["traj_id"][:-1]),
            },
            "future_goal": goal_obs,
        }

        return transition._replace(
            observation=transition.observation[:-1],
            achieved_goal=transition.achieved_goal[:-1],
            action=jnp.squeeze(transition.action[:-1]),
            extras=extras,
        )
    
    @staticmethod
    @functools.partial(jax.jit, static_argnames=("buffer_config"))
    def flatten_sac_her_fn(buffer_config, transition, sample_key):

        gamma, = buffer_config

        seq_len = transition.observation.shape[0]
        arrangement = jnp.arange(seq_len)
        is_future_mask = jnp.array(arrangement[:, None] < arrangement[None], dtype=jnp.float32)
        discount = gamma ** jnp.array(arrangement[None] - arrangement[:, None], dtype=jnp.float32)        
        probs = is_future_mask * discount  
        
        single_trajectories = jnp.concatenate(
            [transition.extras["state_extras"]["traj_id"][:, jnp.newaxis].T] * seq_len, axis=0
        )

        probs = probs * jnp.equal(single_trajectories, single_trajectories.T) + jnp.eye(seq_len) * 1e-5
        goal_index = jax.random.categorical(sample_key, jnp.log(probs))

        goal_obs = jnp.take(transition.achieved_goal, goal_index[:-1], axis=0) 

        extras = {
            "state_extras": {
                "truncation": transition.extras["state_extras"]["truncation"][:-1],
                "traj_id": transition.extras["state_extras"]["traj_id"][:-1],
            },
            "future_goals": goal_obs,
        }

        return transition._replace(
            observation=transition.observation[:-1],
            commanded_goal=transition.commanded_goal[:-1],
            achieved_goal=transition.achieved_goal[:-1],
            action=transition.action[:-1],
            done=transition.done[:-1],
            next_observation=transition.next_observation[:-1],
            extras=extras,
        )

    def size(self, buffer_state):
        return (buffer_state.insert_position - buffer_state.sample_position)