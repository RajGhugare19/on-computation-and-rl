import time
import mujoco
import jax
import jax.numpy as jnp
import numpy as np

from typing import Callable

from models.networks import ScannedRNN
from utils.wrapper import EvalWrapper

def get_trajectory(policy, env, key, unroll_length):
    @jax.jit
    def _get_trajectory(key):
        env_key, key = jax.random.split(key)
        eval_first_state = env.reset(jax.random.split(env_key, 1))
        
        def f(carry, unused_t):
            env_state, key = carry
            key, next_key = jax.random.split(key)
            action, _ = policy(env_state.obs, env_state.info["target_goal"], key) 
            next_env_state = env.step(env_state, action)

            return (next_env_state, next_key), next_env_state
        
        _, env_states = jax.lax.scan(
                f, 
                (eval_first_state, key), 
                (), 
                length=unroll_length
            )
        return env_states
    
    return _get_trajectory(key)

def generate_unroll(env, env_state, policy, key, unroll_length):
    """Step the env for unroll_length steps and return final state."""
    def body(i, carry):
        env_state, key = carry
        key, next_key = jax.random.split(key)
        actions, _ = policy(env_state.obs, env_state.info["target_goal"], key)
        next_env_state = env.step(env_state, actions)
        return (next_env_state, next_key)

    final_state, _ = jax.lax.fori_loop(
        0, unroll_length, body, (env_state, key)
    )
    return final_state

def generate_rnn_unroll(env, env_state, policy, key, unroll_length, rnn_state_dim, num_envs):
    """Step the env for unroll_length steps and return final state."""
    
    rnn_state = ScannedRNN.initialize_carry(
        num_envs, rnn_state_dim,
    )
    def body(i, carry):
        env_state, rnn_state, key = carry
        key, next_key = jax.random.split(key)
        rnn_state, actions, _, _ = policy(rnn_state, env_state.obs, env_state.info['target_goal'], env_state.done, key)
        next_env_state = env.step(env_state, actions)
        return (next_env_state, rnn_state, next_key)

    final_state, _, _ = jax.lax.fori_loop(
        0, unroll_length, body, (env_state, rnn_state, key)
    )
    return final_state

# class Evaluator:
#     def __init__(
#         self,
#         eval_env,
#         eval_policy_fn: Callable,
#         num_eval_envs: int,
#         episode_length: int,
#         key: jax.Array,
#         rnn: bool = False,
#         rnn_state_dim: int = 256,
#     ):
    
#         self._key = key
#         self._eval_walltime = 0.0

#         eval_env = EvalWrapper(eval_env)

#         def generate_eval_unroll(policy_params, key):
#             reset_keys = jax.random.split(key, num_eval_envs)
#             eval_first_state = eval_env.reset(reset_keys)
#             if rnn:
#                 return generate_rnn_unroll(
#                     eval_env,
#                     eval_first_state,
#                     eval_policy_fn(policy_params),
#                     key,
#                     unroll_length=episode_length,
#                     rnn_state_dim=rnn_state_dim,
#                     num_envs=num_eval_envs,
#                 )
        
#             else:
#                 return generate_unroll(
#                     eval_env,
#                     eval_first_state,
#                     eval_policy_fn(policy_params),
#                     key,
#                     unroll_length=episode_length,
#                 )
            
#         self._generate_eval_unroll = jax.jit(generate_eval_unroll)
#         self._steps_per_unroll = episode_length * num_eval_envs

#     def run_evaluation(
#         self,
#         policy_params,
#         training_metrics,
#     ):
#         """Run one epoch of evaluation."""
#         self._key, unroll_key = jax.random.split(self._key)

#         t = time.time()
#         eval_state = self._generate_eval_unroll(policy_params, unroll_key)

#         eval_metrics = eval_state.info['eval_metrics']
#         eval_metrics.active_episodes.block_until_ready()
#         epoch_eval_time = time.time() - t
#         metrics = {}

#         for name, value in eval_metrics.episode_metrics.items():

#             metrics.update({
#                 f'eval/episode_{name}': (
#                     np.mean(value)
#                 )
#             })

#             if name in ["success", "easy_success", "hard_success"]:
#                 value_rate_means = np.mean(value > 0)
#                 metrics[f'eval/episode_{name}_rate'] = value_rate_means

#         metrics['eval/avg_episode_length'] = np.mean(eval_metrics.episode_steps)
#         metrics['eval/epoch_eval_time'] = epoch_eval_time
#         metrics['eval/sps'] = self._steps_per_unroll / epoch_eval_time
#         self._eval_walltime = self._eval_walltime + epoch_eval_time
#         metrics = {
#             'eval/walltime': self._eval_walltime,
#             **training_metrics,
#             **metrics,
#         }

#         return metrics

class Evaluator:
    def __init__(
        self,
        eval_env,
        eval_policy_fn: Callable,
        num_eval_envs: int,
        episode_length: int,
        key: jax.Array,
        rnn: bool = False,
        rnn_state_dim: int = 256,
    ):
    
        self._key = key
        self._eval_walltime = 0.0
        # Wrap the environment with EvalWrapper for metric tracking
        eval_env = EvalWrapper(eval_env)

        # Helper to define the unroll logic for either reset or eval_reset
        def make_unroll_fn(reset_type: str):
            def generate_unroll_logic(policy_params, key):
                reset_keys = jax.random.split(key, num_eval_envs)
                
                # Dynamically call reset() or eval_reset()
                reset_fn = getattr(eval_env, reset_type)
                eval_first_state = reset_fn(reset_keys)
                
                if rnn:
                    return generate_rnn_unroll(
                        eval_env,
                        eval_first_state,
                        eval_policy_fn(policy_params),
                        key,
                        unroll_length=episode_length,
                        rnn_state_dim=rnn_state_dim,
                        num_envs=num_eval_envs,
                    )
                else:
                    return generate_unroll(
                        eval_env,
                        eval_first_state,
                        eval_policy_fn(policy_params),
                        key,
                        unroll_length=episode_length,
                    )
            return jax.jit(generate_unroll_logic)

        # Create two separate JIT'd functions
        self._generate_train_unroll = make_unroll_fn("reset")
        self._generate_test_unroll = make_unroll_fn("eval_reset")
        
        self._steps_per_unroll = episode_length * num_eval_envs

    def run_evaluation(
        self,
        policy_params,
        training_metrics,
    ):
        """Run evaluation on both train and test splits."""
        self._key, train_key, test_key = jax.random.split(self._key, 3)

        t = time.time()
        
        # 1. Run Trajectories for Train Split (In-Distribution)
        train_state = self._generate_train_unroll(policy_params, train_key)
        # 2. Run Trajectories for Test Split (Out-of-Distribution)
        test_state = self._generate_test_unroll(policy_params, test_key)

        # Block to ensure execution finishes for timing
        test_state.info['eval_metrics'].active_episodes.block_until_ready()
        epoch_eval_time = time.time() - t
        
        metrics = {}

        # Process metrics for both splits
        for split_name, state in [("train", train_state), ("test", test_state)]:
            eval_metrics = state.info['eval_metrics']
            
            for name, value in eval_metrics.episode_metrics.items():
                metrics[f'eval/{split_name}_episode_{name}'] = np.mean(value)

                if name in ["success", "easy_success", "hard_success"]:
                    metrics[f'eval/{split_name}_episode_{name}_rate'] = np.mean(value > 0)

            metrics[f'eval/{split_name}_avg_episode_length'] = np.mean(eval_metrics.episode_steps)

        # General evaluation metrics
        metrics['eval/epoch_eval_time'] = epoch_eval_time
        metrics['eval/sps'] = (self._steps_per_unroll * 2) / epoch_eval_time
        self._eval_walltime += epoch_eval_time
        metrics['eval/walltime'] = self._eval_walltime
        
        return {**training_metrics, **metrics}    
        
def get_video(inference_policy, video_env, video_key, episode_length):


    video_env_states = get_trajectory(inference_policy, video_env, video_key, episode_length)
    
    if 'goal_mask' in video_env_states.info:
        video_env.model.geom_rgba[ video_env._mocap_targets_geom, -1 ] = 0.2
        video_env.model.geom_rgba[ video_env._mocap_targets_geom[ ~video_env_states.info['goal_mask'][0][0] ], -1] = 0

    mocap_key = 'target_mocap'
    video_images = []
    for i in range(episode_length):
        if i % 2 == 0:
            video_images.append(video_env.render_from_info(
                video_env_states.data.qpos[i][0], 
                video_env_states.data.qvel[i][0], 
                video_env_states.info[f'{mocap_key}_pos'][i][0],
                video_env_states.info[f'{mocap_key}_quat'][i][0],
            ))
    return video_images