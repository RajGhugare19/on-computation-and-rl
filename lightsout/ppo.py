import os

xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags
os.environ["MUJOCO_GL"] = "egl"

import time
import tyro
import numpy as np
import functools
import pprint
import wandb
import wandb_osh

import jax
import flax
import optax
import distrax
import flax.linen as nn
import jax.numpy as jnp

from pathlib import Path
from flax.training.train_state import TrainState
from dataclasses import dataclass, field
from typing import Any, Sequence, NamedTuple, Optional
from wandb_osh.hooks import TriggerWandbSyncHook

import utils.running_statistics as running_statistics
from utils.wrapper import wrap_env
from utils.evaluation import Evaluator
from models.networks import save_params
from utils.jax import count_parameters
from envs.utils import make_env

@dataclass
class Args:
    # experiment
    agent: str = "ppo-lstmthinker-discrete-action"
    seed: int = 1
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    
    # logging and checkpointing
    track: bool = False
    wandb_project_name: str = "rl_cot"
    wandb_entity: str = 'tmp'
    wandb_mode: str = 'online'
    wandb_dir: str = './'
    wandb_group: str = 'default'
    wandb_name_tag: str = ''

    num_eval_steps: int = 50             # number of evaluation / logging / saving steps
    num_reset_steps: int = 50             # number of times to call true resets (env.reset) instead of soft resets (AutoResetWrapper)

    save_checkpoint: bool = True

    # environment
    env_id: str = 'lightsout-3x3'
    num_envs: int = 2048
    num_eval_envs: int = 128

    # maze
    train_test_divide: str = 'quantile'
    top_quantile: float = 0.8

    # lightsout
    difficulty_threshold: float = 0.5

    # algorithm
    num_timesteps: int = 50000000
    rollout_length: int = 160
    num_minibatches_per_rollout: int = 32
    num_epochs_per_rollout: int = 8    
    learning_rate: float = 1e-4
    discount: float = 0.99
    entropy_cost: float = 0.01
    reward_scaling: float = 1.0
    gae_lambda: float = 0.95
    clipping_epsilon: float = 0.3
    normalize_advantage: bool = True

    architecture_type: str = 'int_thinker'

    #interp layer
    num_interp_layers: int = 1

    # thinking lstm
    hidden_state_dim: int = 64
    num_lstm_layers: int = 2
    thinking_steps: int = 5

    # resnet
    resnet_state_dim: int = 128
    num_resnet_layers: int = 10

    # standard MLP
    policy_hidden_sizes: list = field(default_factory=lambda: [256, 256, 256, 256])
    value_hidden_sizes: list = field(default_factory=lambda: [256, 256, 256, 256])

@flax.struct.dataclass
class PPOTrainingState(TrainState):
  """Contains training state for the learner."""
  normalizer_params: Any
  env_steps: float

class Transition(NamedTuple):
    """Container for a transition."""
    observation: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    discount: jnp.ndarray
    next_observation: jnp.ndarray
    extras: jnp.ndarray = ()

@flax.struct.dataclass
class PPONetworks:
    actor_critic_network: Any

def make_inference_fn(ppo_networks):
    """Creates params and inference function for the PPO agent."""
    def make_policy(params, deterministic: bool = False):
        policy_network = ppo_networks.actor_critic_network

        def policy(observations, goals, key_sample):
            inputs = jnp.concatenate([observations, goals], axis=-1)
            policy_dist, value = policy_network.apply(
                params['policy'], 
                inputs, 
                params['normalizer']
            )
                
            if deterministic:
                return policy_dist.mode(), {'value': value}
            
            actions = policy_dist.sample(seed=key_sample)
            log_prob = policy_dist.log_prob(actions)
            return actions, {
                'log_prob': log_prob,
                'value': value
            }

        return policy
    return make_policy

def main(args: Args):
    
    args.num_training_step = args.num_timesteps // ( args.num_envs * args.rollout_length )
    args.num_training_steps_per_eval = args.num_training_step // args.num_eval_steps
    args.num_training_steps_per_real_reset = args.num_training_step // max(1, args.num_reset_steps)
    args.minibatch_size = args.num_envs * args.rollout_length // ( args.num_minibatches_per_rollout )
        
    print(f"Total number of training steps = {args.num_training_step}")
    print(f"Total number of gradient steps per training step = {args.num_minibatches_per_rollout * args.num_epochs_per_rollout}")
    print(f"Total number of env steps per training step = {args.num_envs * args.rollout_length}")
    print(f"Data to update ratio = {  ( args.num_envs * args.rollout_length ) / (args.num_minibatches_per_rollout * args.num_epochs_per_rollout)}")    

    args.exp_name = f"{args.wandb_name_tag + '__' if args.wandb_name_tag != '' else ''}{args.env_id}__{args.seed}__{os.path.basename(__file__)[: -len('.py')]}__{int(time.time())}"
    
    # Initialize wandb if tracking is enabled
    if args.track:
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            mode=args.wandb_mode,
            dir=args.wandb_dir,
            group=args.wandb_group,
            name=args.exp_name,
            config=vars(args),
            save_code=True,
        )

        if args.wandb_mode == 'offline':
            wandb_osh.set_log_level("ERROR")
            trigger_sync = TriggerWandbSyncHook()
    
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    key, key_env, key_eval, key_policy, key_value = jax.random.split(key, 5)

    # Initialize environment
    env_class, default_config = make_env(args)
    env = wrap_env( env_class(config=default_config), default_config.episode_length )
    eval_env = wrap_env( env_class(config=default_config), default_config.episode_length, train=False )  
    episode_length = default_config.episode_length

    # Initialize checkpoint folder
    if args.save_checkpoint:
        save_path = Path(args.wandb_dir) / f"checkpoints/{args.exp_name}/"
        os.makedirs(save_path, exist_ok=True)

    reset_fn = jax.jit(env.reset)
    key_envs = jax.random.split(key_env, args.num_envs)
    env_state = reset_fn(key_envs)
    obs_size = env.observation_size
    action_size = env.action_size
    goal_size = env.goal_size

    log_data_metric_keys = []
    for k in ("obj_reached_once", "obj_lifted", "obj_moved"):
        if k in env_state.metrics.keys():
            log_data_metric_keys.append(k)
    log_data_metric_keys = tuple(log_data_metric_keys)

    # Initialize PPO networks
    if args.architecture_type == 'lstm_thinker':
        from models.ppo import LSTMThinkerActorValue 
        ppo_network = PPONetworks( 
            actor_critic_network = LSTMThinkerActorValue(hidden_state_dim=args.hidden_state_dim, num_layers=args.num_lstm_layers, thinking_steps=args.thinking_steps, output_dim_1=action_size),
        )
    elif args.architecture_type == 'int_thinker':
        from models.ppo import InterpolationThinkerActorValue 
        ppo_network = PPONetworks(
            actor_critic_network = InterpolationThinkerActorValue(hidden_state_dim=args.hidden_state_dim, num_layers=args.num_interp_layers, thinking_steps=args.thinking_steps, output_dim_1=action_size),
        )
    elif args.architecture_type == 'resnet':
        from models.ppo import ResnetActorValue
        ppo_network = PPONetworks( 
            actor_critic_network = ResnetActorValue(hidden_state_dim=args.resnet_state_dim, num_layers=args.num_resnet_layers, output_dim_1=action_size),
        )
    elif args.architecture_type == 'mlp':
        from models.ppo import MLPActorValue
        ppo_network = PPONetworks( 
            actor_critic_network = MLPActorValue(layer_sizes=args.policy_hidden_sizes + [action_size]),
        )
    else:
        raise NotImplementedError(f"Architecture type {args.architecture_type} not implemented.")

    training_state = PPOTrainingState.create(
        apply_fn=None,
        params=ppo_network.actor_critic_network.init( key_policy, x=jnp.zeros((1, obs_size+goal_size)) ),
        tx=optax.adam(learning_rate=args.learning_rate),  
        normalizer_params=running_statistics.init_state((obs_size+goal_size,) ),
        env_steps=np.zeros((), dtype=np.float64),
    )
    make_policy = make_inference_fn(ppo_network)
    
    print(f'\nNumber of parameters in actor critic network are: {count_parameters(training_state.params)}\n')

    # Initialize evaluators
    evaluator = Evaluator(
        eval_env,
        functools.partial(make_policy, deterministic=True),
        num_eval_envs=args.num_eval_envs,
        episode_length=episode_length,
        key=key_eval,
    )

    def generate_unroll(
        env,
        env_state,
        policy,
        key,
        unroll_length,
        extra_fields,
    ):
        """Collect trajectories of given unroll_length."""        
        @jax.jit
        def f(carry, unused_t):
            env_state, key = carry
            key, next_key = jax.random.split(key)
            actions, policy_extras = policy(env_state.obs, env_state.info['target_goal'], key)  
            
            next_env_state = env.step(env_state, actions)
            state_extras = {x: next_env_state.info[x] for x in extra_fields}

            metrics = {x: next_env_state.metrics[x] for x in log_data_metric_keys}

            transition = Transition(
                observation=jnp.concatenate( [env_state.obs, env_state.info['target_goal']], axis=-1),
                action=actions,
                value=policy_extras['value'],
                reward=next_env_state.reward,
                discount=1 - next_env_state.done,
                next_observation=jnp.concatenate( [next_env_state.obs, next_env_state.info['target_goal']], axis=-1),
                extras={'policy_extras': policy_extras, 'state_extras': state_extras},
            )
            
            return (next_env_state, next_key), (transition, metrics)

        (final_env_state, _), (data, data_metrics) = jax.lax.scan(
            f, (env_state, key), (), length=unroll_length
        )
        return final_env_state, data, data_metrics

    @jax.jit
    def data_collect_step(training_state, env_state, key_generate_rollout):
        policy = make_policy({
            'policy': training_state.params, 
            'normalizer': training_state.normalizer_params,
            })
        
        env_state, data, data_metrics = generate_unroll(
            env,
            env_state,
            policy,
            key_generate_rollout,
            args.rollout_length,
            extra_fields=('truncation',),
        )

        # Update normalization params.
        normalizer_params = running_statistics.update(
            training_state.normalizer_params,
            data.observation,
        )

        training_state = training_state.replace(
            normalizer_params=normalizer_params,
            env_steps=training_state.env_steps + args.rollout_length * args.num_envs,
        )

        return training_state, env_state, data, data_metrics

    def compute_gae(
        truncation: jnp.ndarray,
        termination: jnp.ndarray,
        rewards: jnp.ndarray,
        values: jnp.ndarray,
        bootstrap_value: jnp.ndarray,
        lambda_: float = 1.0,
        discount: float = 0.99,
    ):
        truncation_mask = 1 - truncation
        # Append bootstrapped value to get [v1, ..., v_t+1]
        values_t_plus_1 = jnp.concatenate(
            [values[1:], jnp.expand_dims(bootstrap_value, 0)], axis=0
        )
        deltas = rewards + discount * (1 - termination) * values_t_plus_1 - values
        deltas *= truncation_mask

        acc = jnp.zeros_like(bootstrap_value)
        vs_minus_v_xs = []

        def compute_vs_minus_v_xs(carry, target_t):
            lambda_, acc = carry
            truncation_mask, delta, termination = target_t
            acc = delta + discount * (1 - termination) * truncation_mask * lambda_ * acc
            return (lambda_, acc), (acc)

        (_, _), (vs_minus_v_xs) = jax.lax.scan(
            compute_vs_minus_v_xs,
            (lambda_, acc),
            (truncation_mask, deltas, termination),
            length=int(truncation_mask.shape[0]),
            reverse=True,
        )
        # Add V(x_s) to get v_s.
        vs = jnp.add(vs_minus_v_xs, values)

        vs_t_plus_1 = jnp.concatenate(
            [vs[1:], jnp.expand_dims(bootstrap_value, 0)], axis=0
        )
        advantages = (
            rewards + discount * (1 - termination) * vs_t_plus_1 - values
        ) * truncation_mask
        return jax.lax.stop_gradient(vs), jax.lax.stop_gradient(advantages)


    def compute_ppo_loss(
        params,
        normalizer_params,
        data,
        rng,
    ):
        policy_apply = ppo_network.actor_critic_network.apply

        data, value_targets, advantages = data
        
        policy_dist, baseline = policy_apply(params, data.observation, normalizer_params)

        # Policy function loss
        target_action_log_probs = policy_dist.log_prob( data.action )
        behaviour_action_log_probs = data.extras['policy_extras']['log_prob']
        rho_s = jnp.exp(target_action_log_probs - behaviour_action_log_probs)
        surrogate_loss1 = rho_s * advantages
        surrogate_loss2 = (jnp.clip(rho_s, 1 - args.clipping_epsilon, 1 + args.clipping_epsilon) * advantages)
        policy_loss = -jnp.mean(jnp.minimum(surrogate_loss1, surrogate_loss2))

        # Value function loss
        v_error = value_targets - baseline
        v_loss = jnp.mean(v_error * v_error) * 0.5 * 0.5

        # Entropy loss
        entropy = policy_dist.entropy()
        entropy = jnp.mean( entropy )        

        entropy_loss = args.entropy_cost * -entropy

        total_loss = policy_loss + v_loss + entropy_loss
        return total_loss, {
            'total_loss': total_loss,
            'policy_loss': policy_loss,
            'v_loss': v_loss,
            'entropy_loss': entropy_loss,
        }

    @jax.jit
    def learn_step(training_state, data, key_sgd):

        def _learn_step(carry, unused_t):
            
            def _train_minibatch_step(carry, data):
                training_state, key = carry
                key, key_loss = jax.random.split(key)
                
                (_, metrics), grads = jax.value_and_grad(compute_ppo_loss, has_aux=True)(training_state.params, training_state.normalizer_params, data, key_loss)
                training_state = training_state.apply_gradients(grads=grads)
                
                return (training_state, key), metrics

            training_state, data, value_targets, advantages, key = carry
            key, key_perm, key_grad = jax.random.split(key, 3)
        
            def shuffle_and_reshape(x: jnp.ndarray):
                x = jax.random.permutation(key_perm, x)
                x = jnp.reshape(x, (args.num_minibatches_per_rollout, -1) + x.shape[2:])
                return x

            batch_data = ( data, value_targets, advantages )
            shuffled_batch_data = jax.tree_util.tree_map(shuffle_and_reshape, batch_data)

            (training_state, _), metrics = jax.lax.scan(
                _train_minibatch_step,
                (training_state, key_grad),
                shuffled_batch_data,
                length=args.num_minibatches_per_rollout,
            )
            return (training_state, data, value_targets, advantages, key), metrics

        # calculate gae
        terminal_obs = jax.tree_util.tree_map(lambda x: x[-1], data.next_observation)
        _, bootstrap_value = ppo_network.actor_critic_network.apply(training_state.params, terminal_obs, training_state.normalizer_params)

        rewards = data.reward * args.reward_scaling
        truncation = data.extras['state_extras']['truncation']
        termination = (1 - data.discount) * (1 - truncation)

        value_targets, advantages = compute_gae(
            truncation=truncation,
            termination=termination,
            rewards=rewards,
            values=data.value,
            bootstrap_value=bootstrap_value,
            lambda_=args.gae_lambda,
            discount=args.discount,
        )
        if args.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
        (training_state, _, _, _, _), metrics = jax.lax.scan(
            _learn_step,
            (training_state, data, value_targets, advantages, key_sgd),
            (),
            length=args.num_epochs_per_rollout,
        )

        return training_state, metrics

    def compute_feature_diagnostics(features):
        # adapted from https://github.com/roger-creus/stable-deep-rl-at-scale/blob/main/src/models/agent.py
        """Computes different approximations of the rank of the feature matrices.

        Args:
            feature_matrices (torch.Tensor): A tensor of shape (B_matrices, N_obs, D_dims).

        (1) Effective rank.
        A continuous approximation of the rank of a matrix.
        Definition 2.1. in Roy & Vetterli, (2007) https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7098875
        Also used in Huh et al. (2023) https://arxiv.org/pdf/2103.10427.pdf

        (2) Approximate rank.
        Threshold at the dimensions explaining 99% of the variance in a PCA analysis.
        Section 2 in Yang et al. (2020) https://arxiv.org/pdf/1909.12255.pdf

        (3) srank.
        Another (incorrect?) version of (2).
        Section 3 in Kumar et al. https://arxiv.org/pdf/2010.14498.pdf

        (4) Feature rank.
        A threshold rank: normalize by dim size and discard dimensions with singular values below 0.01.
        Equations (4) and (5). Lyle et al. (2022) https://arxiv.org/pdf/2204.09560.pdf
        """

        feature_matrices = jnp.expand_dims(features, axis=0)

        cutoff = 0.01
        threshold = 1 - cutoff

        # svals shape: (1, K) where K = min(N_obs, D_dims)
        svals = jnp.linalg.svdvals(feature_matrices)

        # (1) Effective rank. Roy & Vetterli (2007)
        sval_sum = jnp.sum(svals, axis=1)  # Shape: (1,)
        sval_dist = svals / sval_sum[..., None] # Use [..., None] for unsqueeze
        # Replace 0 with 1 to avoid log(0) = -inf
        sval_dist_fixed = jnp.where(sval_dist == 0, jnp.ones_like(sval_dist), sval_dist)
        effective_ranks = jnp.exp(-jnp.sum(sval_dist_fixed * jnp.log(sval_dist_fixed), axis=1))

        # (2) Approximate rank. PCA variance. Yang et al. (2020)
        sval_squares = svals**2
        sval_squares_sum = jnp.sum(sval_squares, axis=1) # Shape: (1,)
        cumsum_squares = jnp.cumsum(sval_squares, axis=1)
        threshold_crossed = cumsum_squares >= (threshold * sval_squares_sum[..., None])
        # Use jnp.logical_not for '~' on boolean arrays
        approximate_ranks = jnp.logical_not(threshold_crossed).sum(axis=-1) + 1

        # (3) srank. Weird. Kumar et al. (2020)
        cumsum = jnp.cumsum(svals, axis=1)
        threshold_crossed_srank = cumsum >= threshold * sval_sum[..., None]
        sranks = jnp.logical_not(threshold_crossed_srank).sum(axis=-1) + 1

        # (4) Feature rank. Most basic. Lyle et al. (2022)
        # Get N_obs directly from the shape
        n_obs = jnp.array(feature_matrices.shape[1], dtype=svals.dtype) 
        svals_of_normalized = svals / jnp.sqrt(n_obs)
        over_cutoff = svals_of_normalized > cutoff
        feature_ranks = over_cutoff.sum(axis=-1)

        return {
            'effective_rank_vetterli': effective_ranks,
            'approximate_rank_pca': approximate_ranks,
            'srank_kumar': sranks,
            'feature_rank_lyle': feature_ranks,
        }

    training_walltime, data_collect_step_time, learn_step_time = 0, 0, 0
    xt = time.time()
    metrics = None
    for ts in range(1, args.num_training_step + 1):
        
        key_sgd, key_generate_unroll, key = jax.random.split(key, 3)

        data_collect_start = time.time()
        training_state, env_state, training_data, data_metrics = data_collect_step(training_state, env_state, key_generate_unroll)
        data_collect_step_time += time.time() - data_collect_start
        
        learn_step_start = time.time()
        training_state, training_metrics = learn_step(training_state, training_data, key_sgd)
        learn_step_time += time.time() - learn_step_start

        if metrics is None:
            metrics = data_metrics | training_metrics
        else:
            metrics = jax.tree_util.tree_map(
                lambda x, y: x + y, metrics, (data_metrics | training_metrics)
            )

        if args.num_reset_steps > 0 and ts % args.num_training_steps_per_real_reset == 0:
            key_env, key = jax.random.split(key, 2)
            key_envs = jax.random.split(key_env, args.num_envs)
            env_state = reset_fn(key_envs)

        if ts % args.num_training_steps_per_eval == 0:
            es = ts // args.num_training_steps_per_eval
            
            metrics = jax.tree_util.tree_map(
                lambda x: x / args.num_training_steps_per_eval, metrics
            )
            metrics = jax.tree_util.tree_map(jnp.mean, metrics)
            jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)
            
            training_step_time = time.time() - xt            
            training_walltime += training_step_time

            sps = (
                args.num_training_steps_per_eval
                * args.num_envs * args.rollout_length
            ) / training_step_time

            metrics = {
                'training/sps': sps,
                'training/walltime': training_walltime,
                'training/data_collection_time_fraction' : data_collect_step_time / training_step_time,
                'training/learning_time_fraction' : learn_step_time / training_step_time,
                'training/env_steps': training_state.env_steps,
                'normalizer/count' : training_state.normalizer_params.count,
                'normalizer/mean' : jnp.mean( training_state.normalizer_params.mean ),
                'normalizer/summer_variance' : jnp.mean( training_state.normalizer_params.summed_variance ),
                'normalizer/std' : jnp.mean( training_state.normalizer_params.std ),
                **{f'training/{name}': value for name, value in metrics.items()},
            }

            metrics = evaluator.run_evaluation(
                policy_params={'policy':training_state.params, 'normalizer':training_state.normalizer_params},
                training_metrics=metrics,
            )

            print(f'\nEvaluation step {es}:\n')
            pprint.pprint(metrics)
            if args.track:
                wandb.log(metrics, step=es)
                if args.wandb_mode == 'offline':
                    trigger_sync()
            metrics = None

            if args.save_checkpoint:
                save_params(
                    f"{save_path}/params_{es}.pkl", 
                    params = (
                        training_state.params,
                        training_state.normalizer_params,
                    )
                )

            xt, data_collect_step_time, learn_step_time = time.time(), 0, 0

    if args.track:
        wandb.finish()
            
if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)