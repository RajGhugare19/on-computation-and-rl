import copy
from typing import Any

import distrax
import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from impls.utils.encoders import GCEncoder, encoder_modules
from impls.utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from impls.utils.networks import GCActor, GCDiscreteActor, GCDiscreteCritic, GCValue, LogParam


class ClearnSearchAgent(flax.struct.PyTreeNode):
    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def critic_loss(self, batch, grad_params, rng=None):
        gamma = self.config['discount']

        rolled_goals = jnp.roll(batch['next_observations'], shift=1, axis=0)
        random_values = self.network.select('critic')(
            batch['observations'], rolled_goals, batch['actions'], params=grad_params
        )
        obj_neg = jax.nn.log_sigmoid(-random_values)
        
        if self.config['is_td']:
            next_values = self.network.select('critic')(
               batch['observations'], batch['next_observations'], batch['actions'], params=grad_params
            )
            obj_pos = jax.nn.log_sigmoid(next_values)

            next_actions = self.sample_actions(batch['next_observations'], rolled_goals, rng)
            w = self.network.select('target_critic')(
                batch['next_observations'], rolled_goals, next_actions,
            )
            w = jax.lax.stop_gradient(w)
            w = jnp.clip(w, min=-10, max=20)
            obj_c = jnp.min(jnp.exp(w), axis=0, keepdims=True) / jnp.mean(jnp.exp(w)) * jax.nn.log_sigmoid(random_values)

            critic_loss = -jnp.mean(
                (1 - gamma) * obj_pos
                + obj_neg
                + gamma * obj_c
            )
        else:
            future_values = self.network.select('critic')(
                batch['observations'], batch['value_goals'], batch['actions'], params=grad_params
            )
            obj_pos = jax.nn.log_sigmoid(future_values)
            critic_loss = -jnp.mean(obj_pos + obj_neg)

        q = random_values

        # Update target entropy
        all_actions = jnp.tile(jnp.arange(6), (batch['observations'].shape[0], 1))  # B x 6
        qs = jax.lax.stop_gradient(
            jax.vmap(self.network.select("critic"), in_axes=(None, None, 1))(batch['observations'], jnp.roll(batch['next_observations'], shift=1, axis=0), all_actions)
        )  # 6 x 2 x B
        if len(qs.shape) == 2:  # Non-ensemble.
            qs = qs[:, None, ...]
        qs = qs.mean(axis=1)  # 6 x B
        qs = qs.transpose(1, 0) # B x 6

        alpha_temp = self.network.select('alpha_temp')(params=grad_params)
        dist = distrax.Categorical(logits=qs / jnp.maximum(1e-6, alpha_temp))
        entropy = dist.entropy()
        alpha_temp_loss = ((entropy + self.config['target_entropy'])**2).mean()  

        total_loss = critic_loss +  alpha_temp_loss
        return total_loss, {
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
            'q.std': q.std(),
            'entropy': entropy.mean(),
            'alpha_temp': alpha_temp,
            'entropy_std': dist.entropy().std(),
            'alpha_temp_loss': alpha_temp_loss,
        }


    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        info = {}
        rng = rng if rng is not None else self.rng

        rng, critic_rng = jax.random.split(rng)
        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        loss = critic_loss 
        return loss, info

    def target_update(self, network, module_name):
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'value')
        self.target_update(new_network, 'critic')

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(
        self,
        observations,
        goals=None,
        seed=None,
        temperature=1.0,
    ):
        """
        Returns integer action indices. Continuous actions are not supported here.
        """
        if not self.config['discrete']:
            raise NotImplementedError("ClearnSearchAgent.sample_actions supports only discrete action spaces.")
        
        all_actions = jnp.tile(jnp.arange(6), (observations.shape[0], 1))  # B x 6
        qs = jax.lax.stop_gradient(jax.vmap(self.network.select('critic'), in_axes=(None, None, 1))(observations, goals, all_actions)) # 6 x 2 x B
        qs = qs.mean(axis=1) # 6 x B
        qs = qs.transpose(1, 0) # B x 6
        if self.config['action_sampling'] == 'softmax':
            # Use critic to get Q-values (use first/ensemble as appropriate). Prefer the minimum head for conservative action,
            # or average — here we average the two heads and pick argmax.

            # Softmax actions
            alpha_temp = jax.lax.stop_gradient(self.network.select('alpha_temp')())
            dist = distrax.Categorical(logits=qs / jnp.maximum(1e-6, alpha_temp))
            actions = dist.sample(seed=seed)
        elif self.config['action_sampling'] == 'epsilon_greedy':
            greedy_actions = jnp.argmax(qs, axis=-1)  # B
            # random actions
            rng, rng_uniform = jax.random.split(seed)
            random_actions = jax.random.randint(rng, greedy_actions.shape, 0, 6)

            # ε-greedy: pick random with prob ε, else greedy
            probs = jax.random.uniform(rng_uniform, greedy_actions.shape)
            actions = jnp.where(probs < 0.1, random_actions, greedy_actions)
        else:
            raise ValueError(f"Unknown action sampling type {self.config['action_sampling']}")

        return actions

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):
        """Create a new agent.

        Args:
            seed: Random seed.
            ex_observations: Example batch of observations.
            ex_actions: Example batch of actions. In discrete-action MDPs, this should contain the maximum action value.
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_goals = ex_observations
        if config['discrete']:
            action_dim = ex_actions.max() + 1
        else:
            action_dim = ex_actions.shape[-1]

        # Define encoders.
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['value'] = GCEncoder(concat_encoder=encoder_module())
            encoders['actor'] = GCEncoder(concat_encoder=encoder_module())
            encoders['critic'] = GCEncoder(concat_encoder=encoder_module())

        # Define value and actor networks.
        # GCIQL: Use both V and Q.
        value_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            ensemble=False,
            gc_encoder=encoders.get('value'),
        )
        if config['discrete']:
            critic_def = GCDiscreteCritic(
                hidden_dims=config['value_hidden_dims'],
                layer_norm=config['layer_norm'],
                ensemble=True,
                gc_encoder=encoders.get('critic'),
                action_dim=action_dim,
                net_arch=config['net_arch'],
            )
        else:
            critic_def = GCValue(
                hidden_dims=config['value_hidden_dims'],
                layer_norm=config['layer_norm'],
                ensemble=True,
                gc_encoder=encoders.get('critic'),
                net_arch=config['net_arch'],
            )
        if config['discrete']:
            actor_def = GCDiscreteActor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=action_dim,
                gc_encoder=encoders.get('actor'),
                net_arch=config['net_arch'],
            )
        else:
            actor_def = GCActor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=action_dim,
                state_dependent_std=False,
                const_std=config['const_std'],
                gc_encoder=encoders.get('actor'),
                net_arch=config['net_arch'],
            )

        if config['target_entropy'] is None:
            config['target_entropy'] = -config['target_entropy_multiplier'] * action_dim/2
        alpha_temp_def = LogParam()

        network_info = dict(
            value=(value_def, (ex_observations, ex_goals)),
            target_value=(copy.deepcopy(value_def), (ex_observations, ex_goals)),
            actor=(actor_def, (ex_observations, ex_goals)),
        )
        network_info.update(
            critic=(critic_def, (ex_observations, ex_goals, ex_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, ex_goals, ex_actions)),
            alpha_temp=(alpha_temp_def, ()),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network_params
        params['modules_target_value'] = params['modules_value']
        params['modules_target_critic'] = params['modules_critic']

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            # Agent hyperparameters.
            agent_name='clearn',  # Agent name.
            lr=3e-4,  # Learning rate.
            batch_size=1024,  # Batch size.
            actor_hidden_dims=(512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512),  # Value network hidden dimensions.
            layer_norm=True,  # Whether to use layer normalization.
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network update rate.
            expectile=0.5,  # IQL expectile.
            actor_loss='ddpgbc',  # Actor loss type ('awr' or 'ddpgbc').
            alpha=10.0,  # Temperature in AWR or BC coefficient in DDPG+BC.
            const_std=True,  # Whether to use constant standard deviation for the actor.
            discrete=False,  # Whether the action space is discrete.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
            # Dataset hyperparameters.
            dataset_class='GCDataset',  # Dataset class name.
            value_p_curgoal=0.2,  # Probability of using the current state as the value goal.
            value_p_trajgoal=0.5,  # Probability of using a future state in the same trajectory as the value goal.
            value_p_randomgoal=0.3,  # Probability of using a random state as the value goal.
            value_geom_sample=True,  # Whether to use geometric sampling for future value goals.
            actor_p_curgoal=0.0,  # Probability of using the current state as the actor goal.
            actor_p_trajgoal=1.0,  # Probability of using a future state in the same trajectory as the actor goal.
            actor_p_randomgoal=0.0,  # Probability of using a random state as the actor goal.
            actor_geom_sample=False,  # Whether to use geometric sampling for future actor goals.
            gc_negative=True,  # Whether to use '0 if s == g else -1' (True) or '1 if s == g else 0' (False) as reward.
            p_aug=0.0,  # Probability of applying image augmentation.
            frame_stack=ml_collections.config_dict.placeholder(int),  # Number of frames to stack.
            use_q=True,  # Whether to use Q function 
        )
    )
    return config