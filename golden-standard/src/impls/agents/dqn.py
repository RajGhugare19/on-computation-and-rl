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


class GCDQNAgent(flax.struct.PyTreeNode):
    """Goal-conditioned DQN (discrete actions only).

    Minimal changes from the GCIQL implementation: re-uses critic networks (two-head ensemble),
    trains critic with TD targets computed from the target critic and uses greedy action selection.
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def critic_loss(self, batch, grad_params):
        """Compute the DQN critic loss (discrete actions).

        Assumes:
         - batch['actions'] contains integer action indices.
         - critic when passed actions returns Q(s,a) scalar(s); when called without actions returns
           Q(s, a) vectors (this mirrors the original critic init/signature).
        """
        # Current Q for taken actions (may be scalars if critic uses actions input)
        q1_a, q2_a = self.network.select('critic')(
            batch['observations'], batch['value_goals'], batch['actions'], params=grad_params
        )
        # Ensure shapes are (batch,)
        q1_a = jnp.squeeze(q1_a, axis=-1) if q1_a.ndim > 1 else q1_a
        q2_a = jnp.squeeze(q2_a, axis=-1) if q2_a.ndim > 1 else q2_a

        # Target Q: use target critic to get Q-vector for next states, average ensemble, take max over actions
        all_actions = jnp.tile(jnp.arange(6), (batch['next_observations'].shape[0], 1))  # B x 6
        qs = jax.lax.stop_gradient(jax.vmap(self.network.select('target_critic'), in_axes=(None, None, 1))(batch['next_observations'], batch['value_goals'], all_actions)) # 6 x 2 x B
        qs = qs.mean(axis=1) # 6 x B
        qs = qs.transpose(1, 0) # B x 6
        # q1_next, q2_next expected shape: (batch, action_dim)
        max_next_q = jnp.max(qs, axis=-1)

        # TD or MC target
        if self.config['use_discounted_mc_rewards']:
            target = batch['rewards'] 
        else:
            target = batch['rewards'] + self.config['discount'] * batch['masks'] * max_next_q

        # MSE loss on both heads (keeps two-head training similar to your critic ensemble)
        critic_loss = ((q1_a - target) ** 2 + (q2_a - target) ** 2).mean()

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
        alpha_temp_loss = ((entropy + self.config['target_entropy'])**2).mean()  # Target entropy is a negative constant like -log(6)

        total_loss = critic_loss +  alpha_temp_loss

        return total_loss, {
            'critic_loss': critic_loss,
            'q_mean': target.mean(),
            'q_max': target.max(),
            'q_min': target.min(),
            'q.std': target.std(),
            'entropy': entropy.mean(),
            'alpha_temp': alpha_temp,
            'entropy_std': dist.entropy().std(),
            'alpha_temp_loss': alpha_temp_loss,
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss (only critic loss for DQN)."""
        info = {}
        # Only critic is trained for DQN (no value / actor losses)
        critic_loss, critic_info = self.critic_loss(batch, grad_params)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        loss = critic_loss
        return loss, info

    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @jax.jit
    def update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
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
        """Create a new DQN agent (discrete only)."""
        if not config['discrete']:
            raise ValueError("GCDQNAgent currently supports only discrete action spaces. Set config['discrete']=True.")

        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_goals = ex_observations
        action_dim = int(ex_actions.max() + 1)

        # Define encoders.
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['critic'] = GCEncoder(concat_encoder=encoder_module())

        # For DQN we only need a discrete critic (we keep other modules for compatibility/minimal changes).
        critic_def = GCDiscreteCritic(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            ensemble=True,
            gc_encoder=encoders.get('critic'),
            action_dim=action_dim,
            net_arch=config['net_arch'],
        )

        # Keep dummy value/actor defs to minimize code changes (they won't be used in training).
        value_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            ensemble=False,
            gc_encoder=None,
            net_arch=config['net_arch'],
        )
        actor_def = GCDiscreteActor(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            gc_encoder=None,
            net_arch=config['net_arch'],
        )
        
        if config['target_entropy'] is None:
            config['target_entropy'] = -config['target_entropy_multiplier'] * action_dim/2
        alpha_temp_def = LogParam()

        network_info = dict(
            value=(value_def, (ex_observations, ex_goals)),
            critic=(critic_def, (ex_observations, ex_goals, ex_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, ex_goals, ex_actions)),
            actor=(actor_def, (ex_observations, ex_goals)),
            alpha_temp=(alpha_temp_def, ()),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network_params
        params['modules_target_critic'] = params['modules_critic']

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            # Agent hyperparameters.
            agent_name='gcdqn',
            lr=3e-4,
            batch_size=1024,
            actor_hidden_dims=(512, 512, 512),
            value_hidden_dims=(512, 512, 512),
            layer_norm=True,
            discount=0.99,
            tau=0.005,
            # legacy / unused fields from IQL left for compatibility:
            expectile=0.9,
            actor_loss='ddpgbc',
            alpha=0.3,
            const_std=True,
            discrete=True,  # DQN requires discrete actions
            encoder=ml_collections.config_dict.placeholder(str),
            # Dataset hyperparameters.
            dataset_class='GCDataset',
            value_p_curgoal=0.2,
            value_p_trajgoal=0.5,
            value_p_randomgoal=0.3,
            value_geom_sample=True,
            actor_p_curgoal=0.0,
            actor_p_trajgoal=1.0,
            actor_p_randomgoal=0.0,
            actor_geom_sample=False,
            gc_negative=True,
            p_aug=0.0,
            frame_stack=ml_collections.config_dict.placeholder(int),
        )
    )
    return config
