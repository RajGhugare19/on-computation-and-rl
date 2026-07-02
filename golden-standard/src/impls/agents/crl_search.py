from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax
import distrax
import flax.linen as nn
from impls.utils.encoders import GCEncoder, encoder_modules
from impls.utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from impls.utils.networks import GCActor, GCBilinearValue, GCDiscreteActor, GCDiscreteBilinearCritic, LogParam



class CRLSearchAgent(flax.struct.PyTreeNode):
    """Contrastive RL (CRL) agent.

    This implementation supports both AWR (actor_loss='awr') and DDPG+BC (actor_loss='ddpgbc') for the actor loss.
    CRL with DDPG+BC only fits a Q function, while CRL with AWR fits both Q and V functions to compute advantages.
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def contrastive_loss(self, batch, grad_params, module_name='critic'):
        """Compute the contrastive value loss for the Q or V function."""
        batch_size = batch['observations'].shape[0]
        actions = batch['actions']

        q, phi, psi = self.network.select(module_name)(
            batch['observations'],
            batch['value_goals'],
            actions=actions,
            info=True,
            params=grad_params,
        )
        if len(phi.shape) == 2:  # Non-ensemble.
            phi = phi[None, ...]
            psi = psi[None, ...]
        logits = jnp.einsum('eik,ejk->ije', phi, psi) / jnp.sqrt(phi.shape[-1])
        # logits.shape is (B, B, e) with one term for positive pair and (B - 1) terms for negative pairs in each row.
        I = jnp.eye(batch_size)
        contrastive_loss = jax.vmap(
            lambda _logits: optax.sigmoid_binary_cross_entropy(logits=_logits, labels=I),
            in_axes=-1,
            out_axes=-1,
        )(logits)
        contrastive_loss = jnp.mean(contrastive_loss)

        # Compute additional statistics.
        logits = jnp.mean(logits, axis=-1) # (B, B)
        correct = jnp.argmax(logits, axis=1) == jnp.argmax(I, axis=1)
        logits_pos = jnp.sum(logits * I) / jnp.sum(I)
        logits_neg = jnp.sum(logits * (1 - I)) / jnp.sum(1 - I)

        # Update target entropy
        def value_transform(x):
            return jnp.log(jnp.maximum(x, 1e-6))
        
        all_actions = jnp.tile(jnp.arange(6), (batch['observations'].shape[0], 1))  # B x 6
        qs = jax.lax.stop_gradient(
            jax.vmap(self.network.select("critic"), in_axes=(None, None, 1))(batch['observations'], jnp.roll(batch['next_observations'], shift=1, axis=0), all_actions)
        )  # 6 x 2 x B
        if len(qs.shape) == 2:  # Non-ensemble.
            qs = qs[:, None, ...]
        qs = qs.mean(axis=1)  # 6 x B
        qs = qs.transpose(1, 0) # B x 6
        qs = value_transform(qs)

        alpha_temp = self.network.select('alpha_temp')(params=grad_params)
        dist = distrax.Categorical(logits=qs / jnp.maximum(1e-6, alpha_temp))
        entropy = dist.entropy()
        alpha_temp_loss = ((entropy + self.config['target_entropy'])**2).mean()  

        total_loss = contrastive_loss + alpha_temp_loss
        return total_loss, {
            'contrastive_loss': contrastive_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
            'binary_accuracy': jnp.mean((logits > 0) == I),
            'categorical_accuracy': jnp.mean(correct),
            'logits_pos': logits_pos,
            'logits_neg': logits_neg,
            'logits': logits.mean(),
            'entropy': entropy.mean(),
            'alpha_temp': alpha_temp,
            'entropy_std': dist.entropy().std(),
            'alpha_temp_loss': alpha_temp_loss,
        }


    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        critic_loss, critic_info = self.contrastive_loss(batch, grad_params, 'critic')
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        loss = critic_loss 
        return loss, info

    @jax.jit
    def update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)

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
        
        def value_transform(x):
            return jnp.log(jnp.maximum(x, 1e-6))
        

        all_actions = jnp.tile(jnp.arange(6), (observations.shape[0], 1))  # B x 6
        qs = jax.lax.stop_gradient(jax.vmap(self.network.select('critic'), in_axes=(None, None, 1))(observations, goals, all_actions)) # 6 x 2 x B
        qs = qs.mean(axis=1) # 6 x B
        qs = value_transform(qs)
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
        ex_goals=None,
    ):
        """Create a new agent.

        Args:
            seed: Random seed.
            ex_observations: Example batch of observations.
            ex_actions: Example batch of actions. In discrete-action MDPs, this should contain the maximum action value.
            config: Configuration dictionary.
            ex_goals: Example batch of goals. If None, the goals are set to the observations.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        if ex_goals is None:
            ex_goals = ex_observations

        if config['discrete']:
            action_dim = ex_actions.max() + 1
        else:
            action_dim = ex_actions.shape[-1]

        # Define encoders.
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['critic_state'] = encoder_module()
            encoders['critic_goal'] = encoder_module()
            encoders['actor'] = GCEncoder(concat_encoder=encoder_module())
            if config['actor_loss'] == 'awr':
                encoders['value_state'] = encoder_module()
                encoders['value_goal'] = encoder_module()

        # Define value and actor networks.
        if config['discrete']:
            critic_def = GCDiscreteBilinearCritic(
                hidden_dims=config['value_hidden_dims'],
                latent_dim=config['latent_dim'],
                layer_norm=config['layer_norm'],
                ensemble=True,
                value_exp=True,
                state_encoder=encoders.get('critic_state'),
                goal_encoder=encoders.get('critic_goal'),
                action_dim=action_dim,
                net_arch=config['net_arch'],
            )
        else:
            critic_def = GCBilinearValue(
                hidden_dims=config['value_hidden_dims'],
                latent_dim=config['latent_dim'],
                layer_norm=config['layer_norm'],
                ensemble=True,
                value_exp=True,
                state_encoder=encoders.get('critic_state'),
                goal_encoder=encoders.get('critic_goal'),
                net_arch=config['net_arch'],
            )

        if config['actor_loss'] == 'awr':
            # AWR requires a separate V network to compute advantages (Q - V).
            value_def = GCBilinearValue(
                hidden_dims=config['value_hidden_dims'],
                latent_dim=config['latent_dim'],
                layer_norm=config['layer_norm'],
                ensemble=False,
                value_exp=True,
                state_encoder=encoders.get('value_state'),
                goal_encoder=encoders.get('value_goal'),
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
            critic=(critic_def, (ex_observations, ex_goals, ex_actions)),
            actor=(actor_def, (ex_observations, ex_goals)),
            alpha_temp=(alpha_temp_def, ()),
        )
        if config['actor_loss'] == 'awr':
            network_info.update(
                value=(value_def, (ex_observations, ex_goals)),
            )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.FrozenConfigDict(
        dict(
            # Agent hyperparameters.
            agent_name='crl_search',  # Agent name.
            lr=3e-4,  # Learning rate.
            batch_size=256,  # Batch size.
            actor_hidden_dims=(256, 256),  # Actor network hidden dimensions.
            value_hidden_dims=(256, 256),  # Value network hidden dimensions.
            latent_dim=64, 
            layer_norm=True,  # Whether to use layer normalization.
            discount=0.99,  # Discount factor.
            actor_loss='awr',  # Actor loss type ('awr' or 'ddpgbc').
            alpha=0.1,  # Temperature in AWR or BC coefficient in DDPG+BC.
            actor_log_q=True,  # Whether to maximize log Q (True) or Q itself (False) in the actor loss.
            const_std=True,  # Whether to use constant standard deviation for the actor.
            discrete=True,  # Whether the action space is discrete.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
            # Dataset hyperparameters.
            dataset_class='GCDataset',  # Dataset class name.
            value_p_curgoal=0.0,  # Probability of using the current state as the value goal.
            value_p_trajgoal=1.0,  # Probability of using a future state in the same trajectory as the value goal.
            value_p_randomgoal=0.0,  # Probability of using a random state as the value goal.
            value_geom_sample=True,  # Whether to use geometric sampling for future value goals.
            actor_p_curgoal=0.0,  # Probability of using the current state as the actor goal.
            actor_p_trajgoal=1.0,  # Probability of using a future state in the same trajectory as the actor goal.
            actor_p_randomgoal=0.0,  # Probability of using a random state as the actor goal.
            actor_geom_sample=False,  # Whether to use geometric sampling for future actor goals.
            gc_negative=False,  # Unused (defined for compatibility with GCDataset).
            p_aug=0.0,  # Probability of applying image augmentation.
            frame_stack=ml_collections.config_dict.placeholder(int),  # Number of frames to stack.
            target_entropy_multiplier=0.5,  # Multiplier for the target entropy (used in SAC-like agents), when target_entropy not set
            target_entropy=-jnp.log(6),  # Target entropy (None for automatic tuning).
        )
    )
    return config
