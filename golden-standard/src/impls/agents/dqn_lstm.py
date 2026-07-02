import copy
from typing import Any, Sequence

import distrax
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax
from impls.utils.encoders import GCEncoder, encoder_modules
from impls.utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from impls.utils.networks import GCValue, GCDiscreteActor, default_init, ensemblize, LogParam


class LSTMThinkingCritic(nn.Module):
    """LSTM-based critic with thinking steps for goal-conditioned discrete actions.
    
    This module uses an LSTM to perform iterative "thinking" over the input,
    allowing the model to refine its Q-value estimates through multiple processing steps.
    Takes action as input (concatenated with observations/goals) and outputs single Q-value.
    
    Attributes:
        d_model: Hidden dimension size for LSTM and embeddings.
        thinking_steps: Number of thinking iterations to perform.
        ensemble: Whether to use an ensemble of critics.
        gc_encoder: Optional goal-conditioned encoder.
    """
    
    d_model: int = 256
    thinking_steps: int = 3
    ensemble: bool = True
    gc_encoder: nn.Module = None
    layer_norm: bool = True
    pre_layer_norm: bool = True
    num_layers: int = 1
    
    def setup(self):
        self.input_embed_layer = nn.Dense(self.d_model)
        self.pre_ln = nn.LayerNorm()
        self.start_token = self.param(
            'start_token', 
            nn.initializers.normal(stddev=0.02), 
            (1, 1, self.d_model)
        )
        self.lstm_layers = [
            nn.RNN(nn.LSTMCell(features=self.d_model)) 
            for _ in range(self.num_layers)
        ]
        # Layer norms between LSTM layers for stability
        self.layer_norms = [nn.LayerNorm() for _ in range(self.num_layers)]
        # Standard initialization for value head
        self.value_head = nn.Dense(
            1,
            kernel_init=nn.initializers.orthogonal(1.0),
            bias_init=nn.initializers.zeros,
        )
    
    def __call__(self, observations, goals=None, actions=None):
        """Forward pass with thinking steps.
        
        Args:
            observations: Observation tensor of shape (B, obs_dim)
            goals: Goal tensor of shape (B, goal_dim) (optional)
            actions: One-hot encoded actions of shape (B, action_dim)
            
        Returns:
            Q-values: Q(s,a) for the given action, shape (B,)
        """
        # Encode inputs
        if self.gc_encoder is not None:
            inputs = self.gc_encoder(observations, goals)
        else:
            inputs = [observations]
            if goals is not None:
                inputs.append(goals)
            inputs = jnp.concatenate(inputs, axis=-1)
        
        # Concatenate action (one-hot encoded)
        if actions is not None:
            inputs = jnp.concatenate([inputs, actions], axis=-1)
        
        # Embed input: (B, d_model) -> (B, 1, d_model)
        x_emb = self.input_embed_layer(inputs)
        if self.pre_layer_norm:
            x_emb = self.pre_ln(x_emb)
        x_emb = jnp.expand_dims(x_emb, axis=1)
        
        # Create sequence with start token for first step
        x_first = x_emb + self.start_token
        if self.thinking_steps > 1:
            x_rest = jnp.tile(x_emb, (1, self.thinking_steps - 1, 1))
            x_seq = jnp.concatenate([x_first, x_rest], axis=1)
        else:
            x_seq = x_first

        lstm_out = x_seq
        for i, (lstm_layer, ln) in enumerate(zip(self.lstm_layers, self.layer_norms)):
            lstm_new = lstm_layer(lstm_out)
            # Residual connection + LayerNorm for better gradient flow
            lstm_out = ln(lstm_out + lstm_new)
        
        # Take last timestep output: (B, d_model)
        final_out = lstm_out[:, -1, :]
        
        # Apply layer norm and get Q-value
        q_value = self.value_head(final_out).squeeze(-1)  # (B,)
        
        return q_value


class GCLSTMDiscreteCritic(nn.Module):
    """Goal-conditioned LSTM critic for discrete actions with ensemble support.
    
    Wraps LSTMThinkingCritic with optional ensemble functionality.
    Takes action indices as input and converts to one-hot encoding.
    """
    
    d_model: int = 256
    action_dim: int = 6
    thinking_steps: int = 3
    ensemble: bool = True
    gc_encoder: nn.Module = None
    layer_norm: bool = True
    pre_layer_norm: bool = True
    num_layers: int = 1
    
    def setup(self):
        critic_cls = LSTMThinkingCritic
        if self.ensemble:
            critic_cls = ensemblize(critic_cls, 2)
        
        self.critic = critic_cls(
            d_model=self.d_model,
            thinking_steps=self.thinking_steps,
            ensemble=False,  # Ensemble is handled by ensemblize wrapper
            gc_encoder=self.gc_encoder,
            layer_norm=self.layer_norm,
            pre_layer_norm=self.pre_layer_norm,
            num_layers=self.num_layers,
        )
    
    def __call__(self, observations, goals=None, actions=None):
        """Forward pass.
        
        Args:
            observations: Observation tensor of shape (B, obs_dim)
            goals: Goal tensor of shape (B, goal_dim) (optional)
            actions: Action indices of shape (B,) - will be converted to one-hot
        
        Returns:
            If ensemble=True: Q-values with shape (2, B)
            If ensemble=False: Q-values with shape (B,)
        """
        # Convert action indices to one-hot encoding (like GCDiscreteCritic)
        actions_onehot = jnp.eye(self.action_dim)[actions]
        q = self.critic(observations, goals, actions_onehot)
        return q


class GCDQNLSTMAgent(flax.struct.PyTreeNode):
    """Goal-conditioned DQN with LSTM thinking steps (discrete actions only).
    
    Uses an LSTM-based critic that performs multiple "thinking steps" before
    outputting Q-values, allowing for more sophisticated reasoning about
    state-goal relationships.
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def critic_loss(self, batch, grad_params):
        """Compute the DQN critic loss (discrete actions).

        Assumes:
         - batch['actions'] contains integer action indices.
         - critic takes actions as input and returns Q(s,a) scalar(s).
        """
        action_dim = 6
        
        # Current Q for taken actions - shape (2, B)
        q1_a, q2_a = self.network.select('critic')(
            batch['observations'], batch['value_goals'], batch['actions'], params=grad_params
        )
        # Ensure shapes are (batch,)
        q1_a = jnp.squeeze(q1_a, axis=-1) if q1_a.ndim > 1 else q1_a
        q2_a = jnp.squeeze(q2_a, axis=-1) if q2_a.ndim > 1 else q2_a

        # Target Q: use target critic to get Q-values for all actions, then take max
        # vmap over actions like in dqn.py
        all_actions = jnp.tile(jnp.arange(action_dim), (batch['next_observations'].shape[0], 1))  # B x action_dim
        qs = jax.lax.stop_gradient(
            jax.vmap(self.network.select('target_critic'), in_axes=(None, None, 1))(
                batch['next_observations'], batch['value_goals'], all_actions
            )
        )  # action_dim x 2 x B
        qs = qs.mean(axis=1)  # action_dim x B (average over ensemble)
        qs = qs.transpose(1, 0)  # B x action_dim
        max_next_q = jnp.max(qs, axis=-1)  # B

        # TD or MC target
        if self.config['use_discounted_mc_rewards']:
            target = batch['rewards']
        else:
            target = batch['rewards'] + self.config['discount'] * batch['masks'] * max_next_q

        # MSE loss on both heads
        critic_loss = ((q1_a - target) ** 2 + (q2_a - target) ** 2).mean()

        # Entropy regularization for action selection
        all_actions = jnp.tile(jnp.arange(action_dim), (batch['observations'].shape[0], 1))  # B x action_dim
        current_qs = jax.lax.stop_gradient(
            jax.vmap(self.network.select('critic'), in_axes=(None, None, 1))(
                batch['observations'],
                batch['value_goals'],  # <-- was jnp.roll(batch['next_observations'], ...) before -- not sure what is correct
                all_actions,
            )
        )  # action_dim x 2 x B
        if len(current_qs.shape) == 2:  # Non-ensemble
            current_qs = current_qs[:, None, ...]
        current_qs = current_qs.mean(axis=1)  # action_dim x B
        current_qs = current_qs.transpose(1, 0)  # B x action_dim

        alpha_temp = self.network.select('alpha_temp')(params=grad_params)
        dist = distrax.Categorical(logits=current_qs / jnp.maximum(1e-6, alpha_temp))
        entropy = dist.entropy()
        alpha_temp_loss = ((entropy + self.config['target_entropy']) ** 2).mean()

        total_loss = critic_loss + alpha_temp_loss

        return total_loss, {
            'critic_loss': critic_loss,
            'q_mean': target.mean(),
            'q_max': target.max(),
            'q_min': target.min(),
            'q_std': target.std(),
            'entropy': entropy.mean(),
            'alpha_temp': alpha_temp,
            'entropy_std': entropy.std(),
            'alpha_temp_loss': alpha_temp_loss,
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss (only critic loss for DQN)."""
        info = {}
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
        """Returns integer action indices using the LSTM critic."""
        if not self.config['discrete']:
            raise NotImplementedError("GCDQNLSTMAgent supports only discrete action spaces.")
        
        action_dim = 6
        
        # Get Q-values for all actions by vmapping over action indices (like dqn.py)
        all_actions = jnp.tile(jnp.arange(action_dim), (observations.shape[0], 1))  # B x action_dim
        qs = jax.lax.stop_gradient(
            jax.vmap(self.network.select('critic'), in_axes=(None, None, 1))(
                observations, goals, all_actions
            )
        )  # action_dim x 2 x B
        qs = qs.mean(axis=1)  # action_dim x B (average over ensemble)
        qs = qs.transpose(1, 0)  # B x action_dim

        if self.config['action_sampling'] == 'softmax':
            alpha_temp = jax.lax.stop_gradient(self.network.select('alpha_temp')())
            dist = distrax.Categorical(logits=qs / jnp.maximum(1e-6, alpha_temp))
            actions = dist.sample(seed=seed)
        elif self.config['action_sampling'] == 'epsilon_greedy':
            greedy_actions = jnp.argmax(qs, axis=-1)
            rng, rng_uniform = jax.random.split(seed)
            random_actions = jax.random.randint(rng, greedy_actions.shape, 0, action_dim)
            probs = jax.random.uniform(rng_uniform, greedy_actions.shape)
            actions = jnp.where(probs < self.config.get('epsilon', 0.1), random_actions, greedy_actions)
        elif self.config['action_sampling'] == 'greedy':
            actions = jnp.argmax(qs, axis=-1)
        else:
            raise ValueError(f"Unknown action sampling type: {self.config['action_sampling']}")

        return actions

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):
        """Create a new LSTM DQN agent (discrete only)."""
        if not config['discrete']:
            raise ValueError("GCDQNLSTMAgent supports only discrete action spaces.")

        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_goals = ex_observations
        action_dim = int(ex_actions.max() + 1)

        # Define encoders
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['critic'] = GCEncoder(concat_encoder=encoder_module())

        # LSTM-based discrete critic
        critic_def = GCLSTMDiscreteCritic(
            d_model=config['lstm_hidden_size'],
            action_dim=action_dim,
            thinking_steps=config['thinking_steps'],
            ensemble=config['ensemble'],
            gc_encoder=encoders.get('critic'),
            layer_norm=config['layer_norm'],
            num_layers=1,
        )


        if config['target_entropy'] is None:
            config['target_entropy'] = -config['target_entropy_multiplier'] * action_dim / 2
        alpha_temp_def = LogParam()

        network_info = dict(
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
        params['modules_target_critic'] = params['modules_critic']

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            # Agent hyperparameters
            agent_name='gcdqn_lstm',
            lr=3e-4,
            batch_size=1024,
            
            # LSTM architecture
            lstm_hidden_size=64,
            thinking_steps=3,
            
            # Legacy MLP dims (for compatibility with value/actor)
            actor_hidden_dims=(512, 512, 512),
            value_hidden_dims=(512, 512, 512),
            layer_norm=True,
            net_arch='mlp',
            
            # Training
            discount=0.99,
            tau=0.005,
            target_entropy=None,
            target_entropy_multiplier=0.5,
            use_discounted_mc_rewards=False,
            action_sampling='softmax',  # 'softmax', 'epsilon_greedy', or 'greedy'
            epsilon=0.1,  # For epsilon-greedy
            action_dim=6,
            
            # Legacy fields for compatibility
            expectile=0.9,
            actor_loss='ddpgbc',
            alpha=0.3,
            const_std=True,
            discrete=True,
            encoder=ml_collections.config_dict.placeholder(str),
            
            # Dataset hyperparameters
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
            ensemble=True,
        )
    )
    return config