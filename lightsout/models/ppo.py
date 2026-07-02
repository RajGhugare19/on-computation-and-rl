import jax.numpy as jnp
from flax import linen as nn
import distrax
from typing import Sequence, Any
from models.networks import MLP

import jax
import jax.numpy as jnp
import flax.linen as nn
import distrax
from typing import Any

class InterpolationThinkingCell(nn.Module):
    hidden_dim: int
    layer_norm: bool = True

    def setup(self):
        self.recurrence_ln = nn.LayerNorm()
        self.forget_gate = nn.Dense( self.hidden_dim, bias_init=nn.initializers.constant(1.0) )        
        self.input_gate = nn.Dense( self.hidden_dim )

    def __call__(self, c, h, x_context):
        # Combine static context (x) with dynamic thought (h)
        combined = jnp.concatenate([x_context, h], axis=-1)

        # Apply Layer Norm
        if self.layer_norm:
            combined = self.recurrence_ln(combined)

        # Compute Gates
        f = nn.sigmoid(self.forget_gate(combined))
        i = jnp.tanh(self.input_gate(combined))

        # Interpolation Update
        c_new = c * f + i * (1.0 - f)
        h_new = jnp.tanh(c_new)

        return c_new, h_new
        
class InterpolationThinkerActorValue(nn.Module):
    hidden_state_dim: int = 64
    num_layers: int = 2
    thinking_steps: int = 5
    output_dim_1: int = 1
    output_dim_2: int = 1
    layer_norm: bool = True
    pre_layer_norm: bool = True

    def setup(self):
        # 1. Input Processing
        self.input_embed_layer = nn.Dense(self.hidden_state_dim)
        if self.pre_layer_norm:
            self.input_ln = nn.LayerNorm()

        self.interp_layers = [
            InterpolationThinkingCell(
                hidden_dim=self.hidden_state_dim,
                layer_norm=self.layer_norm
            )
            for _ in range(self.num_layers)
        ]

        # 3. Output Heads
        self.value_head = nn.Dense(
            self.output_dim_2,
            # kernel_init=nn.initializers.orthogonal(1.0),
            bias_init=nn.initializers.zeros,
        )
        self.actor_head = nn.Dense(
            self.output_dim_1,
            # kernel_init=nn.initializers.orthogonal(0.01),
            bias_init=nn.initializers.zeros,
        )

    def __call__(self, x, normalizer_params=None):
        if normalizer_params is not None:
            x = (x - normalizer_params.mean ) / (normalizer_params.std)

        input_shape = x.shape
        x_flat = x.reshape(-1, input_shape[-1])

        # Embed x
        x_emb = self.input_embed_layer(x_flat)
        if self.pre_layer_norm:
            x_emb = self.input_ln(x_emb)

        # --- The "Thinking" Loop ---
        batch_size = x_emb.shape[0]
        # Initialize state
        c = jnp.zeros((batch_size, self.hidden_state_dim))
        h = jnp.zeros((batch_size, self.hidden_state_dim))
                
        for _ in range(self.thinking_steps):
            h_block_input = h
            for layer in self.interp_layers:
                c, h = layer(c, h, x_emb)
            h = h_block_input + h

        # --- Outputs ---
        # We use the final 'h' state after N thinking steps
        logits = self.actor_head(h) 
        logits = logits.reshape(input_shape[:-1] + (-1,))
        
        value = self.value_head(h)
        value = value.reshape(input_shape[:-1] + (-1,))

        return distrax.Categorical(logits=logits), jnp.squeeze(value, axis=-1)
                
class LSTMThinkerActorValue(nn.Module):
    hidden_state_dim: int = 64
    num_layers: int = 2
    thinking_steps: int = 5
    output_dim_1: int = 1
    output_dim_2: int = 1
    layer_norm: bool = True
    pre_layer_norm: bool = True
    
    def setup(self):
        self.input_embed_layer = nn.Dense(self.hidden_state_dim)
        self.pre_ln = nn.LayerNorm()
        self.start_token = self.param(
            'start_token', 
            nn.initializers.normal(stddev=0.02), 
            (1, 1, self.hidden_state_dim)
        )
        self.lstm_layers = [
            nn.RNN(nn.LSTMCell(features=self.hidden_state_dim)) 
            for _ in range(self.num_layers)
        ]
        # Layer norms between LSTM layers for stability
        self.layer_norms = [nn.LayerNorm() for _ in range(self.num_layers)]

        self.value_head = nn.Dense(
            self.output_dim_2,
            kernel_init=nn.initializers.orthogonal(1.0),
            bias_init=nn.initializers.zeros,
        )
        self.actor_head = nn.Dense(
            self.output_dim_1,
            kernel_init=nn.initializers.orthogonal(1.0),
            bias_init=nn.initializers.zeros,
        )
    
    def __call__(self, x, normalizer_params=None):
        if normalizer_params is not None:
            x = (x - normalizer_params.mean ) / (normalizer_params.std)

        input_shape = x.shape
        x_flat = x.reshape(-1, input_shape[-1])

        x_emb = self.input_embed_layer(x_flat)
        if self.pre_layer_norm:
            x_emb = self.pre_ln(x_emb)
        x_emb = jnp.expand_dims(x_emb, axis=1)
        
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
        final_out = lstm_out[:, -1:, :]

        logits = self.actor_head(final_out).squeeze(axis=-2)
        logits = logits.reshape(input_shape[:-1] + (-1,))

        value = self.value_head(final_out).squeeze(axis=-2)
        value = value.reshape(input_shape[:-1] + (-1,))

        return distrax.Categorical(logits=logits), jnp.squeeze(value, axis=-1)

class MLPActorValue(nn.Module):
    layer_sizes: Sequence[int]
    output_dim_1: int = 1
    output_dim_2: int = 1
    activation: Any = nn.swish
    layer_norm: bool = False
    _min_std: float = 0.001
    _var_scale: float = 1
    
    def setup(self):
        self.trunk = MLP(self.layer_sizes, activation=self.activation, layer_norm=self.layer_norm, kernel_init=nn.initializers.orthogonal(1.0), bias_init=nn.initializers.zeros)
        self.logits = nn.Dense(self.output_dim_1, kernel_init=nn.initializers.orthogonal(1.0), bias_init=nn.initializers.zeros)
        self.value = nn.Dense(self.output_dim_2, kernel_init=nn.initializers.orthogonal(1.0), bias_init=nn.initializers.zeros)

    def __call__(self, x, normalizer_params=None):
        if normalizer_params is not None:
            x = (x - normalizer_params.mean ) / (normalizer_params.std)
        x = self.trunk(x)
        logits = self.logits(x)
        value = self.value(x)
        return distrax.Categorical(logits=logits), jnp.squeeze(value, axis=-1)

    def get_representation(self, x, normalizer_params=None):
        """Returns the activated penultimate layer output."""
        if normalizer_params is not None:
            x = (x - normalizer_params.mean ) / (normalizer_params.std)
        representation = self.trunk(x)
        return representation

class ResidualBlock(nn.Module):
    use_ln: bool = True
    activation_fn: str = "relu"
    kernel_init: Any = None
    bias_init: Any = nn.initializers.zeros

    @nn.compact
    def __call__(self, x):
        hidden_size = x.shape[-1]
        residual = x

        # Layer 0
        x = nn.relu(x) if self.activation_fn == "relu" else nn.swish(x)
        if self.kernel_init is not None:
            x = nn.Dense(hidden_size, kernel_init=self.kernel_init, bias_init=self.bias_init)(x)
        else:
            x = nn.Dense(hidden_size, bias_init=self.bias_init)(x)
        if self.use_ln:
            x = nn.LayerNorm()(x)

        # Layer 1
        x = nn.relu(x) if self.activation_fn == "relu" else nn.swish(x)
        if self.kernel_init is not None:
            x = nn.Dense(hidden_size, kernel_init=self.kernel_init, bias_init=self.bias_init)(x)
        else:
            x = nn.Dense(hidden_size, bias_init=self.bias_init)(x)
        if self.use_ln:
            x = nn.LayerNorm()(x)

        return x + residual

class ResnetActorValue(nn.Module):
    hidden_state_dim: int = 64
    num_layers: int = 2
    use_ln: bool = True
    activation_fn: str = "relu"
    output_dim_1: int = 1
    output_dim_2: int = 1
    kernel_init: Any = None
    bias_init: Any = nn.initializers.zeros

    @nn.compact
    def __call__(self, x, normalizer_params=None):
        if normalizer_params is not None:
            x = (x - normalizer_params.mean) / (normalizer_params.std + 1e-8)

        # Input Embedding
        if self.kernel_init is not None:
            x = nn.Dense(self.hidden_state_dim, kernel_init=self.kernel_init, bias_init=self.bias_init)(x)
        else:
            x = nn.Dense(self.hidden_state_dim, bias_init=self.bias_init)(x)

        # Independent Actor Trunk
        for _ in range(self.num_layers):
            x = ResidualBlock(use_ln=self.use_ln, activation_fn=self.activation_fn, kernel_init=self.kernel_init)(x)

        if self.use_ln:
            x = nn.LayerNorm()(x)

        if self.kernel_init is not None:
            logits = nn.Dense(self.output_dim_1, kernel_init=self.kernel_init, bias_init=self.bias_init)(x)
            value = nn.Dense(self.output_dim_2, kernel_init=self.kernel_init, bias_init=self.bias_init)(x)
        else:
            logits = nn.Dense(self.output_dim_1, bias_init=self.bias_init)(x)
            value = nn.Dense(self.output_dim_2, bias_init=self.bias_init)(x)

        return distrax.Categorical(logits=logits), jnp.squeeze(value, axis=-1)

class ResnetActor(nn.Module):
    hidden_state_dim: int = 64
    num_layers: int = 2
    use_ln: bool = True
    activation_fn: str = "relu"
    output_dim: int = 1
    kernel_init: Any = None
    bias_init: Any = nn.initializers.zeros

    @nn.compact
    def __call__(self, x, normalizer_params=None):
        if normalizer_params is not None:
            x = (x - normalizer_params.mean) / (normalizer_params.std + 1e-8)

        # Input Embedding
        if self.kernel_init is not None:
            x = nn.Dense(self.hidden_state_dim, kernel_init=self.kernel_init, bias_init=self.bias_init)(x)
        else:
            x = nn.Dense(self.hidden_state_dim, bias_init=self.bias_init)(x)

        # Independent Actor Trunk
        for _ in range(self.num_layers):
            x = ResidualBlock(use_ln=self.use_ln, activation_fn=self.activation_fn, kernel_init=self.kernel_init)(x)

        if self.use_ln:
            x = nn.LayerNorm()(x)

        if self.kernel_init is not None:
            logits = nn.Dense(self.output_dim, kernel_init=self.kernel_init, bias_init=self.bias_init)(x)
        else:
            logits = nn.Dense(self.output_dim, bias_init=self.bias_init)(x)
        return distrax.Categorical(logits=logits)

class ResnetValue(nn.Module):
    hidden_state_dim: int = 64
    num_layers: int = 2
    use_ln: bool = True
    activation_fn: str = "relu"
    output_dim: int = 1
    kernel_init: Any = None
    bias_init: Any = nn.initializers.zeros

    @nn.compact
    def __call__(self, x, normalizer_params=None):
        if normalizer_params is not None:
            x = (x - normalizer_params.mean) / (normalizer_params.std + 1e-8)

        # Input Embedding
        if self.kernel_init is not None:
            x = nn.Dense(self.hidden_state_dim, kernel_init=self.kernel_init, bias_init=self.bias_init)(x)
        else:
            x = nn.Dense(self.hidden_state_dim, bias_init=self.bias_init)(x)

        # Independent Value Trunk
        for _ in range(self.num_layers):
            x = ResidualBlock(use_ln=self.use_ln, activation_fn=self.activation_fn, kernel_init=self.kernel_init)(x)

        if self.use_ln:
            x = nn.LayerNorm()(x)

        if self.kernel_init is not None:
            value = nn.Dense(self.output_dim, kernel_init=self.kernel_init, bias_init=self.bias_init)(x)
        else:
            value = nn.Dense(self.output_dim, bias_init=self.bias_init)(x)
        return jnp.squeeze(value, axis=-1)