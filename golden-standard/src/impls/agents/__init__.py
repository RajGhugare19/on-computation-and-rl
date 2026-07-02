from impls.agents.clearn_search import ClearnSearchAgent
from impls.agents.crl import CRLAgent
from impls.agents.crl_search import CRLSearchAgent
from impls.agents.dqn import GCDQNAgent
from impls.agents.dqn_lstm import GCDQNLSTMAgent
from impls.agents.dqn_interp import GCDQNInterpAgent
from impls.agents.gcbc import GCBCAgent
from impls.agents.gciql import GCIQLAgent
from impls.agents.gciql_search import GCIQLSearchAgent
from impls.agents.qrl import QRLAgent
from impls.agents.sac import SACAgent
import ml_collections

agents = dict(
    crl=CRLAgent,
    gcbc=GCBCAgent,
    gciql=GCIQLAgent,
    qrl=QRLAgent,
    sac=SACAgent,
)


default_config = ml_collections.FrozenConfigDict(
        dict(
            # Agent hyperparameters.
            agent_name='crl',  # Agent name.
            lr=3e-4,  # Learning rate.
            batch_size=256,  # Batch size.
            actor_hidden_dims=(256, 256),  # Actor network hidden dimensions.
            value_hidden_dims=(256, 256),  # Value network hidden dimensions.
            latent_dim=64,
            net_arch='mlp',
            layer_norm=True,  # Whether to use layer normalization.
            discount=0.99,  # Discount factor.
            contrastive_loss = 'binary',
            energy_fn = 'dot',
            logsumexp_coeff = 0.0, # Coefficient for logsumexp loss in critic loss
            actor_loss='awr',  # Actor loss type ('awr' or 'ddpgbc').
            alpha=0.1,  # Temperature in AWR or BC coefficient in DDPG+BC.
            tau=0.005,  # Target network update rate.
            expectile=0.9, # IQL expectile.
            actor_log_q=True,  # Whether to maximize log Q (True) or Q itself (False) in the actor loss.
            const_std=True,  # Whether to use constant standard deviation for the actor.
            discrete=True,  # Whether the action space is discrete.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
            # Dataset hyperparameters.
            use_next_obs=False, #TODO: This is not used anymore, we should remove it 
            target_entropy_multiplier=0.5,  # Multiplier for the target entropy (used in SAC-like agents).
            target_entropy=-1.1,  # Default target entropy for agents (-ln(|A|/2))
            use_discounted_mc_rewards=False,  # Whether to use discounted Monte Carlo rewards.
            action_sampling='softmax',
            is_td=False,
            ensemble= True,  # Whether to use ensemble of critics in GCDQNLSTMAgent.
            lstm_hidden_size=256,  # Hidden size for LSTM in GCDQNLSTMAgent.
            thinking_steps=2,  # Number of thinking steps for GCDQNLSTMAgent.
            num_layers=2,  # Number of LSTM layers for GCDQNLSTMAgent.
            recurrent_hidden_size=256,  # Hidden size for recurrent network in GCDQNLSTMAgent.
        )
    )

 
def create_agent(config: ml_collections.FrozenConfigDict, example_batch: dict, seed: int):
    if config.agent_name == "crl":
        agent = CRLAgent.create(
            seed,
            example_batch['observations'],
            example_batch['actions'],
            config,
            example_batch['value_goals'],
        )
    elif config.agent_name == "crl_search":
        agent = CRLSearchAgent.create(
            seed,
            example_batch['observations'],
            example_batch['actions'],
            config,
            example_batch['value_goals'],
        )
    elif config.agent_name == "gciql":
        agent = GCIQLAgent.create(
            seed,
            example_batch['observations'],
            example_batch['actions'],
            config,
        )
    elif config.agent_name == "gciql_search":
        agent = GCIQLSearchAgent.create(
            seed,
            example_batch['observations'],
            example_batch['actions'],
            config,
        )
    elif config.agent_name == "gcdqn":
        agent = GCDQNAgent.create(
            seed,
            example_batch['observations'],
            example_batch['actions'],
            config,
        )
    elif config.agent_name == "clearn_search":
        agent = ClearnSearchAgent.create(
            seed,
            example_batch['observations'],
            example_batch['actions'],
            config,
        )
    elif config.agent_name == "gcdqn_lstm":
        agent = GCDQNLSTMAgent.create(
            seed,
            example_batch['observations'],
            example_batch['actions'],
            config,
        )
    elif config.agent_name == "gcdqn_interp":
        agent = GCDQNInterpAgent.create(
            seed,
            example_batch['observations'],
            example_batch['actions'],
            config,
        )
    else:
        raise ValueError(f"Unknown agent class {config.agent_name}")

    return agent