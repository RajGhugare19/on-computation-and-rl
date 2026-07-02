import glob
import os
import pickle

import flax
import flax.serialization
from impls.agents import create_agent


def save_agent(agent, config, save_dir, epoch):
    """Save the agent to a file.

    Args:
        agent: Agent.
        config: Config
        save_dir: Directory to save the agent.
        epoch: Epoch number.
    """

    save_dict = dict(
        agent=flax.serialization.to_state_dict(agent),
        config=config,
    )
    save_path = os.path.join(save_dir, f'params_{epoch}.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(save_dict, f)

    print(f'Saved to {save_path}')


def restore_agent(example_batch, restore_path, restore_epoch):
    candidates = glob.glob(restore_path)

    assert len(candidates) == 1, f'Found {len(candidates)} candidates: {candidates}'

    restore_path = candidates[0] + f'/params_{restore_epoch}.pkl'

    with open(restore_path, 'rb') as f:
        load_dict = pickle.load(f)

    config = load_dict['config']
    agent = create_agent(config.agent, example_batch, config.exp.seed)
    agent = flax.serialization.from_state_dict(agent, load_dict['agent'])

    print(f'Restored from {restore_path}')

    return agent, config