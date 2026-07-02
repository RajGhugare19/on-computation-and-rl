from .rb import get_discounted_rewards
import jax.numpy as jnp


def compute_discounted_rewards_naive(rewards, gamma):
    final_rewards = []
    for episode_rewards in rewards:
        ep_final_rewards = []

        discounted_sum = 0.0

        for r in reversed(episode_rewards):
            discounted_sum = r + gamma * discounted_sum
            ep_final_rewards.append(discounted_sum)

        ep_final_rewards.reverse()
        final_rewards.extend(ep_final_rewards)

    return jnp.array(final_rewards)


def flatten_list(lst_of_lists):
    flat_list = []
    for lst in lst_of_lists:
        flat_list.extend(lst)
    return flat_list


def test_basic():
    rewards = [[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.0, 1.0]]

    gamma = 0.99

    correct_rewards = compute_discounted_rewards_naive(rewards, gamma)

    steps = [list(range(len(r))) for r in rewards]
    flat_steps = jnp.array(flatten_list(steps))

    flat_rewards = jnp.array(flatten_list(rewards))

    computed_rewards = get_discounted_rewards(flat_steps, flat_rewards, gamma)
    assert jnp.allclose(computed_rewards, correct_rewards, atol=1e-5), (
        f"Expected {correct_rewards}, but got {computed_rewards}"
    )


def test_negative_basic():
    rewards = [[-1.0, -1.0, 0.0, -1.0], [-1.0, -1.0, 0.0], [-1.0, -1.0, 0.0, -1.0, 0.0]]

    gamma = 0.99

    correct_rewards = compute_discounted_rewards_naive(rewards, gamma)

    steps = [list(range(len(r))) for r in rewards]
    flat_steps = jnp.array(flatten_list(steps))

    flat_rewards = jnp.array(flatten_list(rewards))

    computed_rewards = get_discounted_rewards(flat_steps, flat_rewards, gamma)
    assert jnp.allclose(computed_rewards, correct_rewards, atol=1e-5), (
        f"Expected {correct_rewards}, but got {computed_rewards}"
    )


def test_wrap_around():
    rewards = [[0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.0, 1.0]]

    gamma = 0.99

    correct_rewards = compute_discounted_rewards_naive(rewards, gamma)

    correct_rewards = jnp.roll(correct_rewards, shift=2)

    steps = [list(range(len(r))) for r in rewards]
    flat_steps = jnp.array(flatten_list(steps))
    flat_steps = jnp.roll(flat_steps, shift=2)

    flat_rewards = jnp.array(flatten_list(rewards))
    flat_rewards = jnp.roll(flat_rewards, shift=2)

    computed_rewards = get_discounted_rewards(flat_steps, flat_rewards, gamma)
    assert jnp.allclose(computed_rewards, correct_rewards, atol=1e-5), (
        f"Expected {correct_rewards}, but got {computed_rewards}"
    )
