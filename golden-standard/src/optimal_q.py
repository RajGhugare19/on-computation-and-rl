from envs.block_moving.env_types import GridStatesEnum, BoxMovingState, TimeStep
from envs.block_moving.block_moving_env import BoxMovingEnv
import jax.numpy as jnp
import jax
from itertools import permutations, product
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import functools

actor_states = jnp.array(
    [
        GridStatesEnum.AGENT,
        GridStatesEnum.AGENT_CARRYING_BOX,
        GridStatesEnum.AGENT_ON_BOX,
        GridStatesEnum.AGENT_ON_TARGET,
        GridStatesEnum.AGENT_ON_TARGET_CARRYING_BOX,
        GridStatesEnum.AGENT_ON_TARGET_WITH_BOX,
        GridStatesEnum.AGENT_ON_TARGET_WITH_BOX_CARRYING_BOX,
        GridStatesEnum.AGENT_ON_BOX_CARRYING_BOX,
    ]
)

box_states = jnp.array(
    [
        GridStatesEnum.BOX,
        GridStatesEnum.AGENT_ON_BOX,
        GridStatesEnum.AGENT_ON_BOX_CARRYING_BOX,
    ]
)

target_states = jnp.array(
    [
        GridStatesEnum.TARGET,
        GridStatesEnum.AGENT_ON_TARGET,
        GridStatesEnum.AGENT_ON_TARGET_CARRYING_BOX,
    ]
)

carrying_states = jnp.array(
    [
        GridStatesEnum.AGENT_CARRYING_BOX,
        GridStatesEnum.AGENT_ON_BOX_CARRYING_BOX,
        GridStatesEnum.AGENT_ON_TARGET_CARRYING_BOX,
        GridStatesEnum.AGENT_ON_TARGET_WITH_BOX_CARRYING_BOX,
    ]
)
# EMPTY = jnp.int8(0)
# BOX = jnp.int8(1)
# TARGET = jnp.int8(2)
# AGENT = jnp.int8(3)
# AGENT_CARRYING_BOX = jnp.int8(4)  # Agent is carrying a box
# AGENT_ON_BOX = jnp.int8(5)  # Agent is on box
# AGENT_ON_TARGET = jnp.int8(6)  # Agent is on target
# AGENT_ON_TARGET_CARRYING_BOX = jnp.int8(7)  # Agent is on target carrying the box
# AGENT_ON_TARGET_WITH_BOX = jnp.int8(8)  # Agent is on target on which there is a box
# AGENT_ON_TARGET_WITH_BOX_CARRYING_BOX = jnp.int8(9)  # noqa: E501 Agent is on target on which there is a box and is carrying a box
# BOX_ON_TARGET = jnp.int8(10)  # Box is on target
# AGENT_ON_BOX_CARRYING_BOX = jnp.int8(11)  # Agent is on box carrying a box


# TODO: agent on box and agent on target states handling
def find_boxes(state):
    return jnp.argwhere(jnp.isin(state, box_states))


def find_goals(state):
    return jnp.argwhere(jnp.isin(state, target_states))


def find_agent(state):
    return jnp.argwhere(jnp.isin(state, actor_states))[0]


def generate_permutations(positions):
    return list(jnp.array(p) for p in permutations(positions))


def compute_cost(agent_pos, box_pos, goal_pos):
    moves = 0
    moves += jnp.abs(agent_pos - box_pos[0]).sum()
    moves += jnp.abs(box_pos - goal_pos).sum()
    moves += box_pos.shape[0]

    return moves


def translate_delta_to_action(delta):
    actions = []
    if delta[0] < 0:
        actions.extend([0] * int(-delta[0]))
    if delta[0] > 0:
        actions.extend([1] * int(delta[0]))
    if delta[1] < 0:
        actions.extend([2] * int(-delta[1]))
    if delta[1] > 0:
        actions.extend([3] * int(delta[1]))
    return actions


def retrieve_moves(agent_pos, box_pos, goal_pos):
    actions = []

    current_pos = agent_pos

    for box, goal in zip(box_pos, goal_pos):
        delta = box - current_pos
        actions.extend(translate_delta_to_action(delta))
        actions.append(4)
        delta = goal - box
        actions.extend(translate_delta_to_action(delta))
        actions.append(5)
        current_pos = goal

    return actions


def retrieve_moves_carrying(agent_pos, box_pos, goal_pos):
    actions = []

    current_pos = agent_pos

    for box, goal in zip(box_pos, goal_pos):
        delta = box - current_pos
        actions.extend(translate_delta_to_action(delta))
        actions.append(4)
        delta = goal - box
        actions.extend(translate_delta_to_action(delta))
        actions.append(5)
        current_pos = goal

    return actions


def solve_state(state: jnp.ndarray):
    agent_position = find_agent(state)
    if (state[agent_position[0], agent_position[1]] == carrying_states).any():
        if state[agent_position[0], agent_position[1]] == GridStatesEnum.AGENT_ON_TARGET_CARRYING_BOX:
            state = state.at[agent_position[0], agent_position[1]].set(GridStatesEnum.AGENT_ON_TARGET_WITH_BOX)
            moves = solve_normal_state(state)
            return [5] + moves
        else:
            return solve_carrying_state(state)
    else:
        return solve_normal_state(state)


def solve_carrying_state(state: jnp.ndarray):
    agent_position = find_agent(state)
    box_positions = find_boxes(state)
    goal_positions = find_goals(state)

    box_permutations = generate_permutations(box_positions)
    goal_permutations = generate_permutations(goal_positions)

    box_permutations = [jnp.concatenate([agent_position[None, ...], p]) for p in box_permutations]

    pairings = list(product(box_permutations, goal_permutations))

    min_cost = jnp.inf
    best_path = None

    for pairing in pairings:
        boxes, goals = pairing
        cost = compute_cost(agent_position, boxes, goals)
        if cost < min_cost:
            min_cost = cost
            best_path = (boxes, goals)

    best_moves = retrieve_moves(agent_position, best_path[0], best_path[1])

    assert best_moves[0] == 4  # first move must be pickup

    return best_moves[1:]


def solve_normal_state(state: jnp.ndarray):
    agent_position = find_agent(state)
    if (state[agent_position[0], agent_position[1]] == carrying_states).any():
        raise ValueError("State with agent carrying box is not supported.")

    box_positions = find_boxes(state)
    goal_positions = find_goals(state)
    box_permutations = generate_permutations(box_positions)
    goal_permutations = generate_permutations(goal_positions)

    pairings = list(product(box_permutations, goal_permutations))

    min_cost = jnp.inf
    best_path = None

    for pairing in pairings:
        boxes, goals = pairing
        cost = compute_cost(agent_position, boxes, goals)
        if cost < min_cost:
            min_cost = cost
            best_path = (boxes, goals)

    best_moves = retrieve_moves(agent_position, best_path[0], best_path[1])

    return best_moves


def calculate_optimal_q_value_and_traj(env: BoxMovingEnv, state: BoxMovingState, discount=0.99):
    best_moves = solve_state(state.grid)

    trajectory = []

    last_state = state

    rewards = []

    for action in best_moves:
        current_state, reward, _, _ = env.step(last_state, action)
        rewards.append(reward)
        timestep = TimeStep(**last_state.__dict__, action=action, done=jnp.array(False), truncated=jnp.array(False))
        trajectory.append(timestep)
        last_state = current_state

    timestep = TimeStep(
        **last_state.__dict__, action=jnp.array(action), done=jnp.array(False), truncated=jnp.array(False)
    )
    trajectory.append(timestep)

    q_value = 0

    for reward in rewards[::-1]:
        q_value = q_value * discount + reward

    return trajectory, q_value


if __name__ == "__main__":
    env = BoxMovingEnv(
        grid_size=3,
        episode_length=100,
        number_of_boxes_max=4,
        number_of_boxes_min=4,
        number_of_moving_boxes_max=4,
    )

    key = jax.random.PRNGKey(3)
    state, _ = env.reset(key)

    traj, q_value = calculate_optimal_q_value_and_traj(env, state)
    print("Optimal Q value:", q_value)

    traj_len = len(traj)

    traj = jax.tree.map(lambda *xs: jnp.array(xs)[None, ...], *traj)

    grid_size = traj.grid.shape[-2:]
    fig, ax = plt.subplots(figsize=grid_size)

    animate = functools.partial(env.animate, ax, traj, img_prefix="assets")

    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=traj_len + 1, interval=80, repeat=False)

    # Save as GIF
    gif_path = "block_moving_epoch.gif"
    anim.save(gif_path, writer="pillow")
    plt.close()
