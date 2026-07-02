from .block_moving_env import BoxMovingEnv
from .wrappers import AutoResetWrapper
import jax

if __name__ == "__main__":
    env = BoxMovingEnv(
        grid_size=6,
        number_of_boxes_max=1,
        number_of_boxes_min=1,
        number_of_moving_boxes_max=1,
        level_generator="variable",
        generator_special=False,
        dense_rewards=False,
        terminate_when_success=True,
        episode_length=10,
        quarter_size=1,
    )
    env = AutoResetWrapper(env)
    key = jax.random.PRNGKey(0)
    env.play_game(key)
