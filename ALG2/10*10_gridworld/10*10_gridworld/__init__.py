from gym.envs.registration import register

register(
    id='10*10_gridworld-v0',
    entry_point='10*10_gridworld.envs:10*10env',
)