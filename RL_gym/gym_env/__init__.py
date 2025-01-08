from gym.envs.registration import register
import gym_env.ISAC_env

register(
    id='ISAC-ENV-v0',
    entry_point='gym_env.ISAC_env:ISACEnv',
)
