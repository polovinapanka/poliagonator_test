from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from UE_env import UnrealEnv


env = Monitor(
    UnrealEnv()
)


model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
)


model.learn(
    total_timesteps=100000
)


model.save("ppo_unreal_100000steps")

env.close()