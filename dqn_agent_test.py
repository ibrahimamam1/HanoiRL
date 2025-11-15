import gymnasium as gym
from gymnasium.envs.registration import register
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env

register(
    id="HanoiEnv-v0",
    entry_point="env:HanoiEnv",
)

env = gym.make("HanoiEnv-v0", n_disks=3, render_mode="human")

print("Checking environment...")
check_env(env)
print("Environment check passed!")

model = DQN.load("dqn_hanoi")
obs, info = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
