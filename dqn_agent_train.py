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

model = DQN("MultiInputPolicy", env, verbose=1, tensorboard_log="./tensorboard/")

print("Starting training...")
model.learn(total_timesteps=100000, log_interval=4, progress_bar=True)
print("Training complete!")

model.save("dqn_hanoi")
