import gymnasium as gym
from gymnasium.envs.registration import register
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback

register(
    id="HanoiEnv-v0",
    entry_point="env:HanoiEnv",
)

env = gym.make("HanoiEnv-v0", n_disks=3, render_mode="human")
model = DQN("MultiInputPolicy", env, verbose=1, tensorboard_log="./tensorboard")

print("Starting training...")
model.learn(total_timesteps=10000, log_interval=4, progress_bar=True)
print("Training complete!")

model.save("dqn_hanoi")
