import gymnasium as gym
import numpy as np
from collections import defaultdict
from typing import Dict, Any, Tuple
import pickle

class QLearningAgent:
    def __init__(
        self,
        env: gym.Env,
        lr: float = 0.01,
        discount_factor: float = 0.95,
        start_epsilon: float = 0.45,
        epsilon_decay: float = 0.1,
        final_epsilon: float = 0.1
    ):
        self.env = env 
        self.alpha = lr
        self.gamma = discount_factor
        self.epsilon = start_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        # Q-table uses a hashable state tuple as the key
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))
        self.training_error = []

    def _to_tuple(self, obs: Dict[str, np.ndarray]) -> Tuple:
        """Converts the dict observation to a hashable tuple representation."""
        # Note: This relies on the keys being consistent: 'tower 1', 'tower 2', 'tower 3'
        state_tuple = (
            tuple(obs["tower 1"].tolist()),
            tuple(obs["tower 2"].tolist()),
            tuple(obs["tower 3"].tolist())
        )
        return state_tuple

    def get_action(self, obs: Dict[str, np.ndarray]) -> int:
        state = self._to_tuple(obs) # Use hashable state
        if(np.random.random() < self.epsilon):
            return self.env.action_space.sample()
        else:
            # Use the hashable state for Q-value lookup
            return int(np.argmax(self.q_values[state]))
 
    def update(
        self,
        obs: Dict[str, np.ndarray],
        action: int ,
        reward: float,
        terminated: bool,
        next_obs: Dict[str, np.ndarray]
    ):
        state = self._to_tuple(obs)
        next_state = self._to_tuple(next_obs)
        
        future_q = (not terminated) * np.max(self.q_values[next_state])
        target = reward + self.gamma * future_q
        
        # Q-value lookup and update using the hashable state
        td = target - self.q_values[state][action]
        self.q_values[state][action] = (
            self.q_values[state][action] + self.alpha * td
        )

        self.training_error.append(td)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)


    def save(self, filename="q_values.pkl"):
        with open(filename, 'wb') as f:
            pickle.dump(dict(self.q_values), f)

    def load(self, filename="q_values.pkl"):
        try:
            with open(filename, 'rb') as f:
                self.q_values = defaultdict(lambda: np.zeros(self.env.action_space.n), pickle.load(f))
            print(f"Q-values loaded from {filename}")
        except FileNotFoundError:
            print(f"No Q-values file found at {filename}, starting with an empty Q-table.")

# Training hyperparameters
learning_rate = 0.01    
n_episodes = 3000      
start_epsilon = 1.0         
epsilon_decay = start_epsilon / (n_episodes / 2)  
final_epsilon = 0.1         

from gymnasium.envs.registration import register
register(
    id="HanoiEnv-v0",
    entry_point="env:HanoiEnv",
)

env = gym.make("HanoiEnv-v0", n_disks=3, render_mode="human")

agent = QLearningAgent(
    env=env,
    lr=learning_rate,
    start_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
)

from tqdm import tqdm
for episode in tqdm(range(n_episodes)):
    obs, info = env.reset()
    done = False

    while not done:
        action = agent.get_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        agent.update(obs, action, reward, terminated, next_obs)

        done = terminated or truncated
        obs = next_obs

    agent.decay_epsilon()

agent.save()
