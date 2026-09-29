"""LineWorld: the 6-cell Gymnasium environment used by the 00_0x micro-course notebooks."""
import gymnasium as gym
from gymnasium import spaces


class LineWorld(gym.Env):
    def __init__(self, n=6, max_steps=30):
        self.n = n
        self.goal = n - 1
        self.max_steps = max_steps
        self.action_space = spaces.Discrete(2)      # 0 LEFT, 1 RIGHT
        self.observation_space = spaces.Discrete(n)
        self.position = 0
        self.steps = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.position = 0
        self.steps = 0
        return self.position, {}

    def step(self, action):
        self.steps += 1
        old = self.position
        if action == 0:
            self.position = max(0, self.position - 1)
        else:
            self.position = min(self.goal, self.position + 1)

        terminated = self.position == self.goal
        truncated = self.steps >= self.max_steps
        reward = 1.0 if terminated else -0.02
        info = {"old_position": old}
        return self.position, reward, terminated, truncated, info
