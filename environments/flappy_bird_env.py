"""
Gymnasium Environment Wrapper for Flappy Bird
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np

try:
    from .flappy_bird import FlappyBirdGame
except ImportError:
    from flappy_bird import FlappyBirdGame


class FlappyBirdEnv(gym.Env):
    """Flappy Bird Gymnasium Environment"""
    
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}
    
    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.game = FlappyBirdGame(render_mode=(render_mode == 'human'))
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=np.array([0, -20, 0, -500], dtype=np.float32),
            high=np.array([512, 20, 400, 500], dtype=np.float32),
            dtype=np.float32
        )
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        observation = self.game.reset()
        info = {'score': 0, 'frame': 0}
        return observation, info
    
    def step(self, action):
        observation, reward, done, info = self.game.step(action)
        terminated = done
        truncated = False
        return observation, reward, terminated, truncated, info
    
    def render(self):
        if self.render_mode == 'human':
            self.game.render()
    
    def close(self):
        self.game.close()