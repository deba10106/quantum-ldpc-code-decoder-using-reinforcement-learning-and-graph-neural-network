"""
Mock implementation of gymnasium for testing.
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional


class Space:
    """Base class for all spaces."""
    def sample(self):
        """Sample from the space."""
        return None
        
    def contains(self, x):
        """Check if x is in the space."""
        return True


class Box(Space):
    """Box space."""
    def __init__(self, low, high, shape=None, dtype=np.float32):
        """Initialize the box space."""
        self.low = low
        self.high = high
        self.shape = shape
        self.dtype = dtype
        
    def sample(self):
        """Sample from the box space."""
        return np.zeros(self.shape, dtype=self.dtype)


class Discrete(Space):
    """Discrete space."""
    def __init__(self, n):
        """Initialize the discrete space."""
        self.n = n
        
    def sample(self):
        """Sample from the discrete space."""
        return 0


class Env:
    """Base class for all environments."""
    def __init__(self):
        """Initialize the environment."""
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(10,))
        self.action_space = Discrete(2)
        
    def reset(self, seed=None, options=None):
        """Reset the environment."""
        return np.zeros(10), {}
        
    def step(self, action):
        """Take a step in the environment."""
        return np.zeros(10), 0.0, False, False, {}
        
    def render(self):
        """Render the environment."""
        pass
        
    def close(self):
        """Close the environment."""
        pass


# Create a mock gym module
class GymModule:
    """Mock gym module."""
    def __init__(self):
        """Initialize the mock gym module."""
        self.Env = Env
        self.spaces = type('spaces', (), {
            'Box': Box,
            'Discrete': Discrete,
            'Space': Space
        })
        
    def make(self, env_id, **kwargs):
        """Create an environment."""
        return Env()


# Create the mock gymnasium module
gymnasium = GymModule()
