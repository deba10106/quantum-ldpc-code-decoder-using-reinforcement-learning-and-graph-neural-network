"""
Custom environment wrappers for LDPC decoder training.

This module provides custom environment wrappers for LDPC decoder training
that are compatible with stable-baselines3 and gymnasium.
"""

import time
import gymnasium as gym
from typing import Dict, Any, Optional, Tuple


class SimpleMonitorWrapper(gym.Wrapper):
    """
    A simple wrapper that tracks episode rewards and lengths without using Monitor.
    This avoids the episode_rewards AttributeError in SubprocVecEnv.
    """
    def __init__(self, env: gym.Env):
        """
        Initialize the wrapper.
        
        Args:
            env: The environment to wrap.
        """
        super().__init__(env)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_times = []
        self.total_steps = 0
        self._current_reward = 0
        self._episode_length = 0
        self._episode_start_time = time.time()
        
    def reset(self, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """
        Reset the environment and episode tracking.
        
        Args:
            **kwargs: Keyword arguments to pass to the environment reset.
            
        Returns:
            Tuple of observation and info dict.
        """
        obs, info = self.env.reset(**kwargs)
        self._current_reward = 0
        self._episode_length = 0
        self._episode_start_time = time.time()
        return obs, info
        
    def step(self, action) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """
        Step the environment and track episode data.
        
        Args:
            action: The action to take.
            
        Returns:
            Tuple of observation, reward, terminated, truncated, and info dict.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._current_reward += reward
        self._episode_length += 1
        self.total_steps += 1
        
        if terminated or truncated:
            # Record episode data
            episode_time = time.time() - self._episode_start_time
            self.episode_rewards.append(self._current_reward)
            self.episode_lengths.append(self._episode_length)
            self.episode_times.append(episode_time)
            
            # Add info for stable-baselines3
            ep_info = {
                "r": float(self._current_reward),
                "l": self._episode_length,
                "t": self.total_steps,
                "time": episode_time
            }
            if "episode" not in info:
                info["episode"] = ep_info
                
            # Reset counters
            self._current_reward = 0
            self._episode_length = 0
            self._episode_start_time = time.time()
            
        return obs, reward, terminated, truncated, info
    
    def get_episode_rewards(self) -> list:
        """
        Get the episode rewards.
        
        Returns:
            List of episode rewards.
        """
        return self.episode_rewards
    
    def get_episode_lengths(self) -> list:
        """
        Get the episode lengths.
        
        Returns:
            List of episode lengths.
        """
        return self.episode_lengths
    
    def get_episode_times(self) -> list:
        """
        Get the episode times.
        
        Returns:
            List of episode times.
        """
        return self.episode_times
