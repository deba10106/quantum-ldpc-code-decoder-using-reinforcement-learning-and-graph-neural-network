"""
Data augmentation strategies for LDPC decoder training.

This module provides various data augmentation techniques to improve
training with harder syndromes and diverse error patterns.
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import logging
from collections import deque
import torch

logger = logging.getLogger(__name__)

class SyndromeBuffer:
    """Buffer for storing and sampling hard syndromes."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize syndrome buffer."""
        self.capacity = config.get('buffer_capacity', 1000)
        self.min_reward = config.get('min_reward', -100.0)
        self.sampling_temp = config.get('sampling_temperature', 1.0)
        self.buffer = deque(maxlen=self.capacity)
        
    def add(self, syndrome: np.ndarray, reward: float, error_pattern: np.ndarray) -> None:
        """Add a syndrome to the buffer if it's hard enough."""
        if reward < self.min_reward:
            self.buffer.append({
                'syndrome': syndrome.copy(),
                'error_pattern': error_pattern.copy(),
                'reward': reward
            })
            logger.debug(f"Added hard syndrome to buffer (reward: {reward})")
            
    def sample(self, batch_size: int) -> List[Dict[str, np.ndarray]]:
        """Sample syndromes from the buffer."""
        if len(self.buffer) == 0:
            return []
            
        rewards = np.array([item['reward'] for item in self.buffer])
        probs = np.exp(-rewards / self.sampling_temp)
        probs = probs / np.sum(probs)
        
        indices = np.random.choice(
            len(self.buffer),
            size=min(batch_size, len(self.buffer)),
            p=probs,
            replace=False
        )
        
        return [self.buffer[i] for i in indices]

class NoiseInjector:
    """Inject additional noise into syndromes during training."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize noise injector."""
        self.noise_prob = config.get('noise_probability', 0.1)
        self.noise_types = config.get('noise_types', ['flip', 'swap', 'random'])
        self.noise_weights = config.get('noise_weights', [0.4, 0.3, 0.3])
        
    def inject_noise(self, syndrome: np.ndarray, error_pattern: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Inject noise into a syndrome."""
        if np.random.random() > self.noise_prob:
            return syndrome, error_pattern
            
        noise_type = np.random.choice(self.noise_types, p=self.noise_weights)
        
        syndrome_aug = syndrome.copy()
        error_pattern_aug = error_pattern.copy()
        
        if noise_type == 'flip':
            flip_mask = np.random.random(syndrome.shape) < 0.05
            syndrome_aug = np.logical_xor(syndrome_aug, flip_mask)
            
        elif noise_type == 'swap':
            if len(syndrome) > 1:
                idx = np.random.randint(0, len(syndrome) - 1)
                syndrome_aug[idx], syndrome_aug[idx + 1] = syndrome_aug[idx + 1], syndrome_aug[idx]
                
        elif noise_type == 'random':
            noise = np.random.random(error_pattern.shape) < 0.05
            error_pattern_aug = np.logical_xor(error_pattern_aug, noise)
            
        logger.debug(f"Injected {noise_type} noise into syndrome")
        return syndrome_aug, error_pattern_aug

class DataAugmentor:
    """Main class for data augmentation during training."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize data augmentor."""
        self.enabled = config.get('enabled', True)
        self.buffer = SyndromeBuffer(config.get('buffer', {}))
        self.noise_injector = NoiseInjector(config.get('noise_injection', {}))
        
    def augment(self, 
                syndrome: np.ndarray,
                error_pattern: np.ndarray,
                reward: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Augment a syndrome for training."""
        if not self.enabled:
            return syndrome, error_pattern
            
        if reward is not None:
            self.buffer.add(syndrome, reward, error_pattern)
            
        return self.noise_injector.inject_noise(syndrome, error_pattern)
        
    def sample_hard_syndromes(self, batch_size: int) -> List[Dict[str, np.ndarray]]:
        """Sample hard syndromes from the buffer."""
        return self.buffer.sample(batch_size)
