"""
Configuration loader for the LDPC decoder.

This module provides functionality to load, validate, and access the YAML configuration
for the LDPC decoder. It ensures that all parameters are properly loaded and validated,
with no hardcoded values in the codebase.
"""

import os
import yaml
from typing import Dict, Any, Optional, Union
import logging
from pprint import pformat

# Set up logging
logger = logging.getLogger(__name__)

class ConfigValidator:
    """Validates the configuration against the expected schema."""
    
    @staticmethod
    def validate_system_config(config: Dict[str, Any]) -> bool:
        """Validate system configuration section."""
        required_keys = ['seed', 'device', 'num_workers', 'log_level']
        return all(key in config for key in required_keys)
    
    @staticmethod
    def validate_code_config(config: Dict[str, Any]) -> bool:
        """Validate code configuration section."""
        if 'type' not in config:
            return False
        
        code_type = config['type']
        if code_type not in ['lifted_product', 'balanced_product']:
            return False
            
        if 'parameters' not in config:
            return False
            
        return True
    
    @staticmethod
    def validate_error_model_config(config: Dict[str, Any]) -> bool:
        """Validate error model configuration section."""
        required_keys = ['primary_type', 'error_rate']
        return all(key in config for key in required_keys)
    
    @staticmethod
    def validate_environment_config(config: Dict[str, Any]) -> bool:
        """Validate environment configuration section."""
        required_keys = ['max_steps', 'observation_type', 'reward_normalization']
        if not all(key in config for key in required_keys):
            return False
            
        # Check reward normalization
        reward_norm = config['reward_normalization']
        if not all(key in reward_norm for key in ['enabled', 'min_reward', 'max_reward']):
            return False
            
        # Check shape shifting if enabled
        if 'shape_shifting' in config and config['shape_shifting']['enabled']:
            if 'curriculum_stage_rewards' not in config['shape_shifting']:
                return False
                
        return True
    
    @staticmethod
    def validate_rl_config(config: Dict[str, Any]) -> bool:
        """Validate RL configuration section."""
        if 'algorithm' not in config or config['algorithm'] != 'PPO':
            return False
            
        required_keys = ['hyperparameters', 'policy_kwargs']
        return all(key in config for key in required_keys)
    
    @staticmethod
    def validate_gnn_config(config: Dict[str, Any]) -> bool:
        """Validate GNN configuration section."""
        required_keys = ['hidden_channels', 'num_layers', 'dropout']
        if not all(key in config for key in required_keys):
            return False
            
        # Validate attention config if present
        if 'attention' in config:
            attention_keys = ['enabled', 'num_heads', 'attention_dropout']
            if not all(key in config['attention'] for key in attention_keys):
                return False
                
        # Validate residual config if present
        if 'residual' in config:
            residual_keys = ['enabled', 'layer_norm']
            if not all(key in config['residual'] for key in residual_keys):
                return False
                
        # Validate architecture config if present
        if 'architecture' in config:
            arch_keys = ['input_embedding', 'global_pooling', 'activation']
            if not all(key in config['architecture'] for key in arch_keys):
                return False
                
        return True
    
    @staticmethod
    def validate_curriculum_config(config: Dict[str, Any]) -> bool:
        """Validate curriculum learning configuration section."""
        if 'enabled' not in config:
            return False
            
        if config['enabled'] and 'stages' not in config:
            return False
            
        return True
    
    @staticmethod
    def validate_training_config(config: Dict[str, Any]) -> bool:
        """Validate training configuration section."""
        required_keys = ['total_timesteps', 'eval_freq', 'save_freq']
        return all(key in config for key in required_keys)
    
    @staticmethod
    def validate_evaluation_config(config: Dict[str, Any]) -> bool:
        """Validate evaluation configuration section."""
        required_keys = ['metrics']
        return all(key in config for key in required_keys)
    
    @staticmethod
    def validate_simulator_config(config: Dict[str, Any]) -> bool:
        """Validate simulator configuration section."""
        required_keys = ['shots', 'seed']
        return all(key in config for key in required_keys)
        
    @classmethod
    def validate_config(cls, config: Dict[str, Any]) -> bool:
        """Validate the entire configuration."""
        required_sections = [
            'system', 'code', 'error_model', 'environment', 
            'rl', 'gnn', 'curriculum', 'training', 'evaluation', 'simulator'
        ]
        
        # Check if all required sections exist
        if not all(section in config for section in required_sections):
            missing = [s for s in required_sections if s not in config]
            logger.error(f"Missing configuration sections: {missing}")
            return False
        
        # Validate each section
        validators = {
            'system': cls.validate_system_config,
            'code': cls.validate_code_config,
            'error_model': cls.validate_error_model_config,
            'environment': cls.validate_environment_config,
            'rl': cls.validate_rl_config,
            'gnn': cls.validate_gnn_config,
            'curriculum': cls.validate_curriculum_config,
            'training': cls.validate_training_config,
            'evaluation': cls.validate_evaluation_config,
            'simulator': cls.validate_simulator_config
        }
        
        for section, validator in validators.items():
            if not validator(config[section]):
                logger.error(f"Invalid configuration in section: {section}")
                return False
                
        return True


class Config:
    """
    Configuration manager for the LDPC decoder.
    
    This class loads, validates, and provides access to the configuration parameters.
    It ensures that all parameters are properly loaded from the YAML file with no
    hardcoded values in the codebase.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to the YAML configuration file.
        """
        self.config_path = config_path
        self.config = self._load_config()
        
        # Validate the configuration
        if not ConfigValidator.validate_config(self.config):
            raise ValueError(f"Invalid configuration in {config_path}")
            
        logger.info(f"Configuration loaded successfully from {config_path}")
        
    def _load_config(self) -> Dict[str, Any]:
        """
        Load the configuration from the YAML file.
        
        Returns:
            The configuration dictionary.
        """
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
            
        with open(self.config_path, 'r') as f:
            try:
                config = yaml.safe_load(f)
                logger.debug(f"Loaded configuration:\n{pformat(config)}")
                return config
            except yaml.YAMLError as e:
                logger.error(f"Error parsing YAML configuration: {e}")
                raise
                
    def get(self, section: str, key: Optional[str] = None, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            section: The configuration section.
            key: The configuration key within the section (optional).
            default: Default value if the key is not found.
            
        Returns:
            The configuration value.
        """
        if section not in self.config:
            logger.warning(f"Configuration section not found: {section}")
            return default
            
        if key is None:
            return self.config[section]
            
        if key not in self.config[section]:
            logger.warning(f"Configuration key not found: {section}.{key}")
            return default
            
        return self.config[section][key]
        
    def get_nested(self, path: str, default: Any = None) -> Any:
        """
        Get a nested configuration value using dot notation.
        
        Args:
            path: The configuration path (e.g., "system.seed").
            default: Default value if the path is not found.
            
        Returns:
            The configuration value.
        """
        keys = path.split('.')
        value = self.config
        
        for key in keys:
            if not isinstance(value, dict) or key not in value:
                logger.warning(f"Configuration path not found: {path}")
                return default
            value = value[key]
            
        return value
        
    def __getitem__(self, key: str) -> Any:
        """
        Get a configuration section using dictionary-like access.
        
        Args:
            key: The configuration section.
            
        Returns:
            The configuration section.
        """
        if key not in self.config:
            raise KeyError(f"Configuration section not found: {key}")
            
        return self.config[key]
        
    def __contains__(self, key: str) -> bool:
        """
        Check if a configuration section exists.
        
        Args:
            key: The configuration section.
            
        Returns:
            True if the section exists, False otherwise.
        """
        return key in self.config
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Get the entire configuration as a dictionary.
        
        Returns:
            The configuration dictionary.
        """
        return self.config.copy()


def load_config(config_path: str) -> Config:
    """
    Load and validate a configuration file.
    
    Args:
        config_path: Path to the YAML configuration file.
        
    Returns:
        A Config object.
    """
    return Config(config_path)
