"""
Unit tests for the configuration loader module.
"""

import unittest
import os
import tempfile
import yaml
from unittest.mock import patch

from ldpc_decoder.config.config_loader import Config, ConfigValidator, load_config


class TestConfig(unittest.TestCase):
    """Test cases for the ConfigLoader class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a valid configuration dictionary
        self.valid_config = {
            'system': {
                'seed': 42,
                'device': 'cuda',
                'num_workers': 4,
                'log_level': 'INFO'
            },
            'code': {
                'type': 'lifted_product',
                'parameters': {
                    'n_checks': 32,
                    'n_bits': 64,
                    'distance': 8
                }
            },
            'error_model': {
                'primary_type': 'depolarizing',
                'error_rate': 0.01,
                'secondary_types': []
            },
            'environment': {
                'max_steps': 100,
                'observation_type': 'syndrome_graph',
                'reward_normalization': {
                    'enabled': True
                }
            },
            'rl': {
                'algorithm': 'PPO',
                'hyperparameters': {
                    'n_steps': 2048,
                    'batch_size': 64,
                    'learning_rate': 0.0003
                }
            },
            'gnn': {
                'model_type': 'GCN',
                'layers': 3,
                'hidden_channels': 64
            },
            'curriculum': {
                'enabled': True,
                'stages': [
                    {
                        'error_rate': 0.01,
                        'reward_scale': 1.0,
                        'steps': 100000
                    },
                    {
                        'error_rate': 0.05,
                        'reward_scale': 0.8,
                        'steps': 100000
                    }
                ]
            },
            'training': {
                'total_timesteps': 1000000,
                'eval_freq': 10000,
                'n_eval_episodes': 100,
                'save_path': 'models/trained_model'
            },
            'evaluation': {
                'metrics': [
                    'logical_error_rate',
                    'syndrome_resolution_rate'
                ]
            },
            'simulator': {
                'shots': 1000,
                'seed': 42,
                'circuit_type': 'stabilizer',
                'noise_model': {
                    'depolarizing': True,
                    'measurement': True
                }
            }
        }
        
        # Create a valid configuration file
        self.config_path = os.path.join(self.temp_dir, 'valid_config.yaml')
        with open(self.config_path, 'w') as f:
            yaml.dump(self.valid_config, f)
            
    def tearDown(self):
        """Clean up after tests."""
        # Remove temporary directory and its contents
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_load_config(self):
        """Test loading a valid configuration file."""
        # Load the configuration
        config = load_config(self.config_path)
        
        # Check that all sections are present
        for section in self.valid_config.keys():
            self.assertIn(section, config)
            
        # Check specific values
        self.assertEqual(config['system']['seed'], 42)
        self.assertEqual(config['code']['type'], 'lifted_product')
        self.assertEqual(config['simulator']['shots'], 1000)
        
    def test_load_nonexistent_config(self):
        """Test loading a nonexistent configuration file."""
        # Try to load a nonexistent file
        with self.assertRaises(FileNotFoundError):
            load_config('nonexistent_config.yaml')
            
    def test_load_invalid_yaml(self):
        """Test loading an invalid YAML file."""
        # Create an invalid YAML file
        invalid_path = os.path.join(self.temp_dir, 'invalid_config.yaml')
        with open(invalid_path, 'w') as f:
            f.write('invalid: yaml: content:')
            
        # Try to load the invalid file
        with self.assertRaises(yaml.YAMLError):
            load_config(invalid_path)
            
    def test_validate_config(self):
        """Test validating a valid configuration."""
        # Validate the configuration
        result = ConfigValidator.validate_config(self.valid_config)
        
        # Check that validation succeeded
        self.assertTrue(result)
        
    def test_validate_missing_section(self):
        """Test validating a configuration with a missing section."""
        # Create a configuration with a missing section
        invalid_config = self.valid_config.copy()
        del invalid_config['simulator']
        
        # Validate the configuration
        result = ConfigValidator.validate_config(invalid_config)
        
        # Check that validation failed
        self.assertFalse(result)
        
    def test_validate_invalid_system_config(self):
        """Test validating a configuration with an invalid system section."""
        # Create a configuration with an invalid system section
        invalid_config = self.valid_config.copy()
        invalid_config['system'] = {}
        
        # Validate the configuration
        result = ConfigValidator.validate_config(invalid_config)
        
        # Check that validation failed
        self.assertFalse(result)
        
    def test_validate_invalid_code_config(self):
        """Test validating a configuration with an invalid code section."""
        # Create a configuration with an invalid code section
        invalid_config = self.valid_config.copy()
        invalid_config['code'] = {}
        
        # Validate the configuration
        result = ConfigValidator.validate_config(invalid_config)
        
        # Check that validation failed
        self.assertFalse(result)
        
    def test_validate_invalid_error_model_config(self):
        """Test validating a configuration with an invalid error model section."""
        # Create a configuration with an invalid error model section
        invalid_config = self.valid_config.copy()
        invalid_config['error_model'] = {}
        
        # Validate the configuration
        result = ConfigValidator.validate_config(invalid_config)
        
        # Check that validation failed
        self.assertFalse(result)
        
    def test_validate_invalid_environment_config(self):
        """Test validating a configuration with an invalid environment section."""
        # Create a configuration with an invalid environment section
        invalid_config = self.valid_config.copy()
        invalid_config['environment'] = {}
        
        # Validate the configuration
        result = ConfigValidator.validate_config(invalid_config)
        
        # Check that validation failed
        self.assertFalse(result)
        
    def test_validate_invalid_rl_config(self):
        """Test validating a configuration with an invalid RL section."""
        # Create a configuration with an invalid RL section
        invalid_config = self.valid_config.copy()
        invalid_config['rl'] = {}
        
        # Validate the configuration
        result = ConfigValidator.validate_config(invalid_config)
        
        # Check that validation failed
        self.assertFalse(result)
        
    def test_validate_invalid_gnn_config(self):
        """Test validating a configuration with an invalid GNN section."""
        # Create a configuration with an invalid GNN section
        invalid_config = self.valid_config.copy()
        invalid_config['gnn'] = {}
        
        # Validate the configuration
        result = ConfigValidator.validate_config(invalid_config)
        
        # Check that validation failed
        self.assertFalse(result)
        
    def test_validate_invalid_curriculum_config(self):
        """Test validating a configuration with an invalid curriculum section."""
        # Create a configuration with an invalid curriculum section
        invalid_config = self.valid_config.copy()
        invalid_config['curriculum'] = {}
        
        # Validate the configuration
        result = ConfigValidator.validate_config(invalid_config)
        
        # Check that validation failed
        self.assertFalse(result)
        
    def test_validate_invalid_training_config(self):
        """Test validating a configuration with an invalid training section."""
        # Create a configuration with an invalid training section
        invalid_config = self.valid_config.copy()
        invalid_config['training'] = {}
        
        # Validate the configuration
        result = ConfigValidator.validate_config(invalid_config)
        
        # Check that validation failed
        self.assertFalse(result)
        
    def test_validate_invalid_evaluation_config(self):
        """Test validating a configuration with an invalid evaluation section."""
        # Create a configuration with an invalid evaluation section
        invalid_config = self.valid_config.copy()
        invalid_config['evaluation'] = {}
        
        # Validate the configuration
        result = ConfigValidator.validate_config(invalid_config)
        
        # Check that validation failed
        self.assertFalse(result)
        
    def test_validate_invalid_simulator_config(self):
        """Test validating a configuration with an invalid simulator section."""
        # Create a configuration with an invalid simulator section
        invalid_config = self.valid_config.copy()
        invalid_config['simulator'] = {}
        
        # Validate the configuration
        result = ConfigValidator.validate_config(invalid_config)
        
        # Check that validation failed
        self.assertFalse(result)


if __name__ == '__main__':
    unittest.main()
