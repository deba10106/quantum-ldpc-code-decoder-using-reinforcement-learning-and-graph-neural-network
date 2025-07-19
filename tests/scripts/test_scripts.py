"""
Unit tests for the scripts in the project.
"""

import unittest
import os
import sys
import tempfile
from unittest.mock import patch, MagicMock
import numpy as np

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import the script modules directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../scripts')))

# Now import the scripts
import run_benchmarks
import evaluate_model


class TestScripts(unittest.TestCase):
    """Test cases for the scripts in the project."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test outputs
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a minimal valid configuration dictionary for testing
        self.valid_config = {
            'system': {
                'seed': 42,
                'device': 'cpu',
                'log_level': 'INFO'
            },
            'code': {
                'type': 'lifted_product',
                'parameters': {
                    'n_checks': 8,
                    'n_bits': 16,
                    'distance': 4
                }
            },
            'error_model': {
                'primary_type': 'depolarizing',
                'error_rate': 0.01
            },
            'environment': {
                'max_steps': 20,
                'observation_type': 'syndrome_graph',
                'reward_normalization': {
                    'enabled': True
                }
            },
            'rl': {
                'algorithm': 'PPO',
                'hyperparameters': {
                    'n_steps': 128,
                    'batch_size': 32
                }
            },
            'gnn': {
                'model_type': 'GCN',
                'layers': 2,
                'hidden_channels': 32
            },
            'curriculum': {
                'enabled': True,
                'stages': [
                    {
                        'error_rate': 0.01,
                        'reward_scale': 1.0,
                        'steps': 1000
                    }
                ]
            },
            'training': {
                'total_timesteps': 2000,
                'eval_freq': 500,
                'n_eval_episodes': 10,
                'save_path': os.path.join(self.temp_dir, 'models/test_model')
            },
            'evaluation': {
                'metrics': ['logical_error_rate', 'syndrome_resolution_rate'],
                'n_episodes': 10,
                'error_rates': [0.01, 0.05],
                'decoder_types': ['gnn', 'mwpm'],
                'output_dir': os.path.join(self.temp_dir, 'results')
            },
            'simulator': {
                'shots': 10,
                'seed': 42
            }
        }
        
        # Create a temporary config file
        self.config_path = os.path.join(self.temp_dir, 'test_config.yaml')
        import yaml
        with open(self.config_path, 'w') as f:
            yaml.dump(self.valid_config, f)
            
        # Create a temporary model path
        self.model_dir = os.path.join(self.temp_dir, 'models')
        os.makedirs(self.model_dir, exist_ok=True)
        self.model_path = os.path.join(self.model_dir, 'test_model.zip')
        
        # Create a temporary results directory
        self.results_dir = os.path.join(self.temp_dir, 'results')
        os.makedirs(self.results_dir, exist_ok=True)
        
    def tearDown(self):
        """Clean up after tests."""
        # Remove temporary directory and its contents
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    @patch('scripts.run_benchmarks.DecoderEvaluator')
    @patch('scripts.run_benchmarks.load_config')
    @patch('scripts.run_benchmarks.LiftedProductCode')
    @patch('scripts.run_benchmarks.DepolarizingErrorModel')
    @patch('scripts.run_benchmarks.SimulatorIntegration')
    def test_run_benchmarks(self, mock_simulator, mock_error_model, mock_code, mock_config_loader, mock_evaluator):
        """Test the run_benchmarks script."""
        # Mock the config loader
        mock_config_loader.load_config.return_value = self.valid_config
        
        # Mock the evaluator
        mock_evaluator_instance = mock_evaluator.return_value
        mock_evaluator_instance.run_benchmarks.return_value = {
            'gnn': {
                '0.01': {
                    'success_rate': 0.9,
                    'logical_error_rate': 0.1
                },
                '0.05': {
                    'success_rate': 0.8,
                    'logical_error_rate': 0.2
                }
            },
            'mwpm': {
                '0.01': {
                    'success_rate': 0.7,
                    'logical_error_rate': 0.3
                },
                '0.05': {
                    'success_rate': 0.6,
                    'logical_error_rate': 0.4
                }
            }
        }
        mock_evaluator_instance.save_results.return_value = os.path.join(self.results_dir, 'results.json')
        mock_evaluator_instance.plot_results.return_value = os.path.join(self.results_dir, 'results.png')
        
        # Run the benchmarks script with command-line arguments
        with patch('sys.argv', ['run_benchmarks.py', '--config', self.config_path, '--output-dir', self.results_dir]):
            run_benchmarks.main()
            
        # Check that the config loader was called with the correct path
        mock_config_loader.load_config.assert_called_once_with(self.config_path)
        
        # Check that the evaluator was created with the correct arguments
        mock_evaluator.assert_called_once()
        
        # Check that the run_benchmarks method was called
        mock_evaluator_instance.run_benchmarks.assert_called_once()
        
        # Check that the save_results and plot_results methods were called
        mock_evaluator_instance.save_results.assert_called_once()
        mock_evaluator_instance.plot_results.assert_called_once()
        
    @patch('scripts.evaluate_model.DecoderEvaluator')
    @patch('scripts.evaluate_model.load_config')
    @patch('scripts.evaluate_model.LiftedProductCode')
    @patch('scripts.evaluate_model.DepolarizingErrorModel')
    @patch('scripts.evaluate_model.SimulatorIntegration')
    @patch('scripts.evaluate_model.stable_baselines3')
    def test_evaluate_model(self, mock_sb3, mock_simulator, mock_error_model, mock_code, mock_config_loader, mock_evaluator):
        """Test the evaluate_model script."""
        # Mock the config loader
        mock_config_loader.load_config.return_value = self.valid_config
        
        # Mock the evaluator
        mock_evaluator_instance = mock_evaluator.return_value
        mock_evaluator_instance.evaluate_rl_model.return_value = {
            'avg_reward': 0.8,
            'success_rate': 0.9,
            'avg_steps': 10.5,
            'logical_error_rate': 0.1
        }
        
        # Mock stable-baselines3
        mock_model = MagicMock()
        mock_sb3.PPO.load.return_value = mock_model
        
        # Run the evaluate_model script with command-line arguments
        with patch('sys.argv', ['evaluate_model.py', '--config', self.config_path, '--model-path', self.model_path]):
            evaluate_model.main()
            
        # Check that the config loader was called with the correct path
        mock_config_loader.load_config.assert_called_once_with(self.config_path)
        
        # Check that the evaluator was created with the correct arguments
        mock_evaluator.assert_called_once()
        
        # Check that the evaluate_rl_model method was called with the correct arguments
        mock_evaluator_instance.evaluate_rl_model.assert_called()
        
        # Check that the model was loaded
        mock_sb3.PPO.load.assert_called_once_with(self.model_path)


if __name__ == '__main__':
    unittest.main()
