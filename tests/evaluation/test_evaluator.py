"""
Unit tests for the evaluator module.
"""

import unittest
import numpy as np
import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

# Add the mocks directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../mocks')))

# Import real torch instead of using mock
# We need the real torch.nn for stable-baselines3 to work properly
import torch

from ldpc_decoder.evaluation.evaluator import DecoderEvaluator
from ldpc_decoder.codes.base_code import LDPCCode
from ldpc_decoder.error_models.base_error_model import ErrorModel
from ldpc_decoder.simulator.simulator_integration import SimulatorIntegration


class TestDecoderEvaluator(unittest.TestCase):
    """Test cases for the DecoderEvaluator class."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock the code, error model, and simulator integration
        self.mock_code = MagicMock(spec=LDPCCode)
        self.mock_code.n_bits = 10
        self.mock_code.n_checks = 5
        self.mock_code.get_stabilizer_generators.return_value = np.ones((5, 20), dtype=np.int8)
        
        self.mock_error_model = MagicMock(spec=ErrorModel)
        
        self.mock_simulator_integration = MagicMock(spec=SimulatorIntegration)
        self.mock_simulator_integration.evaluate_decoder.return_value = {
            'success_rate': 0.9,
            'logical_error_rate': 0.1,
            'avg_syndrome_weight': 2.5,
            'avg_error_weight': 1.5
        }
        
        # Create an evaluator configuration
        self.eval_config = {
            'n_episodes': 100,
            'error_rates': [0.01, 0.05, 0.1],
            'decoder_types': ['gnn', 'mwpm'],
            'output_dir': 'results',
            'model_path': 'models/final_model',
            'metrics': ['logical_error_rate', 'syndrome_resolution_rate'],
            'visualization': {
                'enabled': True,
                'plot_types': ['error_rates']
            }
        }
        
        # Create a temporary directory for test outputs
        self.temp_dir = tempfile.mkdtemp()
        self.eval_config['output_dir'] = self.temp_dir
        
        # Create an evaluator instance
        self.evaluator = DecoderEvaluator(
            self.eval_config,
            self.mock_code,
            self.mock_error_model,
            self.mock_simulator_integration
        )
        
    def tearDown(self):
        """Clean up after tests."""
        # Remove temporary directory and its contents
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_initialization(self):
        """Test evaluator initialization."""
        self.assertEqual(self.evaluator.config, self.eval_config)
        self.assertEqual(self.evaluator.code, self.mock_code)
        self.assertEqual(self.evaluator.error_model, self.mock_error_model)
        self.assertEqual(self.evaluator.simulator, self.mock_simulator_integration)
        
    def test_evaluate_gnn_decoder(self):
        """Test GNN decoder evaluation."""
        # Mock the GNN decoder function
        mock_gnn_decoder = MagicMock(return_value=np.zeros(20, dtype=np.int8))
        self.evaluator._gnn_decoder = mock_gnn_decoder
        
        # Evaluate the GNN decoder
        error_rate = 0.05
        n_samples = 50
        metrics = self.evaluator.evaluate_gnn_decoder(error_rate, n_samples)
        
        # Check that the simulator's evaluate_decoder method was called
        self.mock_simulator_integration.evaluate_decoder.assert_called_once()
        
        # Check the returned metrics
        self.assertEqual(metrics['success_rate'], 0.9)
        self.assertEqual(metrics['logical_error_rate'], 0.1)
        
    def test_evaluate_mwpm_decoder(self):
        """Test MWPM decoder evaluation."""
        # Mock the MWPM decoder function
        mock_mwpm_decoder = MagicMock(return_value=np.zeros(20, dtype=np.int8))
        self.evaluator._mwpm_decoder = mock_mwpm_decoder
        
        # Evaluate the MWPM decoder
        error_rate = 0.05
        n_samples = 50
        metrics = self.evaluator.evaluate_mwpm_decoder(error_rate, n_samples)
        
        # Check that the simulator's evaluate_decoder method was called
        self.mock_simulator_integration.evaluate_decoder.assert_called_once()
        
        # Check the returned metrics
        self.assertEqual(metrics['success_rate'], 0.9)
        self.assertEqual(metrics['logical_error_rate'], 0.1)
        
    @patch('ldpc_decoder.evaluation.evaluator.stable_baselines3')
    def test_evaluate_rl_model(self, mock_sb3):
        """Test RL model evaluation."""
        # Mock the RL model
        mock_model = MagicMock()
        mock_sb3.PPO.load.return_value = mock_model
        mock_model.predict.return_value = (np.zeros(1), None)
        
        # Mock the environment
        mock_env = MagicMock()
        mock_env.reset.return_value = (np.zeros(10), {})
        mock_env.step.return_value = (np.zeros(10), 1.0, False, False, {})
        
        # Evaluate the RL model
        model_path = 'models/test_model'
        error_rate = 0.05
        n_episodes = 50
        
        with patch('ldpc_decoder.evaluation.evaluator.DecoderEnv') as mock_env_class:
            mock_env_class.return_value = mock_env
            metrics = self.evaluator.evaluate_rl_model(model_path, error_rate, n_episodes)
        
        # Check that the model was loaded
        mock_sb3.PPO.load.assert_called_once_with(model_path)
        
        # Check that the model's predict method was called
        self.assertEqual(mock_model.predict.call_count, n_episodes)
        
        # Check that metrics were returned
        self.assertIn('avg_reward', metrics)
        self.assertIn('success_rate', metrics)
        
    def test_run_benchmarks(self):
        """Test running benchmarks."""
        # Mock the evaluation methods
        self.evaluator.evaluate_gnn_decoder = MagicMock(return_value={
            'success_rate': 0.9,
            'logical_error_rate': 0.1
        })
        self.evaluator.evaluate_mwpm_decoder = MagicMock(return_value={
            'success_rate': 0.8,
            'logical_error_rate': 0.2
        })
        
        # Run benchmarks
        results = self.evaluator.run_benchmarks()
        
        # Check that the evaluation methods were called for each error rate
        self.assertEqual(self.evaluator.evaluate_gnn_decoder.call_count, len(self.eval_config['error_rates']))
        self.assertEqual(self.evaluator.evaluate_mwpm_decoder.call_count, len(self.eval_config['error_rates']))
        
        # Check that results were returned for each decoder type and error rate
        for decoder_type in self.eval_config['decoder_types']:
            self.assertIn(decoder_type, results)
            for error_rate in self.eval_config['error_rates']:
                self.assertIn(str(error_rate), results[decoder_type])
                
    def test_save_results(self):
        """Test saving results."""
        # Create test results
        results = {
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
        
        # Save the results
        output_path = self.evaluator.save_results(results)
        
        # Check that the output file exists
        self.assertTrue(os.path.exists(output_path))
        
        # Check that the file is a JSON file
        self.assertTrue(output_path.endswith('.json'))
        
    @patch('ldpc_decoder.evaluation.evaluator.plt')
    def test_plot_results(self, mock_plt):
        """Test plotting results."""
        # Create test results
        results = {
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
        
        # Plot the results
        output_path = self.evaluator.plot_results(results)
        
        # Check that matplotlib was used
        self.assertTrue(mock_plt.figure.called)
        self.assertTrue(mock_plt.savefig.called)
        
        # Check that the output file exists
        self.assertTrue(os.path.exists(output_path))
        
        # Check that the file is an image file
        self.assertTrue(output_path.endswith('.png'))


if __name__ == '__main__':
    unittest.main()
