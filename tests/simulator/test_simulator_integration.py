"""
Unit tests for the simulator integration module.
"""

import unittest
import numpy as np
from unittest.mock import MagicMock, patch

from ldpc_decoder.simulator.simulator_integration import SimulatorIntegration
from ldpc_decoder.env.decoder_env import DecoderEnv
from ldpc_decoder.codes.base_code import LDPCCode
from ldpc_decoder.error_models.base_error_model import ErrorModel


class TestSimulatorIntegration(unittest.TestCase):
    """Test cases for the SimulatorIntegration class."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock the code, error model, and simulator
        self.mock_code = MagicMock(spec=LDPCCode)
        self.mock_code.n_bits = 10
        self.mock_code.n_checks = 5
        self.mock_code.check_syndrome.return_value = np.zeros(5, dtype=np.int8)
        self.mock_code.get_stabilizer_generators.return_value = np.ones((5, 20), dtype=np.int8)
        
        self.mock_error_model = MagicMock(spec=ErrorModel)
        
        # Create a simulator configuration
        self.sim_config = {
            'shots': 100,
            'seed': 42,
            'circuit_type': 'stabilizer',
            'noise_model': {
                'depolarizing': True,
                'measurement': True
            }
        }
        
        # Create a simulator integration instance with mocks
        with patch('ldpc_decoder.simulator.simulator_integration.StimSimulator') as mock_stim_simulator:
            self.mock_simulator = mock_stim_simulator.return_value
            self.mock_simulator.generate_single_error_syndrome.return_value = (
                np.ones(20, dtype=np.int8),  # Error pattern
                np.ones(5, dtype=np.int8)    # Syndrome
            )
            self.mock_simulator.generate_error_syndrome.return_value = (
                np.ones((100, 20), dtype=np.int8),  # Error patterns
                np.ones((100, 5), dtype=np.int8)    # Syndromes
            )
            self.mock_simulator._is_logical_operator.return_value = False
            
            self.simulator_integration = SimulatorIntegration(
                self.sim_config, 
                self.mock_code, 
                self.mock_error_model
            )
            
    def test_initialization(self):
        """Test simulator integration initialization."""
        self.assertEqual(self.simulator_integration.config, self.sim_config)
        self.assertEqual(self.simulator_integration.code, self.mock_code)
        self.assertEqual(self.simulator_integration.error_model, self.mock_error_model)
        
    def test_generate_syndrome(self):
        """Test syndrome generation."""
        error, syndrome = self.simulator_integration.generate_syndrome()
        
        # Check that the simulator's generate_single_error_syndrome method was called
        self.mock_simulator.generate_single_error_syndrome.assert_called_once_with(
            self.mock_error_model
        )
        
        # Check the returned values
        self.assertEqual(error.shape, (20,))
        self.assertEqual(syndrome.shape, (5,))
        
    def test_generate_batch_syndromes(self):
        """Test batch syndrome generation."""
        batch_size = 50
        errors, syndromes = self.simulator_integration.generate_batch_syndromes(batch_size)
        
        # Check that shots was temporarily set to batch_size
        self.assertEqual(self.mock_simulator.shots, self.sim_config['shots'])
        
        # Check that the simulator's generate_error_syndrome method was called
        self.mock_simulator.generate_error_syndrome.assert_called_once_with(
            self.mock_error_model
        )
        
        # Check the returned values
        self.assertEqual(errors.shape, (100, 20))
        self.assertEqual(syndromes.shape, (100, 5))
        
    def test_integrate_with_env(self):
        """Test integration with the decoder environment."""
        # Create a mock environment
        mock_env = MagicMock(spec=DecoderEnv)
        mock_env.n_bits = 10
        mock_env.n_checks = 5
        
        # Integrate the simulator with the environment
        self.simulator_integration.integrate_with_env(mock_env)
        
        # Check that the environment's reset method was replaced
        self.assertNotEqual(mock_env.reset, DecoderEnv.reset)
        
    def test_evaluate_decoder(self):
        """Test decoder evaluation."""
        # Create a mock decoder function
        mock_decoder_fn = MagicMock(return_value=np.zeros(20, dtype=np.int8))
        
        # Evaluate the decoder
        n_samples = 100
        metrics = self.simulator_integration.evaluate_decoder(mock_decoder_fn, n_samples)
        
        # Check that the decoder function was called
        self.assertEqual(mock_decoder_fn.call_count, n_samples)
        
        # Check that the metrics dictionary contains the expected keys
        expected_keys = ['success_rate', 'logical_error_rate', 'avg_syndrome_weight', 'avg_error_weight']
        for key in expected_keys:
            self.assertIn(key, metrics)
            
    def test_check_syndrome(self):
        """Test syndrome checking."""
        # Create a test error pattern
        error = np.ones(20, dtype=np.int8)
        
        # Check the syndrome
        result = self.simulator_integration._check_syndrome(error)
        
        # Check that the code's check_syndrome method was called
        self.mock_code.check_syndrome.assert_called_once_with(error)
        
        # Check the result
        self.assertTrue(result)


if __name__ == '__main__':
    unittest.main()
