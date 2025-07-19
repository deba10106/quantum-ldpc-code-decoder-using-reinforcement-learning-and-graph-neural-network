"""
Integration tests for simulator and environment components.
"""

import unittest
import numpy as np
import sys
import os
from unittest.mock import patch, MagicMock

# Add the mocks directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../mocks')))

# Import our mock gymnasium module
from gymnasium_mock import gymnasium as gym

from ldpc_decoder.codes.lifted_product_code import LiftedProductCode
from ldpc_decoder.error_models.depolarizing_error import DepolarizingErrorModel
from ldpc_decoder.env.decoder_env import DecoderEnv
from ldpc_decoder.simulator.simulator_integration import SimulatorIntegration


class TestSimulatorEnvironmentIntegration(unittest.TestCase):
    """Test the integration between the simulator and environment components."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a small code for testing
        self.code_config = {
            'type': 'lifted_product',
            'parameters': {
                'n_checks': 8,
                'n_bits': 16,
                'distance': 4,
                'lifting_parameter': 4,
                'base_matrix_rows': 2,
                'base_matrix_cols': 4
            }
        }
        self.code = LiftedProductCode(self.code_config['parameters'])
        # Generate the code to avoid "Code not generated yet" errors
        self.code.generate_code()
        
        # Create a simple error model
        self.error_model_config = {
            'primary_type': 'depolarizing',
            'error_rate': 0.05
        }
        self.error_model = DepolarizingErrorModel(self.error_model_config)
        
        # Create environment configuration
        self.env_config = {
            'max_steps': 20,
            'observation_type': 'syndrome_graph',
            'reward_normalization': {
                'enabled': True,
                'scale': 1.0
            }
        }
        
        # Create simulator configuration
        self.sim_config = {
            'shots': 10,
            'seed': 42
        }
        
    @patch('ldpc_decoder.simulator.stim_simulator.stim')
    def test_simulator_env_integration(self, mock_stim):
        """Test that the simulator can be integrated with the environment."""
        # Mock stim to avoid actual circuit simulation
        mock_circuit = MagicMock()
        mock_stim.Circuit.return_value = mock_circuit
        mock_circuit.compile_sampler.return_value = MagicMock()
        mock_circuit.compile_sampler().sample.return_value = np.zeros((10, 16), dtype=np.uint8)
        mock_circuit.compile_detector_sampler.return_value = MagicMock()
        mock_circuit.compile_detector_sampler().sample.return_value = np.zeros((10, 8), dtype=np.uint8)
        
        # Create the environment
        env = DecoderEnv(self.env_config, self.code, self.error_model)
        
        # Create the simulator integration
        simulator = SimulatorIntegration(self.sim_config, self.code, self.error_model)
        
        # Integrate the simulator with the environment
        simulator.integrate_with_env(env)
        
        # Check that the environment's reset method has been replaced
        self.assertNotEqual(env.reset.__name__, DecoderEnv.reset.__name__)
        
        # Reset the environment and check the returned values
        observation, info = env.reset()
        
        # Check that the observation has the expected structure
        self.assertIsNotNone(observation)
        
        # Check that the info dictionary contains the expected keys
        expected_keys = ['syndrome_weight', 'true_error_weight', 'estimated_error_weight', 
                         'resolved_checks', 'total_checks']
        for key in expected_keys:
            self.assertIn(key, info)
            
        # Take a step in the environment
        action = 0  # Flip the first qubit
        next_observation, reward, terminated, truncated, next_info = env.step(action)
        
        # Check that the step returns the expected values
        self.assertIsNotNone(next_observation)
        self.assertIsInstance(reward, float)
        # Convert numpy bool to Python bool if needed
        terminated = bool(terminated)
        truncated = bool(truncated)
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)
        self.assertIsInstance(next_info, dict)
        
    @patch('ldpc_decoder.simulator.stim_simulator.stim')
    def test_batch_syndrome_generation(self, mock_stim):
        """Test batch syndrome generation through the simulator."""
        # Mock stim to avoid actual circuit simulation
        mock_circuit = MagicMock()
        mock_stim.Circuit.return_value = mock_circuit
        mock_circuit.compile_sampler.return_value = MagicMock()
        mock_circuit.compile_sampler().sample.return_value = np.zeros((10, 16), dtype=np.uint8)
        mock_circuit.compile_detector_sampler.return_value = MagicMock()
        mock_circuit.compile_detector_sampler().sample.return_value = np.zeros((10, 8), dtype=np.uint8)
        
        # Create the simulator integration
        simulator = SimulatorIntegration(self.sim_config, self.code, self.error_model)
        
        # Generate a batch of syndromes
        batch_size = 5
        error_patterns, syndromes = simulator.generate_batch_syndromes(batch_size)
        
        # Check the shapes of the returned arrays
        self.assertEqual(error_patterns.shape[0], batch_size)  # Using batch_size instead of shots
        self.assertEqual(error_patterns.shape[1], 2 * self.code.n_bits)  # X and Z errors
        self.assertEqual(syndromes.shape[0], batch_size)  # Using batch_size instead of shots
        self.assertEqual(syndromes.shape[1], self.code.n_checks)
        
    @patch('ldpc_decoder.simulator.stim_simulator.stim')
    def test_decoder_evaluation(self, mock_stim):
        """Test decoder evaluation through the simulator."""
        # Mock stim to avoid actual circuit simulation
        mock_circuit = MagicMock()
        mock_stim.Circuit.return_value = mock_circuit
        mock_circuit.compile_sampler.return_value = MagicMock()
        mock_circuit.compile_sampler().sample.return_value = np.zeros((10, 16), dtype=np.uint8)
        mock_circuit.compile_detector_sampler.return_value = MagicMock()
        mock_circuit.compile_detector_sampler().sample.return_value = np.zeros((10, 8), dtype=np.uint8)
        
        # Create the simulator integration
        simulator = SimulatorIntegration(self.sim_config, self.code, self.error_model)
        
        # Create a simple decoder function
        def simple_decoder(syndrome):
            # Always return a zero error pattern (not a good decoder, but simple for testing)
            # Return array with length 2 * n_bits for both X and Z errors
            return np.zeros(2 * self.code.n_bits, dtype=np.int8)
        
        # Evaluate the decoder
        n_samples = 5
        metrics = simulator.evaluate_decoder(simple_decoder, n_samples)
        
        # Check that the metrics dictionary contains the expected keys
        expected_keys = ['success_rate', 'logical_error_rate', 'avg_syndrome_weight', 'avg_error_weight']
        for key in expected_keys:
            self.assertIn(key, metrics)


if __name__ == '__main__':
    unittest.main()
