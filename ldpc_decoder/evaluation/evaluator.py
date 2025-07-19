"""
Evaluation module for LDPC decoder.

This module provides comprehensive evaluation and benchmarking
functionality for LDPC decoders.
"""

import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple, List, Optional, Union, Callable
import logging
from pathlib import Path
import json
import pandas as pd
from stable_baselines3 import PPO

from ..env.decoder_env import DecoderEnv
from ..gnn.gnn_model import GNNPolicy
from ..simulator.stim_simulator import StimSimulator
from ..simulator.simulator_integration import SimulatorIntegration
from ..codes.base_code import LDPCCode
from ..error_models.base_error_model import ErrorModel

# Set up logging
logger = logging.getLogger(__name__)

class DecoderEvaluator:
    """
    Evaluator for LDPC decoders.
    
    This class provides comprehensive evaluation and benchmarking
    functionality for LDPC decoders.
    """
    
    def __init__(self, config: Dict[str, Any], env: DecoderEnv, code: LDPCCode, error_model: ErrorModel):
        """
        Initialize the decoder evaluator.
        
        Args:
            config: Evaluation configuration.
            env: Decoder environment instance.
            code: LDPC code instance.
            error_model: Error model instance.
        """
        self.config = config
        self.env = env
        self.code = code
        self.error_model = error_model
        
        # Evaluation parameters
        self.n_episodes = config.get('n_episodes', 1000)
        self.error_rates = config.get('error_rates', [0.01, 0.05, 0.1, 0.15, 0.2])
        self.decoder_types = config.get('decoder_types', ['gnn', 'mwpm'])
        self.output_dir = Path(config.get('output_dir', 'results'))
        self.model_path = config.get('model_path', 'models/final_model')
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize simulator integration
        simulator_config = config.get('simulator', {})
        self.simulator_integration = SimulatorIntegration(simulator_config, code, error_model)
        
        logger.info(f"Initialized decoder evaluator with {self.n_episodes} episodes")
        
    def evaluate_decoder(self, decoder_type: str, error_rate: float) -> Dict[str, float]:
        """
        Evaluate a decoder at a specific error rate.
        
        Args:
            decoder_type: Type of decoder to evaluate.
            error_rate: Error rate to evaluate at.
            
        Returns:
            Dictionary of evaluation metrics.
        """
        # Set error rate
        self.error_model.error_rate = error_rate
        
        # Get decoder function
        decoder_fn = self._get_decoder_function(decoder_type)
        
        # Evaluate decoder
        metrics = self.simulator_integration.evaluate_decoder(decoder_fn, self.n_episodes)
        
        # Add additional information
        metrics['error_rate'] = error_rate
        metrics['decoder_type'] = decoder_type
        metrics['n_episodes'] = self.n_episodes
        metrics['code_size'] = self.code.n_bits
        metrics['code_type'] = self.code.__class__.__name__
        
        logger.info(f"Evaluated {decoder_type} decoder at error rate {error_rate}: {metrics}")
        
        return metrics
        
    def run_benchmark(self) -> Dict[str, List[Dict[str, float]]]:
        """
        Run a comprehensive benchmark of decoders across error rates.
        
        Returns:
            Dictionary mapping decoder types to lists of evaluation metrics.
        """
        results = {}
        
        # Evaluate each decoder type
        for decoder_type in self.decoder_types:
            results[decoder_type] = []
            
            # Evaluate across error rates
            for error_rate in self.error_rates:
                metrics = self.evaluate_decoder(decoder_type, error_rate)
                results[decoder_type].append(metrics)
                
        # Save results
        self._save_results(results)
        
        # Generate plots
        self._generate_plots(results)
        
        return results
        
    def evaluate_model(self, model_path: Optional[str] = None) -> Dict[str, float]:
        """
        Evaluate a trained RL model.
        
        Args:
            model_path: Path to the model.
            
        Returns:
            Dictionary of evaluation metrics.
        """
        # Load model
        model_path = model_path or self.model_path
        model = PPO.load(model_path, env=self.env)
        
        # Evaluation metrics
        metrics = {
            'success_rate': 0.0,
            'logical_error_rate': 0.0,
            'avg_steps': 0.0,
            'avg_syndrome_weight': 0.0,
            'avg_time_per_episode': 0.0
        }
        
        # Run evaluation episodes
        total_time = 0.0
        for _ in range(self.n_episodes):
            start_time = time.time()
            
            obs, _ = self.env.reset()
            done = False
            steps = 0
            syndrome_weights = []
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                steps += 1
                
                syndrome_weights.append(info.get('syndrome_weight', 0))
                
                if terminated:
                    if info.get('syndrome_resolved', False):
                        metrics['success_rate'] += 1.0 / self.n_episodes
                        
                        if info.get('logical_error', False):
                            metrics['logical_error_rate'] += 1.0 / self.n_episodes
                            
            episode_time = time.time() - start_time
            total_time += episode_time
            
            metrics['avg_steps'] += steps / self.n_episodes
            metrics['avg_syndrome_weight'] += sum(syndrome_weights) / len(syndrome_weights) / self.n_episodes
            
        metrics['avg_time_per_episode'] = total_time / self.n_episodes
        
        # Add additional information
        metrics['error_rate'] = self.error_model.error_rate
        metrics['decoder_type'] = 'rl_model'
        metrics['n_episodes'] = self.n_episodes
        metrics['code_size'] = self.code.n_bits
        metrics['code_type'] = self.code.__class__.__name__
        
        logger.info(f"Evaluated RL model: {metrics}")
        
        return metrics
        
    def compare_decoders(self, error_rate: float) -> Dict[str, Dict[str, float]]:
        """
        Compare different decoders at a specific error rate.
        
        Args:
            error_rate: Error rate to evaluate at.
            
        Returns:
            Dictionary mapping decoder types to evaluation metrics.
        """
        results = {}
        
        # Set error rate
        self.error_model.error_rate = error_rate
        
        # Evaluate each decoder type
        for decoder_type in self.decoder_types:
            metrics = self.evaluate_decoder(decoder_type, error_rate)
            results[decoder_type] = metrics
            
        # Evaluate RL model if available
        if os.path.exists(self.model_path):
            metrics = self.evaluate_model()
            results['rl_model'] = metrics
            
        # Generate comparison plot
        self._generate_comparison_plot(results, error_rate)
        
        return results
        
    def _get_decoder_function(self, decoder_type: str) -> Callable:
        """
        Get a decoder function based on the decoder type.
        
        Args:
            decoder_type: Type of decoder.
            
        Returns:
            Decoder function.
        """
        if decoder_type == 'gnn':
            # Load GNN policy
            gnn_config = self.config.get('gnn', {})
            gnn_policy = GNNPolicy(gnn_config, self.code.n_bits, self.code.n_checks)
            
            # Create decoder function
            def gnn_decoder(syndrome: np.ndarray) -> np.ndarray:
                # Convert syndrome to observation
                observation = {
                    'syndrome': syndrome,
                    'estimated_error': np.zeros(2 * self.code.n_bits, dtype=np.int8),
                    'step': np.array([0], dtype=np.int32)
                }
                
                # Initialize estimated error
                estimated_error = np.zeros(2 * self.code.n_bits, dtype=np.int8)
                
                # Decode step by step
                max_steps = self.env.max_steps
                for step in range(max_steps):
                    # Update observation
                    observation['step'] = np.array([step], dtype=np.int32)
                    observation['estimated_error'] = estimated_error
                    
                    # Predict action
                    action, _ = gnn_policy.predict(observation)
                    
                    # Apply action
                    qubit_idx, flip_type = divmod(action, 4)
                    
                    if flip_type == 1:  # X flip
                        estimated_error[qubit_idx] ^= 1
                    elif flip_type == 2:  # Z flip
                        estimated_error[self.code.n_bits + qubit_idx] ^= 1
                    elif flip_type == 3:  # Y flip
                        estimated_error[qubit_idx] ^= 1
                        estimated_error[self.code.n_bits + qubit_idx] ^= 1
                        
                    # Calculate syndrome
                    new_syndrome = self._calculate_syndrome(estimated_error)
                    
                    # Check if syndrome is resolved
                    if np.all(new_syndrome == 0):
                        break
                        
                    # Update syndrome
                    observation['syndrome'] = new_syndrome
                    
                return estimated_error
                
            return gnn_decoder
            
        elif decoder_type == 'mwpm':
            # Minimum-weight perfect matching decoder
            # This is a placeholder implementation
            def mwpm_decoder(syndrome: np.ndarray) -> np.ndarray:
                # In a real implementation, this would use a MWPM algorithm
                # For now, we'll just return a random error pattern
                estimated_error = np.zeros(2 * self.code.n_bits, dtype=np.int8)
                
                # Randomly flip qubits until the syndrome is resolved
                max_attempts = 100
                for _ in range(max_attempts):
                    # Choose a random qubit and error type
                    qubit_idx = np.random.randint(0, self.code.n_bits)
                    error_type = np.random.randint(1, 4)
                    
                    if error_type == 1:  # X flip
                        estimated_error[qubit_idx] ^= 1
                    elif error_type == 2:  # Z flip
                        estimated_error[self.code.n_bits + qubit_idx] ^= 1
                    elif error_type == 3:  # Y flip
                        estimated_error[qubit_idx] ^= 1
                        estimated_error[self.code.n_bits + qubit_idx] ^= 1
                        
                    # Calculate syndrome
                    new_syndrome = self._calculate_syndrome(estimated_error)
                    
                    # Check if syndrome is resolved
                    if np.all(new_syndrome == 0):
                        break
                        
                return estimated_error
                
            return mwpm_decoder
            
        else:
            raise ValueError(f"Unsupported decoder type: {decoder_type}")
            
    def _calculate_syndrome(self, error: np.ndarray) -> np.ndarray:
        """
        Calculate the syndrome for a given error pattern.
        
        Args:
            error: Error pattern.
            
        Returns:
            Syndrome.
        """
        # Extract X and Z parts
        x_part = error[:self.code.n_bits]
        z_part = error[self.code.n_bits:]
        
        # Calculate syndrome for X and Z parts separately
        x_syndrome = self.code.check_syndrome(x_part)
        z_syndrome = self.code.check_syndrome(z_part)
        
        # Combine syndromes
        return np.concatenate([x_syndrome, z_syndrome])
        
    def _save_results(self, results: Dict[str, List[Dict[str, float]]]) -> None:
        """
        Save evaluation results to files.
        
        Args:
            results: Evaluation results.
        """
        # Save as JSON
        json_path = self.output_dir / 'evaluation_results.json'
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
            
        # Save as CSV
        csv_data = []
        for decoder_type, metrics_list in results.items():
            for metrics in metrics_list:
                row = {'decoder_type': decoder_type}
                row.update(metrics)
                csv_data.append(row)
                
        df = pd.DataFrame(csv_data)
        csv_path = self.output_dir / 'evaluation_results.csv'
        df.to_csv(csv_path, index=False)
        
        logger.info(f"Saved evaluation results to {json_path} and {csv_path}")
        
    def _generate_plots(self, results: Dict[str, List[Dict[str, float]]]) -> None:
        """
        Generate plots from evaluation results.
        
        Args:
            results: Evaluation results.
        """
        # Plot logical error rate vs. physical error rate
        plt.figure(figsize=(10, 6))
        
        for decoder_type, metrics_list in results.items():
            error_rates = [metrics['error_rate'] for metrics in metrics_list]
            logical_error_rates = [metrics['logical_error_rate'] for metrics in metrics_list]
            
            plt.plot(error_rates, logical_error_rates, marker='o', label=decoder_type)
            
        plt.xlabel('Physical Error Rate')
        plt.ylabel('Logical Error Rate')
        plt.title('Logical Error Rate vs. Physical Error Rate')
        plt.grid(True)
        plt.legend()
        
        plot_path = self.output_dir / 'logical_error_rate.png'
        plt.savefig(plot_path)
        
        # Plot success rate vs. physical error rate
        plt.figure(figsize=(10, 6))
        
        for decoder_type, metrics_list in results.items():
            error_rates = [metrics['error_rate'] for metrics in metrics_list]
            success_rates = [metrics['success_rate'] for metrics in metrics_list]
            
            plt.plot(error_rates, success_rates, marker='o', label=decoder_type)
            
        plt.xlabel('Physical Error Rate')
        plt.ylabel('Success Rate')
        plt.title('Success Rate vs. Physical Error Rate')
        plt.grid(True)
        plt.legend()
        
        plot_path = self.output_dir / 'success_rate.png'
        plt.savefig(plot_path)
        
        logger.info(f"Generated evaluation plots in {self.output_dir}")
        
    def _generate_comparison_plot(self, results: Dict[str, Dict[str, float]], error_rate: float) -> None:
        """
        Generate a comparison plot for different decoders.
        
        Args:
            results: Evaluation results.
            error_rate: Error rate used for comparison.
        """
        # Extract metrics
        decoder_types = list(results.keys())
        success_rates = [results[dt]['success_rate'] for dt in decoder_types]
        logical_error_rates = [results[dt]['logical_error_rate'] for dt in decoder_types]
        
        # Plot success rate and logical error rate
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(decoder_types))
        width = 0.35
        
        ax1.bar(x - width/2, success_rates, width, label='Success Rate', color='blue')
        ax1.set_ylabel('Success Rate', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        
        ax2 = ax1.twinx()
        ax2.bar(x + width/2, logical_error_rates, width, label='Logical Error Rate', color='red')
        ax2.set_ylabel('Logical Error Rate', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        ax1.set_xticks(x)
        ax1.set_xticklabels(decoder_types)
        ax1.set_xlabel('Decoder Type')
        
        plt.title(f'Decoder Comparison at Error Rate {error_rate}')
        
        # Add legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plot_path = self.output_dir / f'decoder_comparison_{error_rate}.png'
        plt.savefig(plot_path)
        
        logger.info(f"Generated decoder comparison plot at {plot_path}")
