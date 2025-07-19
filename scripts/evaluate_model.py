#!/usr/bin/env python3
"""
Evaluation script for trained RL models.

This script evaluates a trained RL model on the LDPC decoding task
and provides detailed performance metrics.
"""

import os
import sys
import argparse
import logging
import yaml
import numpy as np
import matplotlib.pyplot as plt
import time
from pathlib import Path
import json
from stable_baselines3 import PPO

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ldpc_decoder.config.config_loader import Config, load_config
from ldpc_decoder.codes.lifted_product_code import LiftedProductCode
from ldpc_decoder.codes.balanced_product_code import BalancedProductCode
from ldpc_decoder.error_models.depolarizing_error import DepolarizingErrorModel
from ldpc_decoder.error_models.measurement_error import CombinedMeasurementErrorModel
from ldpc_decoder.error_models.correlated_error import SpatiallyCorrelatedErrorModel, TemporallyCorrelatedErrorModel
from ldpc_decoder.error_models.hardware_error import HardwareInspiredErrorModel
from ldpc_decoder.env.decoder_env import DecoderEnv
from ldpc_decoder.gnn.gnn_model import GNNPolicy
from ldpc_decoder.simulator.stim_simulator import StimSimulator
from ldpc_decoder.simulator.simulator_integration import SimulatorIntegration

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("model_evaluation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate trained RL model')
    
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--model-path', type=str, default='models/final_model',
                        help='Path to trained model')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Directory to save results')
    parser.add_argument('--n-episodes', type=int, default=1000,
                        help='Number of episodes to evaluate')
    parser.add_argument('--error-rates', type=float, nargs='+', 
                        default=[0.01, 0.05, 0.1, 0.15, 0.2],
                        help='Error rates to evaluate')
    parser.add_argument('--render', action='store_true',
                        help='Render environment during evaluation')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed information during evaluation')
    
    return parser.parse_args()

def evaluate_model(model, env, n_episodes, error_rate, render=False, verbose=False):
    """
    Evaluate a trained model.
    
    Args:
        model: Trained model.
        env: Environment.
        n_episodes: Number of episodes to evaluate.
        error_rate: Error rate to evaluate at.
        render: Whether to render the environment.
        verbose: Whether to print detailed information.
        
    Returns:
        Dictionary of evaluation metrics.
    """
    # Set error rate
    env.error_model.error_rate = error_rate
    
    # Evaluation metrics
    metrics = {
        'success_rate': 0.0,
        'logical_error_rate': 0.0,
        'avg_steps': 0.0,
        'avg_syndrome_weight': 0.0,
        'avg_time_per_episode': 0.0,
        'avg_reward': 0.0,
        'error_rate': error_rate
    }
    
    # Episode statistics
    episode_steps = []
    episode_rewards = []
    episode_successes = []
    episode_logical_errors = []
    episode_times = []
    episode_syndrome_weights = []
    
    # Run evaluation episodes
    for episode in range(n_episodes):
        start_time = time.time()
        
        obs, _ = env.reset()
        done = False
        steps = 0
        total_reward = 0
        syndrome_weights = []
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
            total_reward += reward
            
            syndrome_weights.append(info.get('syndrome_weight', 0))
            
            if render:
                env.render()
                
            if verbose and steps % 10 == 0:
                print(f"Episode {episode+1}/{n_episodes}, Step {steps}, Reward {reward:.4f}, "
                      f"Syndrome Weight {info.get('syndrome_weight', 0)}")
                
            if terminated:
                success = info.get('syndrome_resolved', False)
                logical_error = info.get('logical_error', False)
                
                if success:
                    metrics['success_rate'] += 1.0 / n_episodes
                    episode_successes.append(1)
                    
                    if logical_error:
                        metrics['logical_error_rate'] += 1.0 / n_episodes
                        episode_logical_errors.append(1)
                    else:
                        episode_logical_errors.append(0)
                else:
                    episode_successes.append(0)
                    episode_logical_errors.append(0)
                
                if verbose:
                    print(f"Episode {episode+1}/{n_episodes} finished in {steps} steps, "
                          f"Success: {success}, Logical Error: {logical_error}")
                    
        episode_time = time.time() - start_time
        
        # Update metrics
        metrics['avg_steps'] += steps / n_episodes
        metrics['avg_reward'] += total_reward / n_episodes
        metrics['avg_time_per_episode'] += episode_time / n_episodes
        
        if syndrome_weights:
            avg_syndrome_weight = sum(syndrome_weights) / len(syndrome_weights)
            metrics['avg_syndrome_weight'] += avg_syndrome_weight / n_episodes
            episode_syndrome_weights.append(avg_syndrome_weight)
            
        # Store episode statistics
        episode_steps.append(steps)
        episode_rewards.append(total_reward)
        episode_times.append(episode_time)
        
        if (episode + 1) % 10 == 0:
            logger.info(f"Evaluated {episode + 1}/{n_episodes} episodes, "
                       f"Success Rate: {metrics['success_rate'] * n_episodes / (episode + 1):.4f}")
            
    # Additional metrics
    metrics['std_steps'] = np.std(episode_steps)
    metrics['std_reward'] = np.std(episode_rewards)
    metrics['std_time'] = np.std(episode_times)
    metrics['std_syndrome_weight'] = np.std(episode_syndrome_weights)
    metrics['min_steps'] = min(episode_steps)
    metrics['max_steps'] = max(episode_steps)
    metrics['n_episodes'] = n_episodes
    
    return metrics, episode_successes, episode_logical_errors, episode_steps, episode_rewards

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    # Get the configuration dictionary
    config_dict = config.to_dict()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize components
    logger.info("Initializing components...")
    
    # Create code
    code_config = config_dict['code']
    code_type = code_config.get('type', 'lifted_product')
    
    if code_type == 'lifted_product':
        code = LiftedProductCode(code_config)
    elif code_type == 'balanced_product':
        code = BalancedProductCode(code_config)
    else:
        raise ValueError(f"Unsupported code type: {code_type}")
        
    # Create error model
    error_model_config = config_dict['error_model']
    error_model_type = error_model_config.get('type', 'depolarizing')
    
    if error_model_type == 'depolarizing':
        error_model = DepolarizingErrorModel(error_model_config)
    elif error_model_type == 'measurement':
        error_model = CombinedMeasurementErrorModel(error_model_config)
    elif error_model_type == 'spatially_correlated':
        error_model = SpatiallyCorrelatedErrorModel(error_model_config)
    elif error_model_type == 'temporally_correlated':
        error_model = TemporallyCorrelatedErrorModel(error_model_config)
    elif error_model_type == 'hardware_inspired':
        error_model = HardwareInspiredErrorModel(error_model_config)
    else:
        raise ValueError(f"Unsupported error model type: {error_model_type}")
        
    # Create environment
    env_config = config_dict['environment']
    env = DecoderEnv(env_config, code, error_model)
    
    # Initialize simulator integration
    simulator_config = config.config.get('simulator', {})
    simulator_integration = SimulatorIntegration(simulator_config, code, error_model)
    simulator_integration.integrate_with_env(env)
    
    # Load model
    logger.info(f"Loading model from {args.model_path}...")
    model = PPO.load(args.model_path, env=env)
    
    # Evaluate model at different error rates
    all_metrics = []
    all_episode_successes = []
    all_episode_logical_errors = []
    all_episode_steps = []
    all_episode_rewards = []
    
    for error_rate in args.error_rates:
        logger.info(f"Evaluating model at error rate {error_rate}...")
        
        metrics, episode_successes, episode_logical_errors, episode_steps, episode_rewards = evaluate_model(
            model, env, args.n_episodes, error_rate, args.render, args.verbose
        )
        
        all_metrics.append(metrics)
        all_episode_successes.append(episode_successes)
        all_episode_logical_errors.append(episode_logical_errors)
        all_episode_steps.append(episode_steps)
        all_episode_rewards.append(episode_rewards)
        
        # Print results
        print(f"\nResults for error rate {error_rate}:")
        print(f"Success Rate: {metrics['success_rate']:.4f}")
        print(f"Logical Error Rate: {metrics['logical_error_rate']:.4f}")
        print(f"Average Steps: {metrics['avg_steps']:.2f} ± {metrics['std_steps']:.2f}")
        print(f"Average Reward: {metrics['avg_reward']:.4f} ± {metrics['std_reward']:.4f}")
        print(f"Average Syndrome Weight: {metrics['avg_syndrome_weight']:.4f} ± {metrics['std_syndrome_weight']:.4f}")
        print(f"Average Time per Episode: {metrics['avg_time_per_episode']:.4f} s")
        
    # Save results
    results = {
        'metrics': all_metrics,
        'error_rates': args.error_rates,
        'n_episodes': args.n_episodes,
        'model_path': args.model_path,
        'code_type': code_type,
        'code_size': code.n_bits,
        'error_model_type': error_model_type
    }
    
    results_path = output_dir / 'model_evaluation_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
        
    # Generate plots
    generate_plots(results, output_dir)
    
    logger.info(f"Evaluation completed. Results saved to {results_path}")

def generate_plots(results, output_dir):
    """
    Generate evaluation plots.
    
    Args:
        results: Evaluation results.
        output_dir: Output directory.
    """
    # Extract data
    error_rates = results['error_rates']
    metrics = results['metrics']
    
    # Plot success rate vs. error rate
    plt.figure(figsize=(10, 6))
    success_rates = [m['success_rate'] for m in metrics]
    plt.plot(error_rates, success_rates, marker='o', label='Success Rate')
    
    logical_error_rates = [m['logical_error_rate'] for m in metrics]
    plt.plot(error_rates, logical_error_rates, marker='s', label='Logical Error Rate')
    
    plt.xlabel('Physical Error Rate')
    plt.ylabel('Rate')
    plt.title('Success and Logical Error Rates vs. Physical Error Rate')
    plt.grid(True)
    plt.legend()
    
    plot_path = output_dir / 'model_success_rates.png'
    plt.savefig(plot_path)
    
    # Plot average steps vs. error rate
    plt.figure(figsize=(10, 6))
    avg_steps = [m['avg_steps'] for m in metrics]
    std_steps = [m['std_steps'] for m in metrics]
    
    plt.errorbar(error_rates, avg_steps, yerr=std_steps, marker='o')
    plt.xlabel('Physical Error Rate')
    plt.ylabel('Average Steps')
    plt.title('Average Steps vs. Physical Error Rate')
    plt.grid(True)
    
    plot_path = output_dir / 'model_avg_steps.png'
    plt.savefig(plot_path)
    
    # Plot average reward vs. error rate
    plt.figure(figsize=(10, 6))
    avg_rewards = [m['avg_reward'] for m in metrics]
    std_rewards = [m['std_reward'] for m in metrics]
    
    plt.errorbar(error_rates, avg_rewards, yerr=std_rewards, marker='o')
    plt.xlabel('Physical Error Rate')
    plt.ylabel('Average Reward')
    plt.title('Average Reward vs. Physical Error Rate')
    plt.grid(True)
    
    plot_path = output_dir / 'model_avg_reward.png'
    plt.savefig(plot_path)
    
    # Plot average syndrome weight vs. error rate
    plt.figure(figsize=(10, 6))
    avg_syndrome_weights = [m['avg_syndrome_weight'] for m in metrics]
    std_syndrome_weights = [m['std_syndrome_weight'] for m in metrics]
    
    plt.errorbar(error_rates, avg_syndrome_weights, yerr=std_syndrome_weights, marker='o')
    plt.xlabel('Physical Error Rate')
    plt.ylabel('Average Syndrome Weight')
    plt.title('Average Syndrome Weight vs. Physical Error Rate')
    plt.grid(True)
    
    plot_path = output_dir / 'model_avg_syndrome_weight.png'
    plt.savefig(plot_path)
    
    # Plot threshold estimation
    plt.figure(figsize=(10, 6))
    plt.plot(error_rates, success_rates, marker='o', label='Success Rate')
    plt.plot(error_rates, [1 - rate for rate in logical_error_rates], marker='s', label='1 - Logical Error Rate')
    
    # Find intersection point (threshold estimate)
    try:
        from scipy.interpolate import interp1d
        from scipy.optimize import fsolve
        
        f_success = interp1d(error_rates, success_rates, kind='cubic')
        f_logical = interp1d(error_rates, [1 - rate for rate in logical_error_rates], kind='cubic')
        
        def func(x):
            return f_success(x) - f_logical(x)
            
        threshold = fsolve(func, 0.1)[0]
        
        plt.axvline(x=threshold, color='r', linestyle='--', label=f'Threshold ≈ {threshold:.4f}')
    except:
        logger.warning("Could not estimate threshold")
    
    plt.xlabel('Physical Error Rate')
    plt.ylabel('Rate')
    plt.title('Threshold Estimation')
    plt.grid(True)
    plt.legend()
    
    plot_path = output_dir / 'model_threshold_estimation.png'
    plt.savefig(plot_path)

if __name__ == "__main__":
    main()
