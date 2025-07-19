#!/usr/bin/env python3
"""
Training script for LDPC decoder RL model.

This script trains a reinforcement learning model for LDPC decoding
using the PPO algorithm and GNN policy.
"""

import os
import sys
import argparse
import logging
import yaml
import numpy as np
from pathlib import Path
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ldpc_decoder.config.config_loader import load_config
from ldpc_decoder.codes.lifted_product_code import LiftedProductCode
from ldpc_decoder.codes.balanced_product_code import BalancedProductCode
from ldpc_decoder.error_models.depolarizing_error import DepolarizingErrorModel
from ldpc_decoder.error_models.measurement_error import CombinedMeasurementErrorModel
from ldpc_decoder.error_models.correlated_error import SpatiallyCorrelatedErrorModel, TemporallyCorrelatedErrorModel
from ldpc_decoder.error_models.hardware_error import HardwareInspiredErrorModel
from ldpc_decoder.env.decoder_env import DecoderEnv
from ldpc_decoder.rl.ppo_trainer import LDPCDecoderTrainer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train LDPC decoder RL model')
    
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--output-dir', type=str, default='models',
                        help='Directory to save models')
    parser.add_argument('--log-dir', type=str, default='logs',
                        help='Directory to save logs')
    parser.add_argument('--total-timesteps', type=int, default=1000000,
                        help='Total timesteps for training')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--n-envs', type=int, default=4,
                        help='Number of parallel environments')
    parser.add_argument('--eval-freq', type=int, default=10000,
                        help='Evaluation frequency')
    parser.add_argument('--save-freq', type=int, default=50000,
                        help='Model saving frequency')
    parser.add_argument('--continue-training', action='store_true',
                        help='Continue training from a checkpoint')
    parser.add_argument('--checkpoint-path', type=str, default=None,
                        help='Path to checkpoint for continued training')
    
    return parser.parse_args()

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Get the raw config dictionary
    config_dict = config.to_dict()
    
    # Update configuration with command line arguments
    if 'training' not in config_dict:
        config_dict['training'] = {}
    
    config_dict['training']['output_dir'] = args.output_dir
    config_dict['training']['log_dir'] = args.log_dir
    config_dict['training']['total_timesteps'] = args.total_timesteps
    config_dict['training']['seed'] = args.seed
    config_dict['training']['n_envs'] = args.n_envs
    config_dict['training']['eval_freq'] = args.eval_freq
    config_dict['training']['save_freq'] = args.save_freq
    
    # Create output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize components
    logger.info("Initializing components...")
    
    # Create code
    code_config = config['code']
    code_type = code_config.get('type', 'lifted_product')
    
    if code_type == 'lifted_product':
        code = LiftedProductCode(code_config)
    elif code_type == 'balanced_product':
        code = BalancedProductCode(code_config)
    else:
        raise ValueError(f"Unsupported code type: {code_type}")
        
    # Generate the code structure
    code.generate_code()
        
    # Create error model
    error_model_config = config['error_model']
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
    env_config = config['environment']
    env = DecoderEnv(env_config, code, error_model)
    
    # Create trainer
    trainer_config = config['training']
    trainer = LDPCDecoderTrainer(trainer_config, env)
    
    # Continue training from checkpoint if specified
    if args.continue_training and args.checkpoint_path:
        logger.info(f"Loading checkpoint from {args.checkpoint_path}")
        trainer.load(args.checkpoint_path)
    
    # Start training
    start_time = time.time()
    logger.info("Starting training...")
    
    trainer.train()
    
    # Log training time
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.2f} seconds")
    
    # Save final model
    final_model_path = output_dir / 'final_model'
    trainer.save(str(final_model_path))
    logger.info(f"Final model saved to {final_model_path}")
    
    # Run evaluation
    logger.info("Evaluating trained model...")
    metrics = trainer.evaluate(n_episodes=100)
    
    # Print evaluation results
    print("\nEvaluation Results:")
    print("-" * 50)
    for metric, value in metrics.items():
        print(f"{metric}: {value}")

if __name__ == "__main__":
    main()
