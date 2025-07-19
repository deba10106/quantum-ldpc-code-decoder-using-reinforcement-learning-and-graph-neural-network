#!/usr/bin/env python3
"""
Benchmarking script for LDPC decoder.

This script runs comprehensive benchmarking of LDPC decoders
across different error rates and code parameters.
"""

import os
import sys
import argparse
import logging
import yaml
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

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
from ldpc_decoder.evaluation.evaluator import DecoderEvaluator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("benchmark.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run LDPC decoder benchmarks')
    
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Directory to save results')
    parser.add_argument('--n-episodes', type=int, default=1000,
                        help='Number of episodes to evaluate')
    parser.add_argument('--error-rates', type=float, nargs='+', 
                        default=[0.01, 0.05, 0.1, 0.15, 0.2],
                        help='Error rates to evaluate')
    parser.add_argument('--decoder-types', type=str, nargs='+',
                        default=['gnn', 'mwpm'],
                        help='Decoder types to evaluate')
    parser.add_argument('--model-path', type=str, default='models/final_model',
                        help='Path to trained model')
    parser.add_argument('--compare-only', action='store_true',
                        help='Only compare decoders at a single error rate')
    parser.add_argument('--compare-error-rate', type=float, default=0.1,
                        help='Error rate for decoder comparison')
    
    return parser.parse_args()

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Update configuration with command line arguments
    config['evaluation'] = config.get('evaluation', {})
    config['evaluation']['output_dir'] = args.output_dir
    config['evaluation']['n_episodes'] = args.n_episodes
    config['evaluation']['error_rates'] = args.error_rates
    config['evaluation']['decoder_types'] = args.decoder_types
    config['evaluation']['model_path'] = args.model_path
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
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
    
    # Create evaluator
    evaluator = DecoderEvaluator(config['evaluation'], env, code, error_model)
    
    # Run benchmarks
    if args.compare_only:
        logger.info(f"Comparing decoders at error rate {args.compare_error_rate}...")
        results = evaluator.compare_decoders(args.compare_error_rate)
        
        # Print results
        print("\nDecoder Comparison Results:")
        print(f"Error Rate: {args.compare_error_rate}")
        print("-" * 80)
        print(f"{'Decoder Type':<15} {'Success Rate':<15} {'Logical Error Rate':<20} {'Avg Syndrome Weight':<20}")
        print("-" * 80)
        
        for decoder_type, metrics in results.items():
            print(f"{decoder_type:<15} {metrics['success_rate']:<15.4f} {metrics['logical_error_rate']:<20.4f} {metrics.get('avg_syndrome_weight', 0):<20.4f}")
    else:
        logger.info("Running comprehensive benchmarks...")
        results = evaluator.run_benchmark()
        
        # Print results
        print("\nBenchmark Results:")
        print("-" * 80)
        print(f"{'Decoder Type':<15} {'Error Rate':<15} {'Success Rate':<15} {'Logical Error Rate':<20}")
        print("-" * 80)
        
        for decoder_type, metrics_list in results.items():
            for metrics in metrics_list:
                print(f"{decoder_type:<15} {metrics['error_rate']:<15.4f} {metrics['success_rate']:<15.4f} {metrics['logical_error_rate']:<20.4f}")
                
    logger.info(f"Benchmarking completed. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
