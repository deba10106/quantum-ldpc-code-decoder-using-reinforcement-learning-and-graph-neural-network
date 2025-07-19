# GNN+RL-Based Decoder for Quantum LDPC Codes

A specialized Graph Neural Network + Reinforcement Learning (GNN+RL) based decoder for quantum LDPC codes, with a primary focus on lifted and balanced product LDPC codes.

## Overview

This project implements a decoder for quantum LDPC codes using a combination of Graph Neural Networks (GNN) for feature extraction and Reinforcement Learning (RL) with Proximal Policy Optimization (PPO) for decoding decisions. The system simulates and decodes error syndromes of various types, including correlated errors, to demonstrate the advantages of GNN+RL approaches for these specific code families.

## Features

- Graph-based RL architecture with transfer learning capabilities
- Curriculum learning with error model progression and shape shifting rewards
- Dynamic reward shaping for complex error models (normalized to [0,1])
- Online code simulation with advanced error models via `stim`
- Specialized support for lifted and balanced product LDPC code families
- Comprehensive evaluation metrics and benchmarking

## Tech Stack

- **RL Framework**: stable-baselines3 (PPO policy exclusively)
- **Environment**: gymnasium
- **Deep Learning**: PyTorch with PyTorch Geometric for GNN
- **Quantum Simulation**: stim
- **Configuration**: Single YAML file for all parameters

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ldpc-decoder-rl-gnn.git
cd ldpc-decoder-rl-gnn

# Install dependencies
pip install -r requirements.txt
```

## Usage

The entire system is configured through a single YAML file:

```bash
# Run training with a specific configuration
python -m ldpc_decoder.train --config path/to/config.yaml

# Run evaluation
python -m ldpc_decoder.evaluate --config path/to/config.yaml --model path/to/model.zip

# Run benchmarks comparing different decoders
python scripts/run_benchmarks.py --config path/to/config.yaml --output-dir results

# Evaluate a trained model across different error rates
python scripts/evaluate_model.py --config path/to/config.yaml --model-path models/final_model
```

### Training Pipeline

The training pipeline consists of the following steps:

1. **Configuration Loading**: Load and validate the YAML configuration file.
2. **Code Initialization**: Create the LDPC code instance based on configuration.
3. **Error Model Setup**: Initialize the error model with specified parameters.
4. **Environment Creation**: Set up the gymnasium environment for RL training.
5. **Simulator Integration**: Integrate the stim simulator for syndrome generation.
6. **GNN Policy Setup**: Initialize the GNN policy for the agent.
7. **Curriculum Learning**: Configure curriculum stages for progressive training.
8. **PPO Training**: Train the agent using PPO with the configured parameters.
9. **Evaluation**: Periodically evaluate the agent's performance.
10. **Model Saving**: Save the trained model and configuration.

### Decoding Process

The decoding process follows these steps:

1. **Syndrome Generation**: Generate error syndromes using the stim simulator.
2. **Graph Construction**: Convert syndromes to Tanner graph representations.
3. **GNN Processing**: Process the graph through the GNN to extract features.
4. **Action Selection**: Select qubit flip actions based on the GNN output.
5. **Syndrome Update**: Update the syndrome based on the selected action.
6. **Iteration**: Repeat steps 3-5 until the syndrome is resolved or max steps reached.
7. **Logical Error Check**: Check if the final error pattern results in a logical error.

## Configuration

All aspects of the system are configured through a single YAML file. See `examples/config.yaml` for a comprehensive example.

### Key Configuration Sections

- **System**: General system settings like random seed, device, etc.
- **Code**: LDPC code parameters and type (lifted or balanced product).
- **Error Model**: Error types, rates, and correlation parameters.
- **Environment**: Gymnasium environment settings, reward normalization, etc.
- **RL**: PPO hyperparameters and policy configuration.
- **GNN**: Graph neural network architecture and parameters.
- **Curriculum**: Curriculum learning stages and progression criteria.
- **Training**: Training parameters like total timesteps, evaluation frequency, etc.
- **Evaluation**: Evaluation metrics and settings.
- **Simulator**: Stim simulator configuration for syndrome generation.

### Example Configuration

```yaml
# Minimal example of key configuration sections
system:
  seed: 42
  device: "cuda"

code:
  type: "lifted_product"
  parameters:
    n_checks: 32
    n_bits: 64
    distance: 8

error_model:
  primary_type: "depolarizing"
  error_rate: 0.01

environment:
  max_steps: 100
  observation_type: "syndrome_graph"
  reward_normalization:
    enabled: true

rl:
  algorithm: "PPO"
  hyperparameters:
    n_steps: 2048
    batch_size: 64

gnn:
  model_type: "GCN"
  layers: 3
  hidden_channels: 64

simulator:
  shots: 1000
  seed: 42
```

## Project Structure

```
ldpc_decoder/
├── config/         # Configuration handling
│   ├── config_loader.py     # YAML configuration loader and validator
├── codes/          # LDPC code implementations
│   ├── base_code.py         # Base LDPC code class
│   ├── lifted_product_code.py  # Lifted product code implementation
│   └── balanced_product_code.py # Balanced product code implementation
├── error_models/   # Error model implementations
│   ├── base_error_model.py  # Base error model class
│   ├── depolarizing_error.py # Depolarizing error model
│   ├── measurement_error.py # Measurement error model
│   ├── correlated_error.py  # Correlated error models
│   └── hardware_error.py    # Hardware-inspired error model
├── env/           # Gymnasium environment
│   ├── decoder_env.py       # DecoderEnv gymnasium environment
│   └── curriculum.py        # Curriculum learning implementation
├── gnn/           # GNN models and layers
│   ├── gnn_model.py         # GNN model implementation
│   └── gnn_policy.py        # GNN policy for RL
├── rl/            # RL training and agents
│   ├── ppo_trainer.py       # PPO trainer implementation
│   └── callbacks.py         # Training callbacks
├── simulator/     # Integration with stim
│   ├── stim_simulator.py    # Stim simulator wrapper
│   └── simulator_integration.py # Integration with environment
├── evaluation/    # Evaluation and benchmarking
│   └── evaluator.py         # Decoder evaluation tools
└── utils/         # Utility functions
    ├── logging_utils.py     # Logging utilities
    └── visualization.py     # Visualization utilities
scripts/
├── run_benchmarks.py  # Script for running benchmarks
└── evaluate_model.py  # Script for evaluating trained models
examples/
└── config.yaml        # Example configuration file
```

## Simulator Integration

The decoder integrates with the `stim` quantum circuit simulator to generate realistic error syndromes:

### Key Components

- **StimSimulator**: Wrapper around the stim simulator for syndrome generation.
- **SimulatorIntegration**: Integrates the simulator with the RL environment.

### Features

- **Circuit Generation**: Automatically generates stim circuits from LDPC codes.
- **Noise Models**: Supports various noise models including depolarizing, measurement, and correlated errors.
- **Batch Generation**: Efficiently generates batches of syndromes for training and evaluation.
- **Logical Error Detection**: Identifies logical errors in decoded patterns.
- **Code Validation**: Validates LDPC codes through commutation checks.

### Usage Example

```python
from ldpc_decoder.codes.lifted_product_code import LiftedProductCode
from ldpc_decoder.error_models.depolarizing_error import DepolarizingErrorModel
from ldpc_decoder.simulator.simulator_integration import SimulatorIntegration
from ldpc_decoder.env.decoder_env import DecoderEnv

# Create code and error model
code_config = {"n_checks": 32, "n_bits": 64, "distance": 8}
code = LiftedProductCode(code_config)

error_config = {"error_rate": 0.01}
error_model = DepolarizingErrorModel(error_config)

# Create environment
env_config = {"max_steps": 100, "observation_type": "syndrome_graph"}
env = DecoderEnv(env_config, code, error_model)

# Initialize simulator integration
sim_config = {"shots": 1000, "seed": 42}
simulator = SimulatorIntegration(sim_config, code, error_model)

# Integrate simulator with environment
simulator.integrate_with_env(env)

# Now the environment will use the simulator for syndrome generation
obs, info = env.reset()
```

## Evaluation and Benchmarking

The codebase includes comprehensive evaluation and benchmarking tools:

### Metrics

- **Logical Error Rate**: Rate at which logical errors occur after decoding.
- **Success Rate**: Rate at which syndromes are successfully resolved.
- **Decoding Time**: Time required for decoding.
- **Syndrome Weight**: Average weight of syndromes.

### Benchmarking

The benchmarking tools allow comparison of different decoders:

- **GNN-based RL Decoder**: Our primary decoder using GNN and PPO.
- **MWPM Decoder**: Minimum-weight perfect matching decoder (baseline).
- **Other Decoders**: Framework for comparing with additional decoders.

### Usage Example

```bash
# Run comprehensive benchmarks
python scripts/run_benchmarks.py --config config.yaml --error-rates 0.01 0.05 0.1 0.15 0.2

# Compare decoders at a specific error rate
python scripts/run_benchmarks.py --config config.yaml --compare-only --compare-error-rate 0.1

# Evaluate a trained model with detailed metrics
python scripts/evaluate_model.py --config config.yaml --model-path models/final_model --verbose
```

## License

[MIT License](LICENSE)

## Citation

If you use this codebase in your research, please cite:

```
@software{ldpc_decoder_rl_gnn,
  author = {Your Name},
  title = {GNN+RL-Based Decoder for Quantum LDPC Codes},
  year = {2023},
  url = {https://github.com/yourusername/ldpc-decoder-rl-gnn}
}
```
