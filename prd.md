

# Product Requirements Document (PRD)

## Title
**GNN+RL-Based Decoder for Quantum LDPC Codes with Focus on Lifted and Balanced Product Codes**

---

## Overview
Develop a specialized Graph Neural Network + Reinforcement Learning (GNN+RL) based decoder specifically for quantum LDPC codes, with a primary focus on lifted and balanced product LDPC codes. The system will simulate and decode error syndromes of various types, including correlated errors, to demonstrate the advantages of GNN+RL approaches for these specific code families. The architecture will be graph-based, leveraging GNNs for feature extraction from code structure, and employ advanced RL techniques such as curriculum learning and dynamic reward shaping to optimize decoding performance. The platform will integrate with simulation tools like `stim` and provide comprehensive evaluation metrics for comparing performance across different error models.

---

## Command-Line Interface (CLI) Only
- The tool will be implemented strictly as a CLI application.
- No frontend or graphical user interface will be provided.
- All operations, configuration, and outputs will be accessible via the command line.

---

## Feature Summary Checklist
- Graph-Based RL Architecture with Advanced Syndrome Simulation
- GNN encoder for syndrome and code structure feature extraction.
- Meta-RL policy for generalization across code instances and complex noise models.
- Curriculum learning for progressive mastery of error correction across increasingly complex error models.
- Dynamic reward shaping based on logical error rate and correction efficiency.
- Comprehensive syndrome simulation pipeline supporting:
  - Multiple error channels (Pauli, measurement, correlated)
  - Hardware-inspired noise models
  - Configurable error correlations and distributions
  - Realistic circuit-level noise simulation scenarios
- Simulation of various error profiles, including correlated errors
- Online code simulation via `stim` with diverse noise models
- Comprehensive evaluation metrics for LDPC decoding performance
- Benchmarking against traditional LDPC decoders (BP, MWPM)

## Core Components
- Complete implementation of `DecoderEnv` (gymnasium environment)
- Full GNN agent implementation with PPO policy
- Comprehensive YAML configuration system

## Optional Extensions
- Integration with visualization tools (e.g., Qcraft)
- Extension to decoder+code co-design for non-algebraic codes

---

## Key Features

### 0. Decoder Strategy for LDPC Codes: BP vs. GNN+RL

#### Motivation
Decoding quantum LDPC (qLDPC) codes is challenging due to sparse stabilizer constraints, varying degeneracy, and complex noise models. While Belief Propagation (BP) is theoretically sufficient for lifted and balanced product LDPC codes due to their long-cycle Tanner graphs, real-world conditions often break BP's assumptions. This project focuses on developing specialized decoders for lifted and balanced product LDPC codes, with particular emphasis on scenarios where GNN+RL approaches outperform traditional BP.

#### When BP is Sufficient
- **Ideal Conditions**: For lifted and balanced product codes with Tanner graphs featuring long cycles (high girth)
- **Noise Types**: Standard Pauli-type noise (bit-flip, phase-flip) with moderate error rates
- **Code Structure**: When the local neighborhood of each variable node is tree-like for several iterations
- **Advantages**: Computationally efficient, theoretically well-understood, and designed specifically for these code structures

#### When GNN/RL Approaches Are Needed

| Scenario | Why Traditional BP Falls Short | GNN/RL Advantage |
|----------|--------------------------------|-------------------|
| **Circuit-level noise** | Correlated errors violate BP's independence assumption | Learns correlation patterns |
| **Large block sizes** | BP performance saturates due to degeneracy | Better scaling with size |
| **Degenerate errors** | Multiple errors yield same syndrome | Learns statistical patterns |
| **Irregular syndromes** | Non-uniform error landscapes | Adapts to irregularity |
| **Unknown/non-Pauli noise** | BP not optimal for these models | Data-driven adaptation |
| **Asymmetric error rates** | BP assumes symmetry | Learns weighted strategies |
| **Hardware constraints** | BP may be too slow/memory-intensive | Hardware-aware optimization |

#### GNN+RL Architecture for LDPC Codes
- **Graph Representation**: Encode LDPC code structure (Tanner graph) directly into the GNN, with nodes representing qubits and stabilizers, and edges representing their relationships.
  - **Hardware-Aware Graph Augmentation**: Extend the Tanner graph with hardware topology information:
    - Physical layout edges between qubits with potential crosstalk
    - Edge features including physical distance, gate fidelity, and readout error rates
    - Node features including idle decoherence rates and measurement fidelity
- **Feature Extraction**: Use GNN to extract meaningful features from the code structure and syndrome patterns, capturing both local and global correlations.
  - **Hardware-Informed Message Passing**: Use edge weights or edge MLPs to reflect physical distances, gate error rates, or routing time between qubits
  - **Attention-Weighted Decoding**: Based on chip layout and hardware constraints
- **RL Policy**: Train an RL agent to make sequential decoding decisions based on the GNN-extracted features, optimizing for logical error rate minimization.
  - **Action Masking/Cost Biasing**: Mask actions that violate connectivity constraints or down-rank actions on high-crosstalk or high-cost qubits
  - **Multi-objective Optimization**: Balance logical fidelity with hardware cost
- **Specialized for LDPC**: Tailor the architecture specifically for lifted and balanced product LDPC codes, exploiting their structural properties for improved decoding.
  - **Hardware-Specific Optimization**: Adapt decoding strategies based on the underlying quantum hardware topology and constraints

#### LDPC Code Families Comparison

| Feature | **Balanced Product Codes** | **Lifted Product Codes** |
|---------|----------------------------|---------------------------|
| **Construction** | Based on balanced tensor product of two classical codes | Based on graph lifts and automorphism groups |
| **Type** | CSS | CSS |
| **Distance** | Linear in block length (d = Ω(n)) | Polylogarithmic or Ω(n^α) |
| **Rate** | Constant | Constant |
| **Sparsity** | Yes (LDPC) | Yes (LDPC) |
| **Algebraic complexity** | High, but explicit | Medium-high, depends on group/graph |
| **Implementation ease** | Harder to construct manually | Easier if base graph and lift are known |
| **Expansion properties** | Can use expander-like graphs | Typically uses expander base graphs |
| **Decoder compatibility** | Works well with BP, GNN, ML | Same (structure helps GNNs) |
| **Strengths** | Optimal asymptotics (distance + rate) | Modular, geometric, intuitive |
| **Weaknesses** | Difficult construction, less modular | Slightly weaker distance bounds |

#### Comprehensive Error Models for Syndrome Simulation

| Error Type | Description | Implementation |
|------------|-------------|----------------|
| **Pauli Errors** | | |
| - Depolarizing | Random X, Y, Z errors with equal probability | Configurable error rate |
| - Biased | Asymmetric X, Y, Z error probabilities | Configurable bias ratios |
| - Amplitude/Phase Damping | Realistic energy relaxation and dephasing | Kraus operator simulation |
| **Measurement Errors** | | |
| - Readout Error | Incorrect syndrome measurement | Configurable false positive/negative rates |
| - Heralded Erasure | Known measurement failures | Flagged erasure simulation |
| - SPAM | State preparation and measurement errors | Pre/post-circuit noise |
| **Correlated Errors** | | |
| - Crosstalk | Errors induced on neighboring qubits | Spatial correlation matrix |
| - Leakage | Information leakage to non-computational states | Extended state space simulation |
| - Coherent Errors | Systematic unitary errors | Rotation angle distributions |
| - Distance-Dependent | Error probability based on physical distance | Configurable decay functions |
| **Hardware-Specific** | | |
| - Fabrication Defects | Static error patterns from device imperfections | Qubit quality variation maps |
| - Calibration Drift | Time-varying error rates | Temporal correlation functions |
| - Control Crosstalk | ZZ-coupling and microwave bleedthrough | Multi-qubit interaction model |
| - Hot Spots | Regions with elevated error rates | Spatial hotspot simulation |

The syndrome simulation will generate realistic error patterns based on these models, allowing the GNN+RL decoder to learn robust correction strategies for real-world quantum hardware scenarios. All error models will be configurable via YAML and can be combined to create complex, realistic noise profiles.

#### Benefits
| Feature                   | GNN+BP Advantage          | GNN+RL Advantage            |
|--------------------------|---------------------------|-----------------------------|
| Speed                    | Fast                   | Slower                   |
| Sample Efficiency        | High                   | Requires training         |
| Adaptability             | Limited                | Strong                   |
| Theoretical Grounding    | BP theory              | No guarantees             |
| Degeneracy Handling      | Weak                   | Strong                   |
| Non-Algebraic Codes      | Fails                  | Strong                   |
| Non-Degenerate Codes     | Moderate               | Precise                  |
| Correlated Noise         | Poor                   | Robust                   |
| Scalability              | Limited                | High                    |
| Generalization           | Needs re-tuning        | Can transfer              |

#### Implementation Notes
- **Modular Design**: Unified GNN encoder feeds both BP and RL modules. Decoding strategy is YAML-configurable with no hardcoded parameters.
- **Reward Shaping**: Normalized RL rewards are tailored for degenerate (logical recovery) and non-degenerate (exact error correction) codes, with all parameters configurable via YAML. Shape shifting is implemented with curriculum learning to progressively adapt reward signals.
- **Non-Algebraic Support**: Custom utilities generate non-algebraic codes via random stabilizer matrices or graph-based representations.
- **Benchmarking**: Automated suite compares BP-only, GNN+BP, GNN+RL, and hybrid performance across algebraic, non-algebraic, degenerate, and non-degenerate codes.
- **Extensibility**: New decoder modules (e.g., Union-Find, Tensor Network) can be integrated.
- **Framework Specifics**: Leverages stable-baselines3 for RL algorithms, gymnasium for environment interfaces, and PyTorch for neural network implementations.

#### Example Decoder Selection Logic
```python
def adaptive_decode(syndrome_graph, code_info, noise_model, config):
    # All thresholds and parameters loaded from config
    threshold = config['decoder']['degeneracy_threshold']
    special_noise_models = config['decoder']['rl_noise_models']
    
    if code_info.is_algebraic and noise_model == "iid":
        return decode_with_gnn_bp(syndrome_graph, config['bp_decoder'])
    elif (code_info.is_non_algebraic or 
          code_info.degeneracy_score < threshold or 
          noise_model in special_noise_models or
          code_info.is_large_scale):
        # Using PPO policy exclusively for RL-based decoding
        return decode_with_gnn_rl_ppo(syndrome_graph, config['rl_decoder'])
    else:
        return decode_with_gnn_bp(syndrome_graph, config['bp_decoder'])
```

> Note: All RL-based decoding uses PPO policy exclusively as implemented in the `decode_with_gnn_rl_ppo` function.

---

### 1. Graph-Based RL Architecture with Transfer Learning
- Use GNNs to model code structure and syndrome information, supporting non-algebraic and dynamic topologies.
- PPO-based agent with transfer learning capabilities adapts to new codes and noise models, prioritizing non-algebraic non-degenerate codes and correlated noise.
- Modular design for swapping GNN components while maintaining PPO as the exclusive RL algorithm.

### 2. Curriculum Learning with Error Model Progression and Shape Shifting
- Progressive training on increasingly complex codes and noise models.
- Structured error model curriculum:
  1. Start with simple independent Pauli errors
  2. Introduce measurement/readout errors
  3. Add spatial correlations and crosstalk
  4. Incorporate temporal correlations and drift
  5. Combine multiple error types in realistic hardware-inspired models
- Adaptive difficulty based on decoder performance for each error regime.
- Transfer learning from simple to complex error models.
- Shape shifting reward functions that evolve with curriculum progression:
  1. Initial stages: Dense rewards for partial syndrome resolution
  2. Intermediate stages: Balanced rewards for both syndrome resolution and logical error prevention
  3. Advanced stages: Sparse rewards focused primarily on logical error rate minimization
- All reward functions are normalized to [0,1] range for stable training.

### 3. Hardware-Aware Dynamic Reward Shaping for Complex Error Models
- Logical error rate minimization as primary objective.
- Modular, hardware-aware reward function with the following structure:
  ```text
  r_t = α · ΔSynd_t                           # syndrome weight change (shaping term)
      − ε · Step_t                            # small negative per action to encourage speed
      + γ · Suc_T                             # large bonus for successful syndrome clearing
      − δ · LogFail_T                         # large penalty for residual logical error
      − β · ActCost_t                         # penalty for "expensive" actions
      + κ · Deg_t                             # degeneracy bonus for minimum-weight correction
      − ζ · HwPenalty_t                       # hardware-specific penalty
  ```
  Where:
  - **ΔSynd**: Change in syndrome weight between t and t+1 (α ≈ 1.0)
  - **Suc**: One-shot bonus for fully clearing syndrome and returning to trivial logical sector (γ ≈ +100-300)
  - **LogFail**: Large negative when agent stops with residual logical error (δ ≈ -200-500)
  - **Step**: Small negative per action to encourage speed (ε ≈ -0.01)
  - **ActCost**: Penalty for "expensive" actions (β ≈ -0.1-0.5)
  - **Deg**: Degeneracy bonus for minimum-weight correction (κ ≈ ±1)
  - **HwPenalty**: Hardware-specific penalty considering:
    - Gate latency/cost
    - Crosstalk risk
    - Gate infidelity
    - Physical distance between qubits

- Normalized reward function (scaled to [0,1] range) for training stability, implemented as follows:
  ```python
  # Example reward normalization implementation
  def normalize_reward(raw_reward, min_possible, max_possible):
      """Normalize rewards to [0,1] range"""
      # Clip to ensure within expected bounds
      clipped = np.clip(raw_reward, min_possible, max_possible)
      # Min-max normalization
      normalized = (clipped - min_possible) / (max_possible - min_possible)
      return normalized
  ```
  - Each error type has predefined min/max reward values in the YAML config
  - Composite rewards combine multiple normalized components with configurable weights

- Hardware-aware reward components specifically for lifted and balanced product LDPC codes:
  - Higher rewards for correctly identifying correlated error patterns
  - Specialized rewards for handling measurement errors vs. data qubit errors
  - Bonuses for correctly identifying error correlations and crosstalk patterns
  - Penalties for operations that violate hardware constraints or induce crosstalk
  - Cost model for operations based on physical qubit layout and connectivity
  - Gate fidelity considerations in action selection

- Intermediate rewards for partial syndrome resolution in complex error landscapes.
- Exploration bonuses for novel decoding strategies that generalize across error types.
- Penalty shaping for efficient resource usage and decoder robustness.
- Shape shifting integration with curriculum learning to adapt reward structure as training progresses.

- Hardware-specific reward tuning:
  - For lifted/balanced product codes in Pauli error scenarios, BP may be sufficient
  - For hardware-constrained scenarios with crosstalk, readout errors, or correlated noise, the full RL+GNN approach with hardware-aware rewards is necessary

### 4. Online Code Simulation with Advanced Error Models
- Integration with `stim` for efficient syndrome generation.
- Extended simulation capabilities for complex error models:
  - Custom circuit-level noise models for realistic error patterns
  - Measurement error and readout error simulation
  - Crosstalk and spatial correlation simulation
  - Temporal drift and calibration error modeling
  - Hardware-specific error pattern generation
- Real-time syndrome generation during training and evaluation.
- Support for algebraic (qLDPC only) and non-algebraic codes via custom graph or stabilizer inputs.
- Primary integration with `stim` is required and must be fully implemented. The codebase should have a modular simulator interface that could potentially support other simulators (e.g., QECsim, Qiskit-QEC) in the future, but actual implementation of these additional integrations is not required.
- Comprehensive error model library with YAML configuration for reproducible experiments.

### 5. Specialized Support for LDPC Code Families

#### Balanced Product Codes
- Implementation based on Panteleev & Kalachev (2021-2022) construction.
- Utilizes balanced tensor product of two classical parity-check matrices.
- Achieves constant rate and linear minimum distance (d = Ω(n)).
- Optimal asymptotic performance with theoretical guarantees.
- Challenging construction but deep theoretical foundation.

#### Lifted Product Codes
- Based on graph lifts and algebraic automorphisms/permutations.
- Takes small base graphs/codes and lifts to larger ones using group-theoretic symmetries.
- More modular, geometric, and intuitive construction.
- Slightly weaker distance bounds but easier to implement.
- Ideal for visualization and geometric understanding.

#### Implementation Features
- Automatic graph extraction from LDPC check matrices and Tanner graph representations.
- Custom utilities for generating and manipulating both code families with tunable parameters.
- Specialized handling of code structure in the GNN architecture to exploit symmetries.
- Curated dataset of codes with varying parameters for training and benchmarking.
- Comparative analysis tools to evaluate decoder performance across code families.

### 6. Extensible Tech Stack
- Modular Python codebase with stable-baselines3 for RL (using PPO policy exclusively), gymnasium for environments, and PyTorch for deep learning with PyTorch Geometric as the exclusive GNN framework.
- Clear plugin interfaces for new decoders, codes, and noise models.
- Optimized for GPU acceleration and resource-constrained environments.

### 7. Evaluation Metrics with Error-Specific Analysis
- Logical error rate, decoding latency, and resource usage (memory, compute).
- Error-specific performance metrics:
  - Pauli error correction success rates by error type (X, Y, Z)
  - Measurement error detection and correction rates
  - Correlated error pattern recognition accuracy
  - Performance under mixed error models vs. single error types
  - Crosstalk identification and correction efficiency
- Generalization metrics: transfer learning success rate across error models, robustness to unseen error correlations.
- Comparative analysis between BP and GNN+RL performance across different error regimes.
- Benchmarking against classical (MWPM, BP) and ML-based decoders across code families and complex noise models.
- Automated suite for reproducible experiments with detailed error model reporting.

---

## Innovation and Performance Optimization

### Decoder Selection Strategy

| Decoder Type | Use When... | Codes It Fits |
|--------------|-------------|---------------|
| **BP-only** | Tanner graph is sparse and high girth, noise is independent Pauli | Balanced, Lifted Product, Hypergraph |
| **GNN+BP** | Graph is still LDPC, but some degeneracy or mild correlations exist | Any qLDPC with moderate noise complexity |
| **GNN+RL (PPO)** | Code structure is complex or unknown, noise model is correlated or learned | Non-CSS, unknown codes, high degeneracy, or decoder-code co-optimization |

> Note: This project always uses GNN for feature extraction. The distinction is whether the decoder uses BP, RL (always with PPO), or a combination. There is no standalone "RL" or "GNN" decoder - the GNN+RL approach always combines both technologies.

### Performance Goals
- The GNN+RL decoder will aim to surpass classical LDPC decoders (BP, MWPM) specifically in challenging scenarios:
  - Circuit-level noise with correlations
  - High degeneracy regimes
  - Hardware-constrained decoding
  - Asymmetric or non-Pauli error channels
- Specialized handling of correlated noise models where traditional decoders typically fail.
- Curriculum learning and reward shaping tailored to LDPC code structure and error patterns.
- Performance optimization for both error correction capability and decoding speed.
- Comparative analysis between BP and GNN+RL approaches across different error regimes.

---

## Configuration Management
- **Single YAML Configuration**: A comprehensive YAML file controls all aspects of the system with no hardcoded parameters anywhere in the codebase.
- Configuration includes: architecture, code selection, hyperparameters, metrics, logging, datasets, error models, and training parameters.
- All parameters for stable-baselines3, gymnasium environments, and PyTorch models are configurable through this YAML file.
- Ensures reproducibility via YAML and random seeds.
- Extensible for new features without hardcoding.
- Hierarchical structure with sections for model, environment, curriculum, evaluation, and system settings.

### YAML Configuration Structure Example

```yaml
# config.yaml - Main configuration file for LDPC Decoder RL-GNN

# System settings
system:
  seed: 42                    # Random seed for reproducibility
  device: "cuda"              # Device to run on ("cuda" or "cpu")
  num_workers: 4              # Number of parallel workers
  log_level: "INFO"           # Logging level

# Code configuration
code:
  type: "lifted_product"      # Code type: "lifted_product" or "balanced_product"
  parameters:                 # Code-specific parameters
    n_checks: 32
    n_bits: 64
    distance: 8
  custom_code_path: null      # Optional path to custom code definition

# Error model configuration
error_model:
  primary_type: "depolarizing"  # Primary error channel
  error_rate: 0.01             # Physical error rate
  measurement_error_rate: 0.005 # Measurement error rate
  correlations:                # Correlation settings
    enabled: true
    spatial_decay: 0.8
    temporal_correlation: 0.2
  custom_models:               # Custom error model definitions
    - name: "hardware_inspired"
      crosstalk_strength: 0.3
      hotspot_regions: [[0, 5], [10, 15]]

# Environment configuration (gymnasium)
environment:
  max_steps: 100              # Maximum steps per episode
  observation_type: "syndrome_graph"  # Type of observation
  reward_function:           # Modular reward function parameters
    delta_syndrome_weight: 1.0  # α: Weight for syndrome change (ΔSynd)
    success_bonus: 200.0      # γ: Bonus for successful syndrome clearing
    logical_fail_penalty: -300.0 # δ: Penalty for residual logical error
    step_penalty: -0.02       # ε: Small penalty per step
    action_cost_weight: -0.1  # β: Weight for action cost penalty
    degeneracy_bonus: 1.0     # κ: Bonus for minimum-weight correction
    hardware_penalty_weight: 5.0 # ζ: Weight for hardware-specific penalty
    
    # Hardware-specific penalty components
    hardware_penalty:
      enabled: true
      gate_latency_map: "configs/hardware/gate_latency.json"
      crosstalk_risk_map: "configs/hardware/crosstalk.json"
      gate_fidelity_map: "configs/hardware/fidelity.json"
      distance_penalty_enabled: true
      distance_penalty_factor: 0.05
      
  reward_normalization:       # Reward normalization parameters
    enabled: true
    min_reward: -500.0
    max_reward: 300.0
    
  shape_shifting:             # Shape shifting parameters
    enabled: true
    curriculum_stage_rewards:  # Rewards per curriculum stage
      stage1:                 # Initial stage (dense rewards)
        syndrome_resolution_weight: 0.7
        logical_error_weight: 0.3
        hardware_penalty_weight: 0.1
      stage2:                 # Intermediate stage
        syndrome_resolution_weight: 0.5
        logical_error_weight: 0.5
        hardware_penalty_weight: 0.3
      stage3:                 # Advanced stage (sparse rewards)
        syndrome_resolution_weight: 0.2
        logical_error_weight: 0.8
        hardware_penalty_weight: 0.5

# RL algorithm configuration (stable-baselines3)
rl:
  algorithm: "PPO"            # Must be PPO (only supported algorithm)
  hyperparameters:            # PPO hyperparameters
    n_steps: 2048
    batch_size: 64
    n_epochs: 10
    gamma: 0.99
    learning_rate: 0.0003
    clip_range: 0.2
    ent_coef: 0.01
    vf_coef: 0.5
    max_grad_norm: 0.5
  policy_kwargs:              # Policy network configuration
    net_arch: [128, 128]
    activation_fn: "tanh"

# GNN configuration (PyTorch Geometric)
gnn:
  model_type: "GCN"           # GNN architecture type
  layers: 3                   # Number of GNN layers
  hidden_channels: 64         # Hidden channels per layer
  dropout: 0.1                # Dropout rate
  aggregation: "mean"         # Aggregation method
  readout: "global_mean_pool" # Readout function

# Curriculum learning configuration
curriculum:
  enabled: true
  stages:                     # Curriculum stages
    - name: "basic"
      error_types: ["depolarizing"]
      error_rates: [0.005, 0.01, 0.02]
      episodes: 10000
    - name: "measurement_errors"
      error_types: ["depolarizing", "measurement"]
      error_rates: [0.01, 0.02]
      measurement_error_rates: [0.005, 0.01]
      episodes: 10000
    - name: "correlations"
      error_types: ["depolarizing", "measurement", "crosstalk"]
      error_rates: [0.01, 0.02]
      correlation_strengths: [0.2, 0.4]
      episodes: 15000
    - name: "hardware_inspired"
      error_types: ["hardware_inspired"]
      episodes: 20000
  progression_criteria:       # Criteria to progress to next stage
    logical_error_threshold: 0.05
    min_episodes: 5000

# Training configuration
training:
  total_timesteps: 2000000    # Total training timesteps
  eval_freq: 10000            # Evaluation frequency
  save_freq: 50000            # Model saving frequency
  n_eval_episodes: 100        # Number of evaluation episodes
  checkpoint_dir: "./checkpoints"  # Directory for checkpoints

# Evaluation configuration
evaluation:
  metrics:                    # Metrics to track
    - logical_error_rate
    - syndrome_resolution_rate
    - decoding_time
    - memory_usage
  error_specific_metrics:     # Error-specific metrics
    pauli_x_success_rate: true
    pauli_y_success_rate: true
    pauli_z_success_rate: true
    measurement_correction_rate: true
    crosstalk_identification_rate: true
  comparison:                 # Comparison settings
    compare_with_bp: true
    compare_with_mwpm: true
  visualization:              # Visualization settings
    enabled: true
    plot_types: ["error_rates", "training_curves", "confusion_matrix"]
```

This comprehensive YAML configuration demonstrates how all aspects of the system are configurable without hardcoding any parameters in the codebase. The structure is hierarchical and modular, allowing for easy extension and modification.

---

## Non-Functional Requirements
- High code quality, modularity, and extensibility.
- No stubs, placeholders, or incomplete implementations anywhere in the codebase; all code must be fully implemented and functional.
- Comprehensive documentation with examples for non-algebraic code handling.
- Unit and integration tests for core components.
- Resource-efficient design for GPU and CPU environments.
- Clear API for external contributions and plugins.
- Complete YAML configuration with no hardcoded parameters anywhere in the codebase.
- Proper integration with stable-baselines3 (using PPO policy exclusively), gymnasium, and PyTorch frameworks.

---

## Out of Scope
- Non-LDPC quantum code families (e.g., surface codes, color codes) are not the focus, though the framework could be extended to them in future work.
- Hardware-specific decoder implementations.
- Proprietary code families without open definitions.
- General-purpose decoders for arbitrary quantum codes.

---

## Milestones
1. **LDPC Code Generation**: Implementation of lifted and balanced product LDPC code generators with configurable parameters.
2. **Comprehensive Error Simulation**: 
   - Development of Pauli error models (depolarizing, biased)
   - Implementation of measurement and readout error simulation
   - Creation of correlated error models (crosstalk, distance-dependent)
   - Integration of hardware-inspired noise models
   - YAML configuration system for complex error profiles
3. **GNN+RL Decoder**: Specialized GNN+RL architecture optimized for LDPC code structure and complex error models.
4. **Training & Optimization**: 
   - Error-progressive curriculum learning
   - Hyperparameter optimization for different error regimes
   - Transfer learning between error models
5. **Evaluation & Benchmarking**: 
   - Error-specific performance analysis
   - Comprehensive comparison against traditional LDPC decoders across error models
   - Threshold estimation under various error types
6. **Documentation & Release**: Full documentation, examples, and reproducible experiments with error model specifications.

---

## Contact & Collaboration
- For extensions, integration, or co-design, specify requirements (e.g., code scaffold, visualization, or co-design for non-algebraic codes).

---

### Summary of Updates
1. **Non-Algebraic Non-Degenerate Codes**:
   - Added explicit support for non-algebraic non-degenerate codes in the architecture, curriculum, and evaluation.
   - Included custom utilities for generating non-algebraic codes and tailored reward functions for non-degenerate codes.
2. **GNN+RL Scenarios**:
   - Defined scenarios where GNN+RL is essential (non-algebraic codes, correlated noise, dynamic codes, large-scale codes).
   - Emphasized Meta-RL for generalization and scalability.
3. **Tech Stack Specification**:
   - Specified stable-baselines3 for RL algorithms with PPO as the exclusive policy.
   - Specified gymnasium for environment interfaces.
   - Specified PyTorch for neural networks and GNN implementations.
4. **Configuration System**:
   - Single comprehensive YAML configuration file.
   - No hardcoded parameters anywhere in the codebase.
   - Complete configurability of all system components.
   - Normalized reward functions with shape shifting tied to curriculum learning.
5. **General Improvements**:
   - Expanded noise model diversity and generalization metrics.
   - Added resource efficiency requirements.
   - Detailed optional extensions for clarity.
   - Strict prohibition of any placeholder or stub code.

This updated PRD strengthens the focus on non-algebraic non-degenerate codes and GNN+RL scenarios while maintaining the original vision of a generic, extensible decoder. It also specifies the required tech stack (stable-baselines3, gymnasium, PyTorch) and ensures all parameters are configurable through a single YAML file.