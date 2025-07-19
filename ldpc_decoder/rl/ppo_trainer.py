"""
PPO trainer implementation for LDPC decoder.

This module provides the implementation of the PPO trainer
for training LDPC decoders using reinforcement learning.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

import numpy as np
import torch
import torch_geometric
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy

from ..env.decoder_env import DecoderEnv
from ..gnn.gnn_model import GNNPolicy
from .custom_wrappers import SimpleMonitorWrapper

# Set up logging
logger = logging.getLogger(__name__)

class FlattenExtractor(BaseFeaturesExtractor):
    """
    Feature extractor that flattens the dictionary observation space for use with MlpPolicy.
    """
    def __init__(self, observation_space: spaces.Dict, total_size: int = 100):
        # We don't use the observation space directly, but we need to provide a features_dim
        super().__init__(observation_space, features_dim=total_size)
        self._total_size = total_size
        
    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Flatten and concatenate all observation components
        syndrome = observations['syndrome'].reshape(observations['syndrome'].shape[0], -1)
        estimated_error = observations['estimated_error'].reshape(observations['estimated_error'].shape[0], -1)
        step = observations['step'].reshape(observations['step'].shape[0], -1)
        
        # Concatenate all features
        features = torch.cat([syndrome, estimated_error, step], dim=1)
        return features


class TensorboardCallback(BaseCallback):
    """
    Custom callback for logging additional metrics to tensorboard.
    """
    
    def __init__(self, verbose: int = 0):
        """
        Initialize the callback.
        
        Args:
            verbose: Verbosity level.
        """
        super().__init__(verbose)
        
    def _on_step(self) -> bool:
        """
        Log metrics on each step.
        
        Returns:
            Whether to continue training.
        """
        # Extract info from the most recent step
        info = {}
        if 'infos' in self.locals and len(self.locals['infos']) > 0:
            info = self.locals['infos'][0]
            
            # Log metrics
            if 'syndrome_weight' in info:
                self.logger.record('env/syndrome_weight', info['syndrome_weight'])
                
            if 'estimated_error_weight' in info:
                self.logger.record('env/estimated_error_weight', info['estimated_error_weight'])
                
            if 'resolved_checks' in info and 'total_checks' in info:
                resolution_ratio = info['resolved_checks'] / info['total_checks']
                self.logger.record('env/resolution_ratio', resolution_ratio)
                
            if 'syndrome_resolved' in info:
                self.logger.record('env/syndrome_resolved', info['syndrome_resolved'])
        
        # Also log episode stats - handle both VecEnv types
        try:
            # For DummyVecEnv and other vectorized environments
            if hasattr(self.model.env, 'get_attr'):
                try:
                    episode_rewards = self.model.env.get_attr('episode_rewards')
                    episode_lengths = self.model.env.get_attr('episode_lengths')
                    if episode_rewards and len(episode_rewards[0]) > 0:
                        self.logger.record('rollout/episode_reward_mean', np.mean([rewards[-1] for rewards in episode_rewards if len(rewards) > 0]))
                        self.logger.record('rollout/episode_length_mean', np.mean([lengths[-1] for lengths in episode_lengths if len(lengths) > 0]))
                except Exception as e:
                    logger.debug(f"Could not log episode stats: {e}")
        except Exception as e:
            logger.debug(f"Error accessing environment attributes: {e}")
                
        if 'logical_error' in info and info['logical_error'] is not None:
            self.logger.record('env/logical_error', int(info['logical_error']))
            
        return True


class CurriculumCallback(BaseCallback):
    """
    Custom callback for curriculum learning.
    """
    
    def __init__(self, 
                 env: DecoderEnv,
                 curriculum_stages: List[str],
                 stage_thresholds: List[float],
                 eval_freq: int = 10000,
                 verbose: int = 0):
        """
        Initialize the callback.
        
        Args:
            env: Decoder environment.
            curriculum_stages: List of curriculum stage names.
            stage_thresholds: List of performance thresholds for advancing to the next stage.
            eval_freq: Frequency of evaluation.
            verbose: Verbosity level.
        """
        super().__init__(verbose)
        self.env = env
        self.curriculum_stages = curriculum_stages
        self.stage_thresholds = stage_thresholds
        self.eval_freq = eval_freq
        self.current_stage_idx = 0
        self.best_success_rate = 0.0
        
        # Start with the first curriculum stage
        self.env.set_curriculum_stage(self.curriculum_stages[0])
        logger.info(f"Starting with curriculum stage {self.curriculum_stages[0]}")
        
    def _on_step(self) -> bool:
        """
        Check if it's time to update the curriculum stage.
        
        Returns:
            Whether to continue training.
        """
        # Check if we should evaluate and potentially update the curriculum
        if self.n_calls % self.eval_freq == 0:
            # Evaluate current performance
            success_rate = self._evaluate_performance()
            
            # Log current performance
            self.logger.record('curriculum/success_rate', success_rate)
            self.logger.record('curriculum/current_stage', self.current_stage_idx)
            
            # Update best success rate
            if success_rate > self.best_success_rate:
                self.best_success_rate = success_rate
                
            # Check if we should advance to the next stage
            if (self.current_stage_idx < len(self.curriculum_stages) - 1 and 
                success_rate >= self.stage_thresholds[self.current_stage_idx]):
                self.current_stage_idx += 1
                new_stage = self.curriculum_stages[self.current_stage_idx]
                self.env.set_curriculum_stage(new_stage)
                logger.info(f"Advanced to curriculum stage {new_stage} with success rate {success_rate:.4f}")
                self.best_success_rate = 0.0  # Reset best success rate for the new stage
                
        return True
        
    def _evaluate_performance(self, n_episodes: int = 100) -> float:
        """
        Evaluate the current performance.
        
        Args:
            n_episodes: Number of episodes to evaluate.
            
        Returns:
            Success rate.
        """
        # Save the current curriculum stage
        original_stage = self.env.curriculum_stage
        
        # Set evaluation mode
        self.env.set_curriculum_stage(self.curriculum_stages[self.current_stage_idx])
        
        # Run evaluation episodes
        successes = 0
        for _ in range(n_episodes):
            obs, _ = self.env.reset()
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                if terminated and info.get('syndrome_resolved', False) and not info.get('logical_error', True):
                    successes += 1
                    
        # Restore original curriculum stage
        self.env.set_curriculum_stage(original_stage)
        
        return successes / n_episodes


# Legacy CustomGNNPolicy class has been removed as it was replaced by MultiInputPolicy
# If you need GNN functionality, use the MultiInputPolicy with appropriate policy_kwargs
        """
        Forward pass.
        
        Args:
            obs: Observation.
            deterministic: Whether to use deterministic action selection.
            
        Returns:
            Tuple of (actions, values, log_probs).
        """
        # Convert observation to PyTorch Geometric Data
        data = self._observation_to_data(obs)
        
        # Get action logits and values from the GNN model
        action_logits, values = self.gnn_model(data)
        
        # Sample actions
        if deterministic:
            actions = torch.argmax(action_logits, dim=-1)
        else:
            actions = torch.multinomial(torch.softmax(action_logits, dim=-1), 1).squeeze(-1)
            
        # Calculate log probabilities
        log_probs = torch.log_softmax(action_logits, dim=-1)
        log_probs = torch.gather(log_probs, -1, actions.unsqueeze(-1)).squeeze(-1)
        
        return actions, values, log_probs
        
    def _observation_to_data(self, obs: Dict[str, torch.Tensor]) -> torch_geometric.data.Batch:
        """
        Convert observation to PyTorch Geometric Data.
        
        Args:
            obs: Observation dictionary.
            
        Returns:
            PyTorch Geometric Data batch.
        """
        batch_size = obs['syndrome'].shape[0]
        data_list = []
        
        for i in range(batch_size):
            # Extract observation components for this batch item
            syndrome = obs['syndrome'][i]
            estimated_error = obs['estimated_error'][i]
            step = obs['step'][i] if 'step' in obs else torch.from_numpy(np.array([0.0], dtype=np.float32))
            
            # Get dimensions
            n_checks = syndrome.shape[0]
            n_bits = estimated_error.shape[0] // 2  # Assuming X and Z errors
            
            # Create node features
            x = torch.zeros(n_checks + n_bits, dtype=torch.float32, device=syndrome.device)
            
            # Set syndrome bits as node features for check nodes
            x[:n_checks] = syndrome
            
            # Set estimated error as node features for bit nodes
            x[n_checks:] = torch.cat([
                estimated_error[:n_bits],
                estimated_error[n_bits:]
            ])
            
            # Get edge index from the observation if available, otherwise use a default
            if 'edge_index' in obs:
                edge_index = obs['edge_index'][i]
            else:
                # In a real implementation, this should be the Tanner graph of the code
                # For now, we'll create a simple bipartite graph connecting each check to each bit
                edge_index = self._create_default_edge_index(n_checks, n_bits, syndrome.device)
            
            # Create node types (0 for check nodes, 1 for variable nodes)
            node_type = torch.cat([
                torch.zeros(n_checks, dtype=torch.long, device=syndrome.device),
                torch.ones(n_bits, dtype=torch.long, device=syndrome.device)
            ])
            
            # Create Data object
            data = torch_geometric.data.Data(
                x=x,
                edge_index=edge_index,
                node_type=node_type,
                n_checks=torch.from_numpy(np.array([n_checks], dtype=np.int64)).to(device=syndrome.device),
                n_bits=torch.from_numpy(np.array([n_bits], dtype=np.int64)).to(device=syndrome.device),
                step=step
            )
            
            data_list.append(data)
        
        # Create batch
        return torch_geometric.data.Batch.from_data_list(data_list)
        
    def _create_default_edge_index(self, n_checks: int, n_bits: int, device: torch.device) -> torch.Tensor:
        """
        Create a default edge index for a Tanner graph.
        
        Args:
            n_checks: Number of check nodes.
            n_bits: Number of bit nodes.
            device: Device to create tensor on.
            
        Returns:
            Edge index tensor.
        """
        # Create a simple bipartite graph connecting each check to each bit
        # This is just a placeholder and should be replaced with the actual Tanner graph
        src = []
        dst = []
        
        # Connect each check node to each bit node (both directions)
        for c in range(n_checks):
            for b in range(n_bits):
                src.extend([c, b + n_checks])
                dst.extend([b + n_checks, c])
        
        # Convert to numpy array first, then to torch tensor
        src_dst_array = np.array([src, dst], dtype=np.int64)
        edge_index = torch.from_numpy(src_dst_array).to(device=device)
        return edge_index
        
    def _predict(self, observation: Dict[str, torch.Tensor], deterministic: bool = False) -> torch.Tensor:
        """
        Predict action from observation.
        
        Args:
            observation: Observation.
            deterministic: Whether to use deterministic action selection.
            
        Returns:
            Action.
        """
        with torch.no_grad():
            actions, _, _ = self.forward(observation, deterministic)
        return actions
        
    def evaluate_actions(self, obs: Dict[str, torch.Tensor], actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions.
        
        Args:
            obs: Observation.
            actions: Actions to evaluate.
            
        Returns:
            Tuple of (values, log_probs, entropy).
        """
        # Convert observation to PyTorch Geometric Data
        data = self._observation_to_data(obs)
        
        # Get action logits and values from the GNN model
        action_logits, values = self.gnn_model(data)
        
        # Calculate log probabilities
        log_probs = torch.log_softmax(action_logits, dim=-1)
        log_probs = torch.gather(log_probs, -1, actions.unsqueeze(-1)).squeeze(-1)
        
        # Calculate entropy
        probs = torch.softmax(action_logits, dim=-1)
        entropy = -torch.sum(probs * log_probs, dim=-1)
        
        return values, log_probs, entropy


class LDPCDecoderTrainer:
    """
    Trainer for LDPC decoder using PPO.
    
    This class implements a trainer for LDPC decoder using PPO algorithm.
    """
    
    def __init__(self, config: Dict[str, Any], env: DecoderEnv):
        """
        Initialize the trainer.
        
        Args:
            config: Trainer configuration.
            env: Decoder environment.
        """
        self.config = config
        self.env = env
        
        # Training parameters
        self.n_envs = config.get('n_envs', 4)
        self.n_steps = config.get('n_steps', 2048)
        self.batch_size = config.get('batch_size', 64)
        self.n_epochs = config.get('n_epochs', 10)
        self.gamma = config.get('gamma', 0.99)
        self.gae_lambda = config.get('gae_lambda', 0.95)
        self.clip_range = config.get('clip_range', 0.2)
        self.ent_coef = config.get('ent_coef', 0.01)
        self.vf_coef = config.get('vf_coef', 0.5)
        self.max_grad_norm = config.get('max_grad_norm', 0.5)
        self.learning_rate = config.get('learning_rate', 3e-4)
        self.total_timesteps = config.get('total_timesteps', 1000000)
        self.log_interval = config.get('log_interval', 10)
        self.save_interval = config.get('save_interval', 10000)
        self.eval_interval = config.get('eval_interval', 10000)
        self.n_eval_episodes = config.get('n_eval_episodes', 10)
        self.seed = config.get('seed', 0)
        
        # Curriculum learning parameters
        self.curriculum = config.get('curriculum', {})
        self.use_curriculum = self.curriculum.get('enabled', True)
        self.curriculum_stages = self.curriculum.get('stages', ['stage1', 'stage2', 'stage3'])
        self.stage_thresholds = self.curriculum.get('thresholds', [0.7, 0.8])
        
        # Paths
        self.log_dir = Path(config.get('log_dir', 'logs'))
        self.save_dir = Path(config.get('save_dir', 'models'))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Set random seed
        set_random_seed(self.seed)
        
        # Create vectorized environment
        self.vec_env = self._make_vec_env()
        
        # Create PPO model
        self.model = self._create_ppo_model()
        
        logger.info(f"Initialized LDPC decoder trainer with {self.n_envs} environments and {self.total_timesteps} total timesteps")
        
    def _make_vec_env(self) -> Union[SubprocVecEnv, DummyVecEnv]:
        """
        Create a vectorized environment.
        
        Returns:
            Vectorized environment.
        """
        # Create environment functions
        env_fns = []
        
        # Get environment configuration
        env_config = self.env.get_config()
        code = self.env.code
        error_model = self.env.error_model
        
        for i in range(self.n_envs):
            def _init(idx=i, config=env_config, c=code, em=error_model):
                # Create a fresh environment instance with the same configuration
                env = DecoderEnv(config, c, em)
                # Use custom SimpleMonitorWrapper instead of Monitor
                env = SimpleMonitorWrapper(env)
                return env
            env_fns.append(_init)
            
        # Create vectorized environment
        if self.n_envs > 1:
            return SubprocVecEnv(env_fns)
        else:
            return DummyVecEnv(env_fns)
            
    def _create_ppo_model(self) -> PPO:
        """
        Create a PPO model.
        
        Returns:
            PPO model.
        """
        # Create callbacks
        self.callbacks = []
        
        # Checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=self.save_interval // self.n_envs,
            save_path=self.save_dir,
            name_prefix='ppo_ldpc_decoder'
        )
        self.callbacks.append(checkpoint_callback)
        
        # Tensorboard callback
        tensorboard_callback = TensorboardCallback()
        self.callbacks.append(tensorboard_callback)
        
        # Re-enable curriculum learning with proper stage progression
        if self.use_curriculum:
            curriculum_callback = CurriculumCallback(
                env=self.env,
                curriculum_stages=self.curriculum_stages,
                stage_thresholds=self.stage_thresholds,
                eval_freq=self.eval_interval // self.n_envs,
                verbose=1
            )
            self.callbacks.append(curriculum_callback)
            
        # Create policy kwargs based on observation space
        policy_kwargs = {}
        
        # Get observation space type
        obs_space = self.vec_env.observation_space
        
        # For dictionary observation spaces, we must use MultiInputPolicy as required by stable-baselines3
        if isinstance(obs_space, spaces.Dict):
            # Define a simple network architecture
            policy_kwargs = {
                'net_arch': [64, 64]
            }
            policy = 'MultiInputPolicy'
        else:
            # For other observation types, use MlpPolicy
            policy_kwargs = {
                'net_arch': [64, 64]
            }
            policy = 'MlpPolicy'
        
        # Create PPO model
        model = PPO(
            policy=policy,
            env=self.vec_env,
            learning_rate=self.learning_rate,
            n_steps=self.n_steps,
            batch_size=self.batch_size,
            n_epochs=self.n_epochs,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            clip_range=self.clip_range,
            ent_coef=self.ent_coef,
            vf_coef=self.vf_coef,
            max_grad_norm=self.max_grad_norm,
            tensorboard_log=self.log_dir,
            policy_kwargs=policy_kwargs,
            verbose=1,
            seed=self.seed
        )
        
        return model
        
    def train(self) -> None:
        """
        Train the model.
        """
        logger.info("Starting training")
        
        # Train the model
        self.model.learn(
            total_timesteps=self.total_timesteps,
            callback=self.callbacks,
            log_interval=self.log_interval
        )
        
        # Save the final model
        self.model.save(os.path.join(self.save_dir, 'final_model'))
        
        logger.info("Training completed")
        
    def evaluate(self, n_episodes: int = 100) -> Dict[str, float]:
        """
        Evaluate the model.
        
        Args:
            n_episodes: Number of episodes to evaluate.
            
        Returns:
            Dictionary of evaluation metrics.
        """
        logger.info(f"Evaluating model for {n_episodes} episodes")
        
        # Reset environment
        obs, _ = self.env.reset()
        
        # Evaluation metrics
        metrics = {
            'success_rate': 0.0,
            'logical_error_rate': 0.0,
            'avg_steps': 0.0,
            'avg_syndrome_weight': 0.0
        }
        
        # Run evaluation episodes
        for _ in range(n_episodes):
            obs, _ = self.env.reset()
            done = False
            steps = 0
            syndrome_weights = []
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                steps += 1
                
                syndrome_weights.append(info.get('syndrome_weight', 0))
                
                if terminated:
                    if info.get('syndrome_resolved', False):
                        metrics['success_rate'] += 1.0 / n_episodes
                        
                        if info.get('logical_error', False):
                            metrics['logical_error_rate'] += 1.0 / n_episodes
                            
            metrics['avg_steps'] += steps / n_episodes
            metrics['avg_syndrome_weight'] += sum(syndrome_weights) / len(syndrome_weights) / n_episodes
            
        logger.info(f"Evaluation results: {metrics}")
        
        return metrics
        
    def save(self, path: str) -> None:
        """
        Save the model.
        
        Args:
            path: Path to save the model.
        """
        self.model.save(path)
        logger.info(f"Saved model to {path}")
        
    def load(self, path: str) -> None:
        """
        Load the model.
        
        Args:
            path: Path to load the model from.
        """
        self.model = PPO.load(path, env=self.vec_env)
        logger.info(f"Loaded model from {path}")
