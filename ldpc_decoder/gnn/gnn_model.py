"""
GNN model implementation for LDPC decoder.

This module provides the implementation of the Graph Neural Network (GNN)
model for LDPC decoding using PyTorch Geometric.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, GCNConv, GATConv
from torch_geometric.data import Data, Batch
from typing import Dict, Any, Tuple, List, Optional, Union
import logging
import numpy as np

# Set up logging
logger = logging.getLogger(__name__)

class TannerGNNLayer(MessagePassing):
    """
    Custom GNN layer for Tanner graph message passing.
    
    This class implements a custom message passing layer for Tanner graphs,
    which are bipartite graphs representing LDPC codes.
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        """
        Initialize a Tanner GNN layer.
        
        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
        """
        super().__init__(aggr='add')
        
        # Message networks with skip connections and layer normalization
        self.message_var_to_check = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.LayerNorm(out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
            nn.LayerNorm(out_channels)
        )
        
        self.message_check_to_var = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.LayerNorm(out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
            nn.LayerNorm(out_channels)
        )
        
        # Attention mechanism for message aggregation
        self.attention = nn.Sequential(
            nn.Linear(2 * out_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, 1)
        )
        
        # Update networks with residual connections
        self.update_var = nn.GRUCell(out_channels, out_channels)
        self.update_check = nn.GRUCell(out_channels, out_channels)
        
        # Layer normalization for residual connections
        self.norm_var = nn.LayerNorm(out_channels)
        self.norm_check = nn.LayerNorm(out_channels)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, node_type: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Node features.
            edge_index: Edge indices.
            node_type: Node types (0 for check nodes, 1 for variable nodes).
            
        Returns:
            Updated node features.
        """
        # Store input for residual connection
        x_input = x.clone()
        
        # Separate check and variable nodes
        check_mask = (node_type == 0)
        var_mask = (node_type == 1)
        
        # Message passing with attention
        var_to_check_msg = self.propagate(
            edge_index[:, var_mask],
            x=x,
            node_type=node_type,
            direction='var_to_check'
        )
        
        check_to_var_msg = self.propagate(
            edge_index[:, check_mask],
            x=x,
            node_type=node_type,
            direction='check_to_var'
        )
        
        # Update check nodes with residual connection
        x_check = x[check_mask]
        x_check_new = self.update_check(var_to_check_msg[check_mask], x_check)
        x_check_new = self.norm_check(x_check_new + x_check)  # Residual connection
        
        # Update variable nodes with residual connection
        x_var = x[var_mask]
        x_var_new = self.update_var(check_to_var_msg[var_mask], x_var)
        x_var_new = self.norm_var(x_var_new + x_var)  # Residual connection
        
        # Combine updated features
        x_new = x_input.clone()  # Use input for global residual connection
        x_new[check_mask] = x_check_new
        x_new[var_mask] = x_var_new
        
        return x_new
        
    def message(self, x_i: torch.Tensor, x_j: torch.Tensor, direction: str) -> torch.Tensor:
        """
        Message function.
        
        Args:
            x_i: Features of target nodes.
            x_j: Features of source nodes.
            direction: Direction of message passing ('var_to_check' or 'check_to_var').
            
        Returns:
            Messages.
        """
        # Compute messages with attention weights
        if direction == 'var_to_check':
            msg = self.message_var_to_check(x_j)
        else:
            msg = self.message_check_to_var(x_j)
            
        # Compute attention scores
        attention_input = torch.cat([x_i, msg], dim=-1)
        attention_weights = torch.sigmoid(self.attention(attention_input))
        
        # Apply attention
        return msg * attention_weights


class LDPCDecoderGNN(nn.Module):
    """
    GNN model for LDPC decoding.
    
    This class implements a GNN model for LDPC decoding using PyTorch Geometric.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize an LDPC decoder GNN.
        
        Args:
            config: GNN configuration.
        """
        super().__init__()
        
        # GNN parameters with deeper architecture
        self.hidden_channels = config.get('hidden_channels', 128)
        self.num_layers = config.get('num_layers', 4)  # Increased from default
        self.dropout = config.get('dropout', 0.1)
        
        # Input embedding
        self.input_embed = nn.Sequential(
            nn.Linear(1, self.hidden_channels),
            nn.LayerNorm(self.hidden_channels),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        
        # Multiple GNN layers with skip connections
        self.gnn_layers = nn.ModuleList([
            TannerGNNLayer(
                self.hidden_channels if i > 0 else self.hidden_channels,
                self.hidden_channels
            ) for i in range(self.num_layers)
        ])
        
        # Global attention pooling
        self.global_attention = nn.Sequential(
            nn.Linear(self.hidden_channels, self.hidden_channels),
            nn.ReLU(),
            nn.Linear(self.hidden_channels, 1)
        )
        
        # Output networks
        self.action_net = nn.Sequential(
            nn.Linear(self.hidden_channels, self.hidden_channels),
            nn.LayerNorm(self.hidden_channels),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_channels, 4)  # 4 possible actions per qubit
        )
        
        self.value_net = nn.Sequential(
            nn.Linear(self.hidden_channels, self.hidden_channels),
            nn.LayerNorm(self.hidden_channels),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_channels, 1)
        )
        
        logger.info(f"Initialized LDPC decoder GNN with {self.num_layers} layers and {self.hidden_channels} hidden dimensions")
        
    def forward(self, data: Union[Data, Batch]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            data: PyTorch Geometric Data or Batch object.
            
        Returns:
            Tuple of (action logits, value).
        """
        # Extract features and edge indices
        x = data.x
        edge_index = data.edge_index
        node_type = data.node_type
        
        # Add channel dimension for node features
        x = x.unsqueeze(-1)
        
        # Initial node embedding
        x = self.input_embed(x)
        
        # Apply GNN layers
        for layer in self.gnn_layers:
            x = layer(x, edge_index, node_type)
            
        # Extract variable node features (for action prediction)
        var_features = x[node_type == 1]
        
        # Global attention pooling
        global_attention_weights = torch.sigmoid(self.global_attention(var_features))
        global_features = torch.sum(var_features * global_attention_weights, dim=0, keepdim=True)
        
        # Readout for action logits
        action_logits = self.action_net(var_features)
        
        # Value prediction
        value = self.value_net(global_features)
        
        return action_logits, value
        
    def get_action_probs(self, data: Union[Data, Batch]) -> torch.Tensor:
        """
        Get action probabilities.
        
        Args:
            data: PyTorch Geometric Data or Batch object.
            
        Returns:
            Action probabilities.
        """
        action_logits, _ = self.forward(data)
        return F.softmax(action_logits, dim=-1)
        
    def get_value(self, data: Union[Data, Batch]) -> torch.Tensor:
        """
        Get value prediction.
        
        Args:
            data: PyTorch Geometric Data or Batch object.
            
        Returns:
            Value prediction.
        """
        _, value = self.forward(data)
        return value


class GNNPolicy:
    """
    GNN policy for LDPC decoding.
    
    This class implements a policy that uses a GNN model for LDPC decoding.
    """
    
    def __init__(self, config: Dict[str, Any], n_bits: int, n_checks: int):
        """
        Initialize a GNN policy.
        
        Args:
            config: Policy configuration.
            n_bits: Number of bits in the code.
            n_checks: Number of check nodes in the code.
        """
        self.config = config
        self.n_bits = n_bits
        self.n_checks = n_checks
        
        # Create GNN model
        self.model = LDPCDecoderGNN(config)
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        logger.info(f"Initialized GNN policy with {n_bits} bits and {n_checks} checks on device {self.device}")
        
    def predict(self, observation: Dict[str, np.ndarray], deterministic: bool = False) -> Tuple[int, Dict[str, Any]]:
        """
        Predict action from observation.
        
        Args:
            observation: Observation dictionary.
            deterministic: Whether to use deterministic action selection.
            
        Returns:
            Tuple of (action, info).
        """
        # Convert observation to PyTorch Geometric Data
        data = self._observation_to_data(observation)
        data = data.to(self.device)
        
        # Get action probabilities
        with torch.no_grad():
            action_logits, value = self.model(data)
            action_probs = F.softmax(action_logits, dim=-1)
            
        # Select action
        if deterministic:
            action_idx = torch.argmax(action_probs).item()
        else:
            action_idx = torch.multinomial(action_probs, 1).item()
            
        # Convert to environment action
        qubit_idx = action_idx // 4
        flip_type = action_idx % 4
        action = qubit_idx * 4 + flip_type
        
        # Info dictionary
        info = {
            'action_probs': action_probs.cpu().numpy(),
            'value': value.item(),
            'qubit_idx': qubit_idx,
            'flip_type': flip_type
        }
        
        return action, info
        
    def _observation_to_data(self, observation: Dict[str, np.ndarray]) -> Data:
        """
        Convert observation to PyTorch Geometric Data.
        
        Args:
            observation: Observation dictionary.
            
        Returns:
            PyTorch Geometric Data object.
        """
        # Extract observation components
        syndrome = observation['syndrome']
        estimated_error = observation['estimated_error']
        step = observation['step'][0]
        
        # Create node features
        x = np.zeros(self.n_checks + self.n_bits, dtype=np.float32)
        
        # Set syndrome bits as node features for check nodes
        x[:self.n_checks] = syndrome
        
        # Set estimated error as node features for bit nodes
        x[self.n_checks:] = np.concatenate([
            estimated_error[:self.n_bits],
            estimated_error[self.n_bits:]
        ])
        
        # Create edge index (this is a placeholder, the actual Tanner graph
        # should be provided by the environment or code)
        # In a real implementation, this would be the Tanner graph of the code
        edge_index = np.zeros((2, 0), dtype=np.int64)
        
        # Create PyTorch tensors using from_numpy which is preferred over torch.tensor
        x_tensor = torch.from_numpy(x).float()
        edge_index_tensor = torch.from_numpy(edge_index).long()
        
        # Create Data object
        data = Data(x=x_tensor, edge_index=edge_index_tensor)
        
        # Add additional information
        data.n_checks = self.n_checks
        data.n_bits = self.n_bits
        # Use from_numpy for scalar values too
        data.step = torch.from_numpy(np.array([step], dtype=np.float32)).float()
        
        return data
        
    def save(self, path: str) -> None:
        """
        Save the model.
        
        Args:
            path: Path to save the model.
        """
        torch.save(self.model.state_dict(), path)
        logger.info(f"Saved model to {path}")
        
    def load(self, path: str) -> None:
        """
        Load the model.
        
        Args:
            path: Path to load the model from.
        """
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        logger.info(f"Loaded model from {path}")
