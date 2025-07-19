"""
Tensor utility functions for LDPC decoder.

This module provides utility functions for working with tensors and PyTorch Geometric data objects.
"""

import numpy as np
import torch
from typing import Any, Dict, Optional, Union


def create_data(x=None, edge_index=None, **kwargs):
    """
    Create a PyTorch Geometric Data object.
    
    Args:
        x: Node feature matrix
        edge_index: Graph connectivity
        **kwargs: Additional attributes
        
    Returns:
        torch_geometric.data.Data: A PyTorch Geometric Data object
    """
    try:
        from torch_geometric.data import Data
        return Data(x=x, edge_index=edge_index, **kwargs)
    except ImportError:
        raise ImportError("PyTorch Geometric not installed or incompatible with current PyTorch version")
