"""
Correlated error model implementation.

This module provides the implementation of correlated error models,
which simulate spatially and temporally correlated errors in quantum systems.
"""

import numpy as np
from typing import Dict, Any, Tuple, List, Optional
import logging
from scipy import stats
from .base_error_model import CorrelatedErrorModel

# Set up logging
logger = logging.getLogger(__name__)

class SpatiallyCorrelatedErrorModel(CorrelatedErrorModel):
    """
    Spatially correlated error model.
    
    This class implements an error model where errors are spatially correlated,
    meaning that errors on nearby qubits are more likely to occur together.
    """
    
    def __init__(self, parameters: Dict[str, Any]):
        """
        Initialize a spatially correlated error model.
        
        Args:
            parameters: Dictionary of error model parameters.
        """
        super().__init__(parameters)
        
        # Additional parameters for spatial correlation
        self.correlation_length = parameters.get('correlation_length', 2)
        
        logger.info(f"Initialized spatially correlated error model with error rate {self.error_rate} and spatial decay {self.spatial_decay}")
        
    def generate_error(self, code_size: int) -> np.ndarray:
        """
        Generate an error pattern according to the spatially correlated error model.
        
        Args:
            code_size: Size of the quantum code (number of qubits).
            
        Returns:
            Binary array representing the error pattern in the form [X errors, Z errors].
        """
        # Generate correlation matrix
        corr_matrix = self.get_correlation_matrix(code_size)
        
        # Generate multivariate normal distribution with the correlation matrix
        mean = np.zeros(code_size)
        mvn = stats.multivariate_normal(mean=mean, cov=corr_matrix)
        
        # Sample from the multivariate normal distribution
        samples = mvn.rvs(random_state=self.rng)
        
        # Convert to binary error pattern based on error rate threshold
        threshold = stats.norm.ppf(1 - self.error_rate)
        
        # Initialize error pattern
        error_pattern = np.zeros(2 * code_size, dtype=np.int8)
        
        # X part (first half)
        x_part = error_pattern[:code_size]
        # Z part (second half)
        z_part = error_pattern[code_size:]
        
        # Apply X errors based on samples
        x_part[samples > threshold] = 1
        
        # Generate independent samples for Z errors
        samples_z = mvn.rvs(random_state=self.rng)
        z_part[samples_z > threshold] = 1
        
        # Convert some to Y errors (both X and Z)
        y_mask = (x_part == 1) & (self.rng.random(code_size) < 1/3)
        z_part[y_mask] = 1
        
        logger.debug(f"Generated spatially correlated error with {np.sum(x_part)} X errors and {np.sum(z_part)} Z errors")
        return error_pattern


class TemporallyCorrelatedErrorModel(CorrelatedErrorModel):
    """
    Temporally correlated error model.
    
    This class implements an error model where errors are correlated in time,
    meaning that errors in consecutive time steps are more likely to occur together.
    """
    
    def __init__(self, parameters: Dict[str, Any]):
        """
        Initialize a temporally correlated error model.
        
        Args:
            parameters: Dictionary of error model parameters.
        """
        super().__init__(parameters)
        
        # State for temporal correlation
        self.previous_error = None
        
        logger.info(f"Initialized temporally correlated error model with error rate {self.error_rate} and temporal correlation {self.temporal_correlation}")
        
    def generate_error(self, code_size: int) -> np.ndarray:
        """
        Generate an error pattern according to the temporally correlated error model.
        
        Args:
            code_size: Size of the quantum code (number of qubits).
            
        Returns:
            Binary array representing the error pattern in the form [X errors, Z errors].
        """
        # Initialize error pattern
        error_pattern = np.zeros(2 * code_size, dtype=np.int8)
        
        # X part (first half)
        x_part = error_pattern[:code_size]
        # Z part (second half)
        z_part = error_pattern[code_size:]
        
        # If there's a previous error, use it for correlation
        if self.previous_error is not None and len(self.previous_error) == 2 * code_size:
            prev_x = self.previous_error[:code_size]
            prev_z = self.previous_error[code_size:]
            
            # Apply temporal correlation
            for i in range(code_size):
                # X errors
                if prev_x[i] == 1 and self.rng.random() < self.temporal_correlation:
                    # Propagate previous X error
                    x_part[i] = 1
                elif self.rng.random() < self.error_rate / 3:
                    # New X error
                    x_part[i] = 1
                    
                # Z errors
                if prev_z[i] == 1 and self.rng.random() < self.temporal_correlation:
                    # Propagate previous Z error
                    z_part[i] = 1
                elif self.rng.random() < self.error_rate / 3:
                    # New Z error
                    z_part[i] = 1
                    
                # Y errors (both X and Z)
                if x_part[i] == 1 and self.rng.random() < 1/3:
                    z_part[i] = 1
                if z_part[i] == 1 and self.rng.random() < 1/3:
                    x_part[i] = 1
        else:
            # No previous error, generate independent errors
            for i in range(code_size):
                r = self.rng.random()
                if r < self.error_rate / 3:
                    # X error
                    x_part[i] = 1
                elif r < 2 * self.error_rate / 3:
                    # Z error
                    z_part[i] = 1
                elif r < self.error_rate:
                    # Y error (both X and Z)
                    x_part[i] = 1
                    z_part[i] = 1
                    
        # Store current error for next time
        self.previous_error = error_pattern.copy()
        
        logger.debug(f"Generated temporally correlated error with {np.sum(x_part)} X errors and {np.sum(z_part)} Z errors")
        return error_pattern
        
    def reset(self) -> None:
        """
        Reset the temporal correlation state.
        """
        self.previous_error = None


class FullyCorrelatedErrorModel(CorrelatedErrorModel):
    """
    Fully correlated error model.
    
    This class implements an error model with both spatial and temporal correlations.
    """
    
    def __init__(self, parameters: Dict[str, Any]):
        """
        Initialize a fully correlated error model.
        
        Args:
            parameters: Dictionary of error model parameters.
        """
        super().__init__(parameters)
        
        # Create component models
        self.spatial_model = SpatiallyCorrelatedErrorModel(parameters)
        self.temporal_model = TemporallyCorrelatedErrorModel(parameters)
        
        # Mixing weight between spatial and temporal correlations
        self.spatial_weight = parameters.get('spatial_weight', 0.5)
        
        logger.info(f"Initialized fully correlated error model with error rate {self.error_rate}")
        
    def generate_error(self, code_size: int) -> np.ndarray:
        """
        Generate an error pattern according to the fully correlated error model.
        
        Args:
            code_size: Size of the quantum code (number of qubits).
            
        Returns:
            Binary array representing the error pattern in the form [X errors, Z errors].
        """
        # Generate errors from both models
        spatial_errors = self.spatial_model.generate_error(code_size)
        temporal_errors = self.temporal_model.generate_error(code_size)
        
        # Mix the errors based on the weight
        if self.rng.random() < self.spatial_weight:
            return spatial_errors
        else:
            return temporal_errors
            
    def reset(self) -> None:
        """
        Reset the correlation state.
        """
        self.temporal_model.reset()
