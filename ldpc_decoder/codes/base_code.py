"""
Base classes for quantum LDPC codes.

This module provides the base classes for quantum LDPC codes, including
abstract base classes and common functionality.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, List, Optional
import logging

# Set up logging
logger = logging.getLogger(__name__)

class QuantumCode(ABC):
    """
    Abstract base class for quantum error-correcting codes.
    
    This class defines the interface for all quantum codes in the system.
    """
    
    def __init__(self, parameters: Dict[str, Any]):
        """
        Initialize a quantum code.
        
        Args:
            parameters: Dictionary of code parameters.
        """
        self.parameters = parameters
        self.n_checks = parameters.get('n_checks', 0)
        self.n_bits = parameters.get('n_bits', 0)
        self.distance = parameters.get('distance', 0)
        
        # These will be initialized by the derived class
        self.parity_check_matrix = None
        self.stabilizer_generators = None
        self.logical_operators = None
        
    @abstractmethod
    def generate_code(self) -> None:
        """
        Generate the code structure.
        
        This method should initialize the parity check matrix, stabilizer generators,
        and logical operators.
        """
        pass
        
    @abstractmethod
    def get_parity_check_matrix(self) -> np.ndarray:
        """
        Get the parity check matrix for the code.
        
        Returns:
            The parity check matrix as a numpy array.
        """
        pass
        
    @abstractmethod
    def get_stabilizer_generators(self) -> np.ndarray:
        """
        Get the stabilizer generators for the code.
        
        Returns:
            The stabilizer generators as a numpy array.
        """
        pass
        
    @abstractmethod
    def get_logical_operators(self) -> np.ndarray:
        """
        Get the logical operators for the code.
        
        Returns:
            The logical operators as a numpy array.
        """
        pass
        
    def get_code_parameters(self) -> Dict[str, Any]:
        """
        Get the code parameters.
        
        Returns:
            Dictionary of code parameters.
        """
        return {
            'n_checks': self.n_checks,
            'n_bits': self.n_bits,
            'distance': self.distance,
            'rate': (self.n_bits - self.n_checks) / self.n_bits if self.n_bits > 0 else 0
        }
        
    def get_code_graph(self) -> Tuple[List[int], List[int], List[int]]:
        """
        Get the Tanner graph representation of the code.
        
        Returns:
            Tuple of (row indices, column indices, values) for the edges in the graph.
        """
        if self.parity_check_matrix is None:
            raise ValueError("Code not generated yet")
            
        # Convert parity check matrix to COO format
        rows, cols = np.where(self.parity_check_matrix != 0)
        values = self.parity_check_matrix[rows, cols]
        
        return rows.tolist(), cols.tolist(), values.tolist()
        
    def check_syndrome(self, error_pattern: np.ndarray) -> np.ndarray:
        """
        Calculate the syndrome for a given error pattern.
        
        Args:
            error_pattern: Binary vector representing the error pattern.
            
        Returns:
            Binary vector representing the syndrome.
        """
        if self.parity_check_matrix is None:
            raise ValueError("Code not generated yet")
            
        if len(error_pattern) != self.n_bits:
            raise ValueError(f"Error pattern length ({len(error_pattern)}) does not match code length ({self.n_bits})")
            
        return np.mod(np.dot(self.parity_check_matrix, error_pattern), 2)
        
    def is_valid_codeword(self, codeword: np.ndarray) -> bool:
        """
        Check if a vector is a valid codeword.
        
        Args:
            codeword: Binary vector to check.
            
        Returns:
            True if the vector is a valid codeword, False otherwise.
        """
        syndrome = self.check_syndrome(codeword)
        return np.all(syndrome == 0)
        
    def __str__(self) -> str:
        """
        String representation of the code.
        
        Returns:
            String representation.
        """
        return f"{self.__class__.__name__}(n_checks={self.n_checks}, n_bits={self.n_bits}, distance={self.distance})"


class LDPCCode(QuantumCode):
    """
    Base class for quantum LDPC codes.
    
    This class provides common functionality for quantum LDPC codes.
    """
    
    def __init__(self, parameters: Dict[str, Any]):
        """
        Initialize a quantum LDPC code.
        
        Args:
            parameters: Dictionary of code parameters.
        """
        super().__init__(parameters)
        self.sparse_parity_check = None
        
    def get_parity_check_matrix(self) -> np.ndarray:
        """
        Get the parity check matrix for the code.
        
        Returns:
            The parity check matrix as a numpy array.
        """
        if self.parity_check_matrix is None:
            raise ValueError("Code not generated yet")
            
        return self.parity_check_matrix
        
    def get_stabilizer_generators(self) -> np.ndarray:
        """
        Get the stabilizer generators for the code.
        
        Returns:
            The stabilizer generators as a numpy array.
        """
        if self.stabilizer_generators is None:
            raise ValueError("Code not generated yet")
            
        return self.stabilizer_generators
        
    def get_logical_operators(self) -> np.ndarray:
        """
        Get the logical operators for the code.
        
        Returns:
            The logical operators as a numpy array.
        """
        if self.logical_operators is None:
            raise ValueError("Code not generated yet")
            
        return self.logical_operators
        
    def get_sparse_parity_check(self) -> Tuple[List[int], List[int]]:
        """
        Get the sparse representation of the parity check matrix.
        
        Returns:
            Tuple of (row indices, column indices) for the non-zero entries.
        """
        if self.sparse_parity_check is None:
            if self.parity_check_matrix is None:
                raise ValueError("Code not generated yet")
                
            rows, cols = np.where(self.parity_check_matrix != 0)
            self.sparse_parity_check = (rows.tolist(), cols.tolist())
            
        return self.sparse_parity_check
        
    def get_check_degree(self) -> List[int]:
        """
        Get the degree of each check node.
        
        Returns:
            List of degrees for each check node.
        """
        if self.parity_check_matrix is None:
            raise ValueError("Code not generated yet")
            
        return np.sum(self.parity_check_matrix != 0, axis=1).tolist()
        
    def get_bit_degree(self) -> List[int]:
        """
        Get the degree of each bit node.
        
        Returns:
            List of degrees for each bit node.
        """
        if self.parity_check_matrix is None:
            raise ValueError("Code not generated yet")
            
        return np.sum(self.parity_check_matrix != 0, axis=0).tolist()
        
    def get_check_neighbors(self, check_idx: int) -> List[int]:
        """
        Get the neighboring bit nodes for a check node.
        
        Args:
            check_idx: Index of the check node.
            
        Returns:
            List of indices of neighboring bit nodes.
        """
        if self.parity_check_matrix is None:
            raise ValueError("Code not generated yet")
            
        if check_idx < 0 or check_idx >= self.n_checks:
            raise ValueError(f"Check index {check_idx} out of range [0, {self.n_checks-1}]")
            
        return np.where(self.parity_check_matrix[check_idx] != 0)[0].tolist()
        
    def get_bit_neighbors(self, bit_idx: int) -> List[int]:
        """
        Get the neighboring check nodes for a bit node.
        
        Args:
            bit_idx: Index of the bit node.
            
        Returns:
            List of indices of neighboring check nodes.
        """
        if self.parity_check_matrix is None:
            raise ValueError("Code not generated yet")
            
        if bit_idx < 0 or bit_idx >= self.n_bits:
            raise ValueError(f"Bit index {bit_idx} out of range [0, {self.n_bits-1}]")
            
        return np.where(self.parity_check_matrix[:, bit_idx] != 0)[0].tolist()
