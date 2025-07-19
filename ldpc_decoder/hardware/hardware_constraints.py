"""
Hardware constraints for quantum LDPC decoders.

This module provides classes and utilities for modeling hardware constraints
such as gate latency, crosstalk, and topology for quantum LDPC decoders.
"""

import os
import json
import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple

# Set up logging
logger = logging.getLogger(__name__)

class HardwareConstraints:
    """
    Hardware constraints for quantum LDPC decoders.
    
    This class models hardware constraints such as gate latency, crosstalk,
    and topology for quantum LDPC decoders.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize hardware constraints.
        
        Args:
            config: Hardware constraints configuration.
        """
        self.config = config
        self.enabled = config.get('enabled', True)
        
        # Set default paths for hardware constraint maps
        base_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'configs', 'hardware')
        default_gate_latency_map = os.path.join(base_path, 'gate_latency.json')
        default_crosstalk_map = os.path.join(base_path, 'crosstalk.json')
        default_fidelity_map = os.path.join(base_path, 'fidelity.json')
        
        # Load hardware constraint maps
        self.gate_latency_map = self._load_constraint_map(config.get('gate_latency_map', default_gate_latency_map))
        self.crosstalk_map = self._load_constraint_map(config.get('crosstalk_map', default_crosstalk_map))
        self.fidelity_map = self._load_constraint_map(config.get('fidelity_map', default_fidelity_map))
        
        # Distance penalty
        self.distance_penalty_enabled = config.get('distance_penalty_enabled', True)
        self.distance_penalty_factor = config.get('distance_penalty_factor', 0.05)
        
        # Default values for missing data
        self.default_gate_latency = 1.0
        self.default_crosstalk = 0.02
        self.default_fidelity = 0.995
        
        # Penalty weights
        self.latency_weight = config.get('latency_weight', 1.0)
        self.crosstalk_weight = config.get('crosstalk_weight', 5.0)
        self.fidelity_weight = config.get('fidelity_weight', 3.0)
        
        logger.info("Initialized hardware constraints")
        
        # Log loaded maps
        logger.debug(f"Loaded gate latency map with {len(self.gate_latency_map)} entries")
        logger.debug(f"Loaded crosstalk map with {len(self.crosstalk_map)} entries")
        logger.debug(f"Loaded fidelity map with {len(self.fidelity_map)} entries")
        
    def _load_constraint_map(self, file_path: Optional[str]) -> Dict[str, Any]:
        """
        Load a constraint map from a JSON file.
        
        Args:
            file_path: Path to the JSON file.
            
        Returns:
            Dictionary containing the constraint map.
        """
        if not file_path:
            return {}
            
        try:
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    return json.load(f)
            else:
                logger.warning(f"Constraint map file not found: {file_path}")
                return {}
        except Exception as e:
            logger.error(f"Error loading constraint map: {e}")
            return {}
            
    def get_gate_latency(self, qubit_idx: int, operation_type: int) -> float:
        """
        Get the gate latency for a specific qubit and operation.
        
        Args:
            qubit_idx: Index of the qubit.
            operation_type: Type of operation (0: No op, 1: X, 2: Z, 3: Y).
            
        Returns:
            Gate latency value.
        """
        if not self.enabled or not self.gate_latency_map:
            return self.default_gate_latency
            
        qubit_key = str(qubit_idx)
        op_key = str(operation_type)
        
        # Check if we have specific data for this qubit and operation
        if qubit_key in self.gate_latency_map:
            if op_key in self.gate_latency_map[qubit_key]:
                return self.gate_latency_map[qubit_key][op_key]
            elif 'default' in self.gate_latency_map[qubit_key]:
                return self.gate_latency_map[qubit_key]['default']
                
        # Check if we have default data for this operation type
        if 'default' in self.gate_latency_map:
            if op_key in self.gate_latency_map['default']:
                return self.gate_latency_map['default'][op_key]
            elif 'default' in self.gate_latency_map['default']:
                return self.gate_latency_map['default']['default']
                
        return self.default_gate_latency
        
    def get_crosstalk(self, qubit_idx: int, operation_type: int) -> float:
        """
        Get the crosstalk for a specific qubit and operation.
        
        Args:
            qubit_idx: Index of the qubit.
            operation_type: Type of operation (0: No op, 1: X, 2: Z, 3: Y).
            
        Returns:
            Crosstalk value.
        """
        if not self.enabled or not self.crosstalk_map:
            return self.default_crosstalk
            
        qubit_key = str(qubit_idx)
        op_key = str(operation_type)
        
        # Check if we have specific data for this qubit and operation
        if qubit_key in self.crosstalk_map:
            if op_key in self.crosstalk_map[qubit_key]:
                return self.crosstalk_map[qubit_key][op_key]
            elif 'default' in self.crosstalk_map[qubit_key]:
                return self.crosstalk_map[qubit_key]['default']
                
        # Check if we have default data for this operation type
        if 'default' in self.crosstalk_map:
            if op_key in self.crosstalk_map['default']:
                return self.crosstalk_map['default'][op_key]
            elif 'default' in self.crosstalk_map['default']:
                return self.crosstalk_map['default']['default']
                
        return self.default_crosstalk
        
    def get_fidelity(self, qubit_idx: int, operation_type: int) -> float:
        """
        Get the gate fidelity for a specific qubit and operation.
        
        Args:
            qubit_idx: Index of the qubit.
            operation_type: Type of operation (0: No op, 1: X, 2: Z, 3: Y).
            
        Returns:
            Gate fidelity value.
        """
        if not self.enabled or not self.fidelity_map:
            return self.default_fidelity
            
        qubit_key = str(qubit_idx)
        op_key = str(operation_type)
        
        # Check if we have specific data for this qubit and operation
        if qubit_key in self.fidelity_map:
            if op_key in self.fidelity_map[qubit_key]:
                return self.fidelity_map[qubit_key][op_key]
            elif 'default' in self.fidelity_map[qubit_key]:
                return self.fidelity_map[qubit_key]['default']
                
        # Check if we have default data for this operation type
        if 'default' in self.fidelity_map:
            if op_key in self.fidelity_map['default']:
                return self.fidelity_map['default'][op_key]
            elif 'default' in self.fidelity_map['default']:
                return self.fidelity_map['default']['default']
                
        return self.default_fidelity
        
    def calculate_distance_penalty(self, qubit_idx1: int, qubit_idx2: int) -> float:
        """
        Calculate the distance penalty between two qubits.
        
        Args:
            qubit_idx1: Index of the first qubit.
            qubit_idx2: Index of the second qubit.
            
        Returns:
            Distance penalty value.
        """
        if not self.enabled or not self.distance_penalty_enabled:
            return 0.0
            
        # In a real implementation, this would use the physical layout of the qubits
        # Here we use a simplified approach based on qubit indices
        distance = abs(qubit_idx1 - qubit_idx2)
        return distance * self.distance_penalty_factor
        
    def get_adjacent_crosstalk(self, qubit_idx1: int, qubit_idx2: int) -> float:
        """
        Get the crosstalk between two adjacent qubits.
        
        Args:
            qubit_idx1: Index of the first qubit.
            qubit_idx2: Index of the second qubit.
            
        Returns:
            Adjacent crosstalk value.
        """
        if not self.enabled or not self.crosstalk_map:
            return self.default_crosstalk
            
        # Sort qubit indices to create a consistent key
        q1, q2 = sorted([qubit_idx1, qubit_idx2])
        adjacent_key = f"{q1}-{q2}"
        
        # Check if we have specific data for this qubit pair
        if 'adjacent_pairs' in self.crosstalk_map and adjacent_key in self.crosstalk_map['adjacent_pairs']:
            return self.crosstalk_map['adjacent_pairs'][adjacent_key]
            
        # If no specific data, use a distance-based approximation
        return self.calculate_distance_penalty(qubit_idx1, qubit_idx2) * self.default_crosstalk
    
    def calculate_hardware_penalty(self, qubit_idx: int, operation_type: int) -> float:
        """
        Calculate the hardware penalty for a specific qubit and operation.
        
        Args:
            qubit_idx: Index of the qubit.
            operation_type: Type of operation (0: No op, 1: X, 2: Z, 3: Y).
            
        Returns:
            Hardware penalty value.
        """
        if not self.enabled:
            return 0.0
            
        # No penalty for no-op
        if operation_type == 0:
            return 0.0
            
        # Calculate penalty based on gate latency, crosstalk, and gate fidelity
        gate_latency = self.get_gate_latency(qubit_idx, operation_type)
        crosstalk = self.get_crosstalk(qubit_idx, operation_type)
        fidelity = self.get_fidelity(qubit_idx, operation_type)
        
        # Combine penalties
        # Higher latency, higher crosstalk, and lower fidelity result in higher penalty
        penalty = (
            self.latency_weight * gate_latency + 
            self.crosstalk_weight * crosstalk + 
            self.fidelity_weight * (1.0 - fidelity)
        )
        
        return penalty

    def calculate_penalty(self, qubit_idx: int, operation_type: int, action_history: List[Dict[str, Any]]) -> float:
        """
        Calculate the hardware penalty for a specific qubit and operation, considering action history.
        This is an alias for calculate_hardware_penalty with additional action history handling.
        
        Args:
            qubit_idx: Index of the qubit.
            operation_type: Type of operation (0: No op, 1: X, 2: Z, 3: Y).
            action_history: List of previous actions taken.
            
        Returns:
            Hardware penalty value.
        """
        base_penalty = self.calculate_hardware_penalty(qubit_idx, operation_type)
        
        # Add crosstalk penalties from recent actions in history
        if action_history and len(action_history) > 1:
            # Get the previous action
            prev_action = action_history[-2]
            # If the previous action was on a different qubit and in the same or previous step
            if prev_action['qubit_idx'] != qubit_idx and \
               abs(prev_action['step'] - len(action_history)) <= 1:
                # Add crosstalk penalty for concurrent operations
                base_penalty += self.get_adjacent_crosstalk(
                    prev_action['qubit_idx'], qubit_idx
                )
        
        return base_penalty
