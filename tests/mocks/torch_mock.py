"""
Mock implementation of PyTorch for testing.
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional


class Tensor:
    """Mock PyTorch tensor."""
    def __init__(self, data):
        """Initialize the tensor."""
        if isinstance(data, (list, tuple, np.ndarray)):
            self.data = np.array(data)
        else:
            self.data = np.array([data])
        self.shape = self.data.shape
        self.device = 'cpu'
        
    def numpy(self):
        """Convert to numpy array."""
        return self.data
        
    def to(self, device):
        """Move tensor to device."""
        self.device = device
        return self
        
    def __getitem__(self, idx):
        """Get item at index."""
        return Tensor(self.data[idx])
        
    def __add__(self, other):
        """Add operation."""
        if isinstance(other, Tensor):
            return Tensor(self.data + other.data)
        return Tensor(self.data + other)
        
    def __mul__(self, other):
        """Multiply operation."""
        if isinstance(other, Tensor):
            return Tensor(self.data * other.data)
        return Tensor(self.data * other)
        
    def __sub__(self, other):
        """Subtract operation."""
        if isinstance(other, Tensor):
            return Tensor(self.data - other.data)
        return Tensor(self.data - other)
        
    def __truediv__(self, other):
        """Divide operation."""
        if isinstance(other, Tensor):
            return Tensor(self.data / other.data)
        return Tensor(self.data / other)
        
    def __len__(self):
        """Length of tensor."""
        return len(self.data)
        
    def size(self):
        """Size of tensor."""
        return self.shape
        
    def view(self, *shape):
        """Reshape tensor."""
        return Tensor(self.data.reshape(*shape))
        
    def squeeze(self, dim=None):
        """Squeeze tensor."""
        if dim is None:
            return Tensor(self.data.squeeze())
        return Tensor(self.data.squeeze(dim))
        
    def unsqueeze(self, dim):
        """Unsqueeze tensor."""
        return Tensor(np.expand_dims(self.data, dim))


def tensor(data, dtype=None, device='cpu'):
    """Create a tensor."""
    return Tensor(data)


def zeros(*shape, dtype=None, device='cpu'):
    """Create a tensor of zeros."""
    return Tensor(np.zeros(shape))


def ones(*shape, dtype=None, device='cpu'):
    """Create a tensor of ones."""
    return Tensor(np.ones(shape))


def rand(*shape, dtype=None, device='cpu'):
    """Create a tensor of random values."""
    return Tensor(np.random.rand(*shape))


def randn(*shape, dtype=None, device='cpu'):
    """Create a tensor of random values from a normal distribution."""
    return Tensor(np.random.randn(*shape))


# Create a mock nn module
class Module:
    """Base class for all neural network modules."""
    def __init__(self):
        """Initialize the module."""
        self.training = True
        
    def __call__(self, *args, **kwargs):
        """Forward pass."""
        return self.forward(*args, **kwargs)
        
    def forward(self, *args, **kwargs):
        """Forward pass."""
        return Tensor(np.zeros(10))
        
    def parameters(self):
        """Get parameters."""
        return [Tensor(np.zeros(10))]
        
    def train(self, mode=True):
        """Set training mode."""
        self.training = mode
        return self
        
    def eval(self):
        """Set evaluation mode."""
        self.training = False
        return self
        
    def to(self, device):
        """Move module to device."""
        return self


class Linear(Module):
    """Linear layer."""
    def __init__(self, in_features, out_features, bias=True):
        """Initialize the linear layer."""
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Tensor(np.zeros((out_features, in_features)))
        self.bias = Tensor(np.zeros(out_features)) if bias else None
        
    def forward(self, x):
        """Forward pass."""
        return Tensor(np.zeros(self.out_features))


class Sequential(Module):
    """Sequential container."""
    def __init__(self, *args):
        """Initialize the sequential container."""
        super().__init__()
        self.modules = list(args)
        
    def forward(self, x):
        """Forward pass."""
        for module in self.modules:
            x = module(x)
        return x


# Create a mock torch module
nn = type('nn', (), {
    'Module': Module,
    'Linear': Linear,
    'Sequential': Sequential,
    'functional': type('functional', (), {
        'relu': lambda x: x,
        'sigmoid': lambda x: x,
        'tanh': lambda x: x,
        'softmax': lambda x, dim=None: x
    })
})

# Create the mock torch module
class MockTorch:
    def __init__(self):
        self.tensor = tensor
        self.zeros = zeros
        self.ones = ones
        self.rand = rand
        self.randn = randn
        self.nn = nn
        self.device = lambda x: 'cpu'
        self.cuda = type('cuda', (), {
            'is_available': lambda: False
        })
        
    def from_numpy(self, array):
        """Create a tensor from a numpy array (preferred over tensor())."""
        return tensor(array)
    
    def no_grad(self):
        return type('no_grad_context', (), {
            '__enter__': lambda x: None,
            '__exit__': lambda x, *args: None
        })()

# Create the torch instance
torch = MockTorch()
