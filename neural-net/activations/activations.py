"Collection of activation functions for building the neural network"

from abc import ABC, abstractmethod
import numpy as np

class ActivationBase(ABC):
    def __init__(self, **kwargs):
        """Initialize the ActivationBase object"""
        super().__init__()

    def __call__(self, z):
        """Apply the activation function to an input"""
        if z.ndim == 1:
            z = z.reshape(1, -1)
        return self.fn(z)

    @abstractmethod
    def fn(self, z):
        """Apply the activation function to an input"""
        raise NotImplementedError

    @abstractmethod
    def grad(self, x):
        """Compute the gradient of the activation function wrt the input"""
        raise NotImplementedError
        
        
class Sigmoid(ActivationBase):
    "Sigmoid function"
    def __init__(self):
        super().__init__()
        
    def __str__(self):
        """Return a string representation of the activation function"""
        return "Sigmoid"
    
    def __call__(self, x):
        return self.fn(x)
    
    def fn(self, x):
        return 1 / (1 + np.exp(-x))
    
    def grad(self, x):
        fn_x = self.fn(x)
        return fn_x * (1 - fn_x)
    

class Tanh(ActivationBase):
    "Tanh function"
    def __init__(self):
        super().__init__()
        
    def __str__(self):
        return "Returns the string representation of the Tanh function"
        
    def __call__(self, x):
        return self.fn(x)
    
    def fn(self, x):
        return (np.exp(2 * x) - 1)/(np.exp(2 * x) + 1)
    
    def grad(self, x):
        fn_x = self.fn(x)
        return 1 - fn_x ** 2
    
    
class Exp(ActivationBase):
    "Exponential function"
    def __init__(self):
        super().init__()
        
    def __str__(self):
        return "Exponential"
    
    def __call__(self, x):
        return np.exp(x)
    
    def grad(self, x):
        return np.exp(x)
        
        
class Relu(ActivationBase):
    """This is the class for Relu

    Args:
        ActivationBase (_type_): _description_
    """
    def __init__(self):
        super().__init__()
        
    def __call__(self, x):
        return 0 if x <= 0 else x
    
    def grad(self, x):
        return 0 if x <= 0 else 1