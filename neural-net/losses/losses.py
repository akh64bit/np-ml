from abc import ABC, abstractmethod

import numpy as np

def is_binary(x):
    """Return True if array `x` consists only of binary values"""
    msg = "Matrix must be binary"
    assert np.array_equal(x, x.astype(bool)), msg
    return True


def is_stochastic(X):
    """True if `X` contains probabilities that sum to 1 along the columns"""
    msg = "Array should be stochastic along the columns"
    assert len(X[X < 0]) == len(X[X > 1]) == 0, msg
    assert np.allclose(np.sum(X, axis=1), np.ones(X.shape[0])), msg
    return True


class ObjectiveBase(ABC):
    
    def __init__(self):
        super().__init__()
        
    
    @abstractmethod   
    def loss(self, ytrue, ypred, **kwargs):
        pass
    
    @abstractmethod
    def grad(self, ytrue, ypred, **kwargs):
        pass
    
    
class SquaredError(ObjectiveBase):
    
    def __init__(self):
        super().__init__()
       
    def __str__(self):
        return "SquaredError"
     
    def __call__(self, y, ypred):
        return self.loss(y, ypred)
    
    def loss(self, y, ypred):
        return 0.5 * np.linalg.norm(y - ypred) ** 2
    
    def grad(self, y, y_pred, z, act_fn):
        return (ypred - y) * act_fn.grad(z)
    
    
class CrossEntropy(ObjectiveBase):
    
    def __init__(self):
        super().__init__()
        
    def __str__(self):
        return "CrossEntropy"
    
    def __call__(self, y, ypred):
        
        