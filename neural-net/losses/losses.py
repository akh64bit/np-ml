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
        return self.loss(y, ypred)
    
    def loss(self, y, ypred):
        is_binary(y)
        is_stochastic(y_pred)

        # prevent taking the log of 0
        eps = np.finfo(float).eps

        # each example is associated with a single class; sum the negative log
        # probability of the correct label over all samples in the batch.
        # observe that we are taking advantage of the fact that y is one-hot
        # encoded
        cross_entropy = -np.sum(y * np.log(y_pred + eps))
        return cross_entropy
    
    def grad(self):
        is_binary(y)
        is_stochastic(y_pred)

        # derivative of xe wrt z is y_pred - y_true, hence we can just
        # subtract 1 from the probability of the correct class labels
        grad = y_pred - y

        # [optional] scale the gradients by the number of examples in the batch
        # n, m = y.shape
        # grad /= n
        return grad