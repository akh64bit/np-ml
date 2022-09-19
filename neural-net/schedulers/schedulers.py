from abc import ABC, abstractmethod
from copy import deepcopy

import numpy as np

class SchedulerBase(ABC):
    
    def __init__(self):
        self.hyperparameters = {}
        
    def __call__(self, step=None, cur_loss=None):
        return self.learning_rate(step=step, cur_loss=cur_loss)
    
    def copy(self):
        return copy.deepcopy(self)
    
    def set_params(self, hparam_dict):
        """Set the scheduler hyperparameters from a dictionary."""
        if hparam_dict is not None:
            for k, v in hparam_dict.items():
                if k in self.hyperparameters:
                    self.hyperparameters[k] = v
                    
    @abstractmethod
    def learning_rate(self, step=None):
        raise NotImplementedError
    
    
class ConstantScheduler(SchedulerBase):
    def __init__(self, lr=0.01, **kwargs):
        super().__init__()
        self.lr = lr
        self.hyperparameters = {"id": "ConstantScheduler", "lr": self.lr}
        
    def __str__(self):
        return f"ConstantSchedulur(lr={self.lr})"
    
    def learning_rate(self, **kwargs):
        return self.lr