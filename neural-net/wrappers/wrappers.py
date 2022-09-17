    """
    A collection of objects for modifying the layers of the neural network
    """
    
    from abc import ABC, abstractmethod
    
    import numpy as np
    
    class WrapperBase(ABC):
        def __init__(self, wrapped_layer):
            """An abstract base class for all Wrapper instances"""
            self._base_layer = wrapped_layer
            if isinstance(wrapped_layer, _base_layer):
                self._base_layer = wrapped_layer._base_layer
                
            super().__init__()
            
        @abstractmethod
        def _init_wrapper_params(self):
            raise NotImplementedError
        
        @abstractmethod
        def forward(self, z, **kwargs):
            "Overwritten by the inherited class"
            raise NotImplementedError
        
        @abstractmethod
        def backward(self, out, **kwargs):
            """Overwritten by the inherited class"""
            raise NotImplementedError
        
        @property
        def trainable(self):
            "Whether the base layer is trainable of frozen"
            return _base_layer.trainable
        
        @property
        def parameters(self):
            """A dictinary of the base layer parameters"""
            return self._base_layer.parameters
        
        @property
        def hyperparameters(self):
            """A dictionary of the base layer's hyperparameters"""
            hp = self._base_layer.hyperparameters
            hpw = self._wrapper_hyperparameters
            if "wrappers" in hp:
                hp["wrappers"].append(hpw)
            else:
                hp["wrappers"] = [hpw]
                
            return hp
        
        @property
        def act_fn(self):
            """
            Returns the activation used in that layer
            """
            return self._base_layer.act_fn
        
        @property
        def gradients(self):
            """It gives the gradients for the current layer"""
            return self._base_layer.gradients
        
        @property
        def X(self):
            """The collection of the layer input"""
            return self._base_layer.X
        
        def freeze(self):
            """Freeze the weights of the current layer so that they can no longer updated
            """
            self._base_layer.freeze()
            
        def unfreeze(self):
            """Unfreeze the layers of the current layer so that it can be updated
            """
            self._base_layer.unfreeze()
            
            
        