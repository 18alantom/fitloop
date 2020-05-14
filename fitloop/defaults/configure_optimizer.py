import numpy as np
from copy import deepcopy
from fitloop.helpers.constants import Module, Optimizer, Optional, Union, List
from fitloop.helpers.utils import unf_last_n, get_layers, get_lrs

def configure_optimizer(state,
        lr:Optional[Union[List[float], slice, float, np.ndarray]]=None, 
        unlock:Optional[Union[bool, int]]=None): 
    """
    Can be used to set variable lrs and unlock and lock layers for training.

    Note: Works properly only when initial optimizer has a single param_group,
        and for single optimizers and lr_schedulers.
    ----
    PARAMETERS:
    state : The FitLoop object whose optimizer is to be configured, must
        pass this if calling externally.
    lr : If `lr` is a `slice` then spread the learning rates exponentially
        over all the unlocked layers of the neural network.
    unlock : If `unlock` is `True` unlock all the layers
        else if `unlock` is `int`, unlock the last [`unlock`] layers.
    """
    
    model = state.model
    optimizer = state.optimizer
    
    # Create a copy of the param_groups without params
    _param_groups = optimizer.param_groups
    param_groups = []
    for param_group in _param_groups:
        default = deepcopy({**param_group})
        del default['params']
        param_groups.append(default)
    del _param_groups
        
    if lr is None and unlock is None:
        """
        lr is None : configure_optimizer called internally.
        no changes to lr for all layers.
        """
        # Get all parametrized layers
        optimizer.param_groups.clear()
        layers = get_layers(model)
        for i,layer in enumerate(layers):
            if len(param_groups) < (i + 1):
                defaults = param_groups[-1]
            else:
                defaults = param_groups[i]
            
            pg = {
                'params': layer.parameters(),
                **defaults
            }
            optimizer.add_param_group(pg)
            
    else:
        """
        This block will be reached only through an external call.
        """
        # Unlock the last n layers only.
        if unlock is not None:
            if unlock is True:
                unf_last_n(model)
            else:
                unf_last_n(model, n=unlock)
                
        # Set learning rate for the last n unlocked layers.
        if lr is not None:
            # Get learning rates.
            layers = [*get_layers(model)]
            l = len(layers)
            if isinstance(lr, slice):
                lr = get_lrs(lr, count=len([*get_layers(model, True)]))
                
            if isinstance(lr, (list,np.ndarray)):
                diff = len(lr) - l
                if diff < 0:
                    lrs = [*([0]*-diff),*lr]
                elif diff > 0:
                    lrs = lr[diff:]
                else:
                    lrs = lr
            else:
                lrs = [lr] * l
                
            optimizer.param_groups.clear()
            # Set rates to all the layers.
            for i,(lr, layer) in enumerate(zip(lrs, layers)):
                if len(param_groups) < (i + 1):
                    defaults = param_groups[-1]
                else:
                    defaults = param_groups[i]
                    
                pg = {
                    "params":layer.parameters(),
                    **defaults
                }
                pg['lr'] = lr
                pg['initial_lr'] = lr
                
                optimizer.add_param_group(pg)
            
    if state.lr_scheduler is not None:
        state.lr_scheduler.optimizer = optimizer
        