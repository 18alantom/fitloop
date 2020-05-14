import numpy as np
from fitloop.helpers.constants import Module, Optimizer, Optional, Union, List

def get_layers(model:Module, rgrad:bool=False):
    """
    Returns all layers of the model
    that have no children ie will return the
    contents of a sequential but not the Sequential
    
    rgrad : return layers whose parameters `requires_grad`
    """
    for l in model.modules():
        if len([*l.children()]) == 0:
            params = [*l.parameters()]
            if len(params) > 0:
                if rgrad:
                    if params[0].requires_grad:
                        yield l
                    else:
                        continue
                else:
                    yield l

def freeze(model:Module)-> None:
    # Freeze all layers in the model
    for params in model.parameters():
        params.requires_grad = False
        
def unfreeze(model:Module)-> None:
    # Unfreeze all layers in the model
    for params in model.parameters():
        params.requires_grad = True

def unf_last_n(model:Module, n:Optional[int]=None):
    """
    Unfreeze last `n` parametric layers of the 
    model.
    if `n is None` then all layers are unfrozen.
    """
    # Freeze all the layers
    freeze(model)
    
    # Unfreeze only the required layers
    if n is None:
        unfreeze(model)
    else:
        layers = [*get_layers(model)][::-1][:n]
        for layer in layers:
            unfreeze(layer)
        
def get_lrs(lr:slice, count:Optional[int]=None):
    """
    Exponentially increasing lr from 
    slice.start to slice.stop.
    if `count is None` then count = int(stop/start)
    """
    lr1 = lr.start
    lr2 = lr.stop
    if count is None:
        count = int(lr2/lr1)
        
    incr = np.exp((np.log(lr2/lr1)/(count-1)))
    return [lr1*incr**i for i in range(count)]

def print_lr_layer(model:Module, optimizer:Optimizer):
    """
    Function to print requires_grad learning_rates and layer
    for all parameterized layers of an NN.
    """
    layers = [*get_layers(model)]
    pgroup = optimizer.param_groups
    
    if len(pgroup) != len(layers):  
        if len(pgroup) > 1:
            print('param_group, unfrozen layer length mismatch')
        elif len(pgroup) == 1:
            lr = pgroup[0]['lr']
            print(f"lr: {lr}, for: ")
            for l in layers:
                print(l)
    else:
        for pg, layer in zip(pgroup, layers):
            rg = [*layer.parameters()][0].requires_grad
            lr = pg['lr']
            print(f"rg: {str(rg).ljust(5)} lr: {lr:0.10f} :: {layer}")