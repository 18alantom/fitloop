# Imports for type annotations
from typing import NewType

from typing import Union, Callable, Optional
from typing import Tuple, List, Any, Dict

from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau

LRScheduler = NewType("LRScheduler", Union[_LRScheduler, ReduceLROnPlateau])
SETS = ['train','valid','test']
STAGES = ['batch','epoch_start','epoch_end']
MODEL_TYPE = ['pretrained','best']