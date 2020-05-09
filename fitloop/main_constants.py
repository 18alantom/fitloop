# from typing
from fitloop.helpers.constants import List,  Any, Dict
from fitloop.helpers.constants import Union, Callable, Optional

# Pytorch types
from fitloop.helpers.constants import Tensor 
from fitloop.helpers.constants import Module
from fitloop.helpers.constants import Optimizer
from fitloop.helpers.constants import DataLoader
from fitloop.helpers.constants import LRScheduler

# Constants
SETS = ['train','valid','test']
STAGES = ['batch','epoch_start','epoch_end']
MODEL_TYPE = ['pretrained','best']