import math
import torch

from .constants import STAGES
from .constants import Any, Tensor, DataLoader
from .constants import Tuple, Dict, List
from .constants import Optional, Union

class LoopState:
    """
    Maintains train/valid/test loop state for a single run of 
    a certain number of epochs, does not used to preserve state 
    between runs.
    """
    _stages = STAGES
    _batch_step, _epoch_start, _epoch_end = _stages
    def __init__(self, phase:str, floop:object, no_cast:bool, 
                 no_float:bool, is_train:bool, is_test:bool,
                 dl:DataLoader
                ):
        """
        phase : phase name 'train', 'valid' or 'test'
        floop : the calling FitLoop object
        """
        self.__batch = ()
        self.__floop = floop
        self._no_cast = no_cast
        self._no_float = no_float
        self.phase = phase
        self.batch_num = 0
        self.epoch_num = 0
        self.metrics = {s:{} for s in self._stages}
        self.is_train = is_train
        self.is_test = is_test
        
        # For easy access
        bs = dl.batch_size
        dr = dl.drop_last
        sz = len(dl.dataset)
        bt = sz / bs
        
        # Gives dataset size and batch count
        self.size = sz
        self.batches = math.floor(bt) if dr else math.ceil(bt)
        self.batch_size = 0
    
    def __getattr__(self, name:str) -> Any:
        # To get attributes from the FitLoop object 
        # for use in the stage functions.
        return getattr(self.__floop, name)
    
    def __getitem__(self, metric_name:str):
        # To get the metrics stored in the batch step stage
        metric_value = self.metrics[self._batch_step][metric_name]
        try:
            return torch.tensor(metric_value).float()
        except:
            return metric_value
    
    """
    Getter and setter for the current batch
    """
    @property
    def batch(self) -> Tuple[Tensor,...]:
        if self._no_cast:
            return self.__batch
        
        return (
            d.to(device=self.device,dtype=self.dtype) 
            if d.is_floating_point() 
            else d.to(device=self.device,dtype=torch.long) 
            for d in self.__batch
        )
    
    @batch.setter
    def batch(self, current_batch:Tuple[Tensor,...]) -> None:
        self.__batch = current_batch
        
    """
    Functions to append rdict values to self.metrics
    """
    def _append(self, rdict:Dict[str, float], stage:str) -> None:
        #  Append metrics to the specific stage.
        if rdict is None:
            if stage == self._epoch_end:
                print(f"no rdict returned from: f{self.phase}_{stage}")
            """
            TODO: Add warning if rdict of stage is None
            """
            return
        
        for key in rdict:
            if key not in self.metrics[stage]:
                self.metrics[stage][key] = []
            self.metrics[stage][key].append(rdict[key])
            
    def _append_batch_step(self, rdict:Dict[str, float]) -> None:
        # Called after batch step rdict is returned
        self._append(rdict, self._batch_step)
        
    def _append_epoch_start(self, rdict:Dict[str, float]) -> None:
        # Called before epoch start
        self._append(rdict, self._epoch_start)
        
    def _append_epoch_end(self, rdict:Dict[str, float]) -> None:
        # Called after epoch end step rdict is returned
        self._append(rdict, self._epoch_end)
    
        
    """
    Functions to clear rdict values from self.metrics
    """
    def _clear(self, stage:str) -> None:
        # Clear the batch metrics at the end of the batch.
        for mlist in self.metrics[stage]:
            self.metrics[stage][mlist].clear()
            
    def _clear_batch_step(self) -> None:
        # Called before epoch start
        self._clear(self._batch_step)
        
    def _clear_epoch_start(self) -> None:
        # Called ??
        self._clear(self._epoch_start)
        
    def _clear_epoch_end(self) -> None:
        # Called after loop end
        self._clear(self._epoch_end)
    
    """
    State updates before epoch start and batch step stages
    """
    def _pre_epoch_start_update(self, epoch_num:int) -> None:
        self._clear_batch_step()
        self.batch_num = 0
        self.epoch_num = epoch_num
    
    def _pre_batch_step_update(self, current_batch):
        self.batch_size = current_batch[0].size(0)
        self.batch_num += 1
        self.batch = current_batch
    
    """
    Functions to get various metrics at different stages 
    """
    def _get_epoch_metric(self, criteria:str) -> float:
        # Last added metric that is to be used as a model 
        # selection criteria
        metric = self.metrics[self._epoch_end][criteria][-1]
        if self._no_float:
            return metric
        else:
            return float(metric)
    
    def _get_epoch_metrics(self, 
                display_metrics:Optional[Union[str,List[str]]]=None
                ) -> Dict[str,float]:
        # Return the last saved epoch metrics
        if isinstance(display_metrics, str):
            return {display_metrics:self._get_epoch_metric(display_metrics)}
        elif isinstance(display_metrics, list):
            return {
                metric:self._get_epoch_metric(metric)
                for metric in display_metrics
            }
        else:
            return {
                metric: self._get_epoch_metric(metric)
                for metric in self.metrics[self._epoch_end]
            }