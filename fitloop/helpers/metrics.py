import numpy as np
import matplotlib.pyplot as plt

from .constants import SETS
from .constants import Tensor
from .constants import Dict, Any
from .constants import Optional, Union

"""
Accessing metrics:
  run = 0
  metric = 'loss'

  FitLoop.metrics.valid.start[metric:dict][run:array] -> float
  ----
  FitLoop.metrics.valid.start[metric][run]
  MetricsAggregator.valid.start[metric][run]
  Metrics.start[metric][run]

"""

class Metrics:
    """
    Class to keep track of metrics for a single phase.
    """
    def __init__(self, name, is_train, no_float):
        self.stages = []
        self._name = name
        self._runs = 0
        self._in_run = False
        self._is_train = is_train
        self._no_float = no_float
        
    def __repr__(self):
        stages = ', '.join(self.stages)
        return f"<Metrics({self._name}) :: stages:[{stages}] at {hex(id(self))}>"
    
    def _complete_run(self, is_train):
        """
        Set appropriate flags.
        Convert values for the run to numpy arrays.
        """
        if not is_train == self._is_train:
            return
        
        self._in_run = False
        for stage in self.stages:
            self_stage = getattr(self, stage)
            for metric in self_stage:
                m_run = self_stage[metric][self._runs - 1]
                try:
                    self_stage[metric][self._runs - 1] = np.array(m_run)
                except:
                    pass
                
    def _append(self, stage, rdict):
        """
        Add values in rdict to this object.
        Metrics.stage[name:str][run:int] -> value:float
        """
        if not self._in_run:
            # Set run flag and value
            self._in_run = True
            self._runs += 1
            
        if not hasattr(self, stage):
            self.stages.append(stage)
            setattr(self, stage, {})
            
        self_stage = getattr(self, stage)
        
        for key in rdict:
            val = rdict[key]
            if key not in self_stage:
                self_stage[key] = []
                
            if len(self_stage[key]) < self._runs:
                self_stage[key].append([])
                
            if not self._no_float:
                try:
                    val = float(val)
                except:
                    pass
            self_stage[key][self._runs - 1].append(val)
    
    def clear(self):
        """
        Clears all recorded metrics for this phase.
        """
        for stage in self.stages:
            getattr(self, stage).clear()
        self._runs = 0
        
    def __getitem__(self, item):
        stages = ['epoch_end','batch_step','epoch_start']
        for stage in stages:
            if stage not in self.stages:
                continue
            if item in getattr(self, stage):
                return getattr(self, stage)[item]
        else:
            raise KeyError(f"{item} not found")
            
    def _plot(self, name, metric, run_number, *args, **kwargs):
        if run_number is None:
            m = metric[-1]
        elif run_number is 'all':
            m = np.concatenate(metric)
        elif isinstance(run_number,int):
            m = metric[run_number]
        else:
            raise TypeError("invalid run_number")
        
        plt.plot(np.arange(len(m)),m,*args,**kwargs)
        plt.xlabel('iteration')
        plt.ylabel(name)
        plt.show()
        
    def plot(self, metric:str,run_number:Optional[Union[int,str]]=None, stage:Optional[str]=None,  
             *args, **kwargs) -> None:
        """
        Plots metric against iteration count for
        all phases ['train','valid'] available if the 
        iteration count matches, else it plots metric for 'train'
        
        To plot individual metrics use `FitLoop.M.phase.plot`
        To plot test metrics use `FitLoop.M.test.plot`
        
        phase = ['train','valid','test']
        check FitLoop.M.[phase].stages to check stages available.
        stages = ['epoch_end','batch_step','epoch_start']
        ----
        PARAMETERS:
         - metric : name of the metric that has to be plotted. eg: "running_loss"
         - run_number : metric of which run to be plotted.
             'all' -  metrics of all runs are plotted.
             n (int) - metrics of the nth run are plotted.
             None -  (default) the last run's metrics are plotted.
         - stage : metric from which stage to plot, 
             if None (default) metrics are checked in all stages in the 
             above order.
         - *args : agrs to pass to plt.plot
         - **kwargs : kwargs to pass to plt.plot
            
        """
        if stage is not None:
            if stage not in self.stages:
                raise ValueError(f'invalid stage: {stage}, choose from: {self.stages}')
            else:
                m = getattr(self, stage)[metric]
                self._plot(metric, m, run_number, *args,**kwargs)
        else:
            m = self[metric]
            self._plot(metric, m, run_number, *args, **kwargs)
    
class MetricsAggregator:
    """
    Class to keep track of metrics for all phases.
    """
    _sets =  SETS
    _TR, _VA, _TE = _sets
    def __init__(self):
        self.train_runs = 0
        self.test_runs = 0
        self._in_test = False
        self._in_train = False
        self._is_checkup = False
        self._no_float = None
        self.phases = []
    
    def __repr__(self):
        return f"<MetricsAggregator :: train_runs:{self.train_runs} test_runs:{self.test_runs} at {hex(id(self))}>"
    
    def _start_run(self, is_checkup:bool, is_test:bool, no_float:bool) -> None:
        """
        Set flags and run counters.
        """
        if is_checkup:
            self._is_checkup = True
            return
        
        self._no_float = no_float
        if is_test:
            self._in_test = True
            self.test_runs += 1
        else:
            self._in_train = True
            self.train_runs += 1
    
    def _complete_run(self, is_checkup:bool, is_test:bool) -> None:
        """
        Set flags and run counters for self and
        attached Metrics objects
        """
        if is_checkup:
            self._is_checkup = False
            return
        for phase in self.phases:
            getattr(self, phase)._complete_run(self._in_train)
            
        if is_test:
            self._in_test = False
        else:
            self._in_train = False
        
    def _append(self, phase:str, stage:str, rdict:Dict[str,Union[Tensor, float, Any]]) -> None:
        """
        Create a Metrics object for each phase and attach
        it to self if not present.
        
        ._append rdict values to a stage in each of the 
        metrics object.
        """
        if self._is_checkup:
            return
        
        if not hasattr(self, phase):
            self.phases.append(phase)
            setattr(self, phase, Metrics(phase, self._in_train,self._no_float))
        if self._in_train:
            getattr(self, phase)._append(stage, rdict)
        elif self._in_test:
            getattr(self, phase)._append(stage, rdict)
    
    def clear(self):
        """
        Clears recorded metrics for all phases.
        """
        for phase in self.phases:
            getattr(self, phase).clear()
        self.test_runs = 0
        self.train_runs = 0
    
    def plot(self, metric:str, run_number:Optional[Union[int,str]]=None, *args, **kwargs) -> None:
        """
        Plots metric against iteration count for
        all phases ['train','valid'] available if the 
        iteration count matches, else it plots metric for 'train'
        
        To plot individual metrics use `FitLoop.M.phase.plot`
        To plot test metrics use `FitLoop.M.test.plot`
        
        phase = ['train','valid','test']
        ----
        PARAMETERS:
         - metric : name of the metric that has to be plotted. eg: "running_loss"
         - run_number : metric of which run to be plotted.
             'all' -  metrics of all runs are plotted.
             n (int) - metrics of the nth run are plotted.
             None -  (default) the last run's metrics are plotted.
         - *args : agrs to pass to plt.plot
         - **kwargs : kwargs to pass to plt.plot
            
        """
        m = {}
        for phase in self.phases:
            if phase == self._TE:
                continue
            try:
                ph_metric = getattr(self,phase)[metric]
            except:
                continue
            if run_number is None:
                m[phase] = ph_metric[-1]
            elif run_number is 'all':
                m[phase] = np.concatenate(ph_metric)
            elif isinstance(run_number,int):
                m[phase] = ph_metric[run_number]
            else:
                raise TypeError("invalid run_number")
            
        # Plot both train and valid if the lengths match, else only Train
        if self._TR in m and self._VA in m:
            tr_l = len(m[self._TR])
            va_l = len(m[self._VA])
            if tr_l == va_l:
                plt.plot(np.arange(tr_l),m[self._TR], label=self._TR,*args,**kwargs)
                plt.plot(np.arange(va_l),m[self._VA], label=self._VA,*args,**kwargs)
            else:
                plt.plot(np.arange(tr_l),m[self._TR], label=self._TR,*args,**kwargs)
        elif self._TR in m and self._VA not in m:
            tr_l = len(m[self._TR])
            plt.plot(np.arange(tr_l),m[self._TR], label=self._TR,*args,**kwargs)
        elif self._VA in m and self._TR not in m:
            va_l = len(m[self._VA])
            plt.plot(np.arange(va_l),m[self._VA], label=self._VA,*args,**kwargs)
        else:
            print("no values to plot")
            return

        plt.xlabel('iteration')
        plt.ylabel(metric)
        plt.legend()
        plt.show()
