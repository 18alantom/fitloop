import time
import math
import warnings
import logging
import torch

from uuid import uuid4
from pathlib import Path
from copy import deepcopy
from tqdm.autonotebook import tqdm

from fitloop.helpers.state import LoopState
from fitloop.helpers.helpers import ftime, ptime
from fitloop.helpers.metrics import MetricsAggregator
from fitloop.main_constants import *
from fitloop.defaults.batch_step import DefaultBatchStep
from fitloop.defaults.epoch_end import DefaultEpochEnd
from fitloop.defaults.configure_optimizer import configure_optimizer
from fitloop.helpers.utils import get_layers

class FitLoop:
    """
    FitLoop trains Pytorch models.
    ----
    PARAMETERS:
    # Basic Blocks
        The bare minimum required along with train_dl.
        - model : nn.Module model that has to be trained
        - optimizer : an optimizer from torch.optim
        - loss_function : function to compute loss
        
    # DataLoader
        - train_dl : training DataLoader
        - valid_dl : validation DataLoader, if None validation will be ignored
        - test_dl : testing DataLoader, if None `.test()` will not run
        
    # Batch Step
        Functions that take in a LoopState object to perform 
        required calculations, functions should return a dict with values
        to be used in the epoch end step.
        - train_step : portion of the loop where forward and backward 
            passes take place.
        - valid_step : validation portion of the loop.
        - test_step : called when `FitLoop.test()` is called.
    
    # Epoch Start Step
        - train_epoch_start : Train phase stage function in the epoch loop at the start.
        - valid_epoch_start : Valid phase stage function in the epoch loop at the start.
        - test_epoch_start : Test phase stage function in the epoch loop at the start.
    
    # Epoch End Step
        Functions that take in a LoopState object to perform 
        required calculations, functions should return a dict with values
        that are to be returned when the loop is over.
        - train_epoch_end : after training epoch has ended.
        - valid_epoch_end : after validation epoch has ended.
        - test_epoch_end : called when the test loop is done, one iteration
            over all batches in the test dataloader.
            
    # Other Stage Functions
        - preloop : function that is called before the epoch loop runs, it is passed
            all the loop variables (local()) in a dict.
        - postloop : function that is called after the epoch loop runs, it is passed
            all the loop variables (local()) in a dict.
    
    # Other Args
        - lr_scheduler : scheduler from torch.optim.lr_scheduler
        - device : torch.device model will be cast to device this prior to the loop
        - configure_optimizer : function that configures the optimizer, will be called
            whenever the model weights have to be restored.
        - dtype : floating point dtype to cast model and data to
        
    # Model Evaluation
        - criteria : model evaluation metric that is returned in the dict of the
            `valid_epoch_end` stage function if None (default) best model and 
            best score are not tracked.
        - criteria_direction : whether more is better (1) or less is better (-1) 
            for model score criteria.
    
    # Model Preservation
        - save_to_disk : True then save state and models to the disk, else best_model
            state_dict is stored as an attribute and state can't be saved.
        - save_path : location to where models and state are saved if `save_to_disk` is
            `True`
        - best_model_name : Name to save the best model by, defaults to a random string 
            appended name.
    """
    
    # ---------------------------------------------------------------------
    """
    SECTION: 0 
    
    Initialization
    """
    _sets = SETS
    _TR, _VA, _TE = _sets
    
    _model_type = MODEL_TYPE
    _PR, _BS = _model_type
    def __init__(self, 
                 # Basic Blocks
                 model: Module, 
                 optimizer: Union[Optimizer,List[Optimizer]], 
                 loss_function: Callable[[Tensor,Tensor],Tensor], 
                 
                 # DataLoader
                 train_dl: Optional[DataLoader]=None, 
                 valid_dl: Optional[DataLoader]=None, 
                 test_dl: Optional[DataLoader]=None, 
                 
                 # Batch Step
                 train_step: Callable[[LoopState],Dict[str, Any]]=DefaultBatchStep.train_step,
                 valid_step: Optional[Callable[[LoopState],Dict[str, Any]]]=DefaultBatchStep.valid_step,
                 test_step: Optional[Callable[[LoopState],Dict[str, Any]]]=None,
                 
                 # Epoch Start Step
                 train_epoch_start: Optional[Callable[[LoopState],Dict[str, Any]]]=None,
                 valid_epoch_start: Optional[Callable[[LoopState],Dict[str, Any]]]=None,
                 test_epoch_start: Optional[Callable[[LoopState],Dict[str, Any]]]=None,
                 
                 # Epoch End Step
                 train_epoch_end: Callable[[LoopState],Dict[str, Any]]=DefaultEpochEnd.train_epoch_end,
                 valid_epoch_end: Optional[Callable[[LoopState],Dict[str, Any]]]=DefaultEpochEnd.valid_epoch_end,
                 test_epoch_end: Optional[Callable[[LoopState],Dict[str, Any]]]=None,
                 
                 # Other Stage Functions
                 preloop: Optional[Callable[[dict],None]]=None,
                 postloop: Optional[Callable[[dict],None]]=None,
                 
                 # Other Args
                 lr_scheduler: Optional[Union[LRScheduler, Any, List[Union[LRScheduler,Any]]]]=None,
                 device: torch.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), 
                 configure_optimizer:Callable[[object],None]=configure_optimizer,
                 dtype: torch.dtype=torch.float32,
                 
                 # Model Evaluation
                 criteria: Optional[str]=None,
                 criteria_direction: int=1,
                 
                 # Model Preservation
                 save_to_disk: bool=True,
                 save_path: str="fitloop_state",
                 best_model_name: Optional[str]=None,
                ) -> None:
        # Basic Blocks
        self._model = None # Setter called below
        self.optimizer = optimizer
        self.loss_function = loss_function
        
        # DataLoaders
        self.train_dl = train_dl
        self.valid_dl = valid_dl
        self.test_dl = test_dl
        
        # Batch Step
        self.train_step = train_step
        self.valid_step = valid_step
        self.test_step = test_step
        
        # Epoch Start Step
        self.train_epoch_start = train_epoch_start
        self.valid_epoch_start = valid_epoch_start
        self.test_epoch_start = test_epoch_start
        
        # Epoch End Step
        self.train_epoch_end = train_epoch_end
        self.valid_epoch_end = valid_epoch_end
        self.test_epoch_end = test_epoch_end
        
        # Other Stage Functions
        self.preloop = preloop
        self.postloop = postloop
        
        # Other Args
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.configure_optimizer = configure_optimizer
        self.dtype = dtype
        
        # Model Evaluation
        self.criteria = criteria
        self.criteria_direction = criteria_direction
        
        
        # Model Preservation
        if best_model_name is None:
            u = str(uuid4()).split('-')[1]
            best_model_name = f"best_{u}.pt"
        self.best_model_name = best_model_name
        self.save_to_disk = save_to_disk
        self.save_path = Path(save_path)
        
        # INITIALIZE NON ARGS
        self.best_model_state_dict = None
        self.epoch_num = 0
        self.best_score = self.criteria_direction * float('-inf')
        self.time_profile = {}
        self.metrics = MetricsAggregator()
        self._temp_state_name = "_temp_state.pt"
        self._temp_model_name = "_temp_model.pt"
        self.state_name = "state.pt"
        self.model_name = "model.pt"
        self._temp = None
        self._temp_state_dict = None
        
        # Change criteria if defaults are being used
        if self.valid_epoch_end is DefaultEpochEnd.valid_epoch_end:
            self.criteria = DefaultEpochEnd.criteria
            
        # Basic Blocks - Calling model setter
        self.model = model
    
    @property
    def model(self):
        return self._model
    
    @model.setter
    def model(self, model):
        self._model = model.to(device=self.device, dtype=self.dtype)
        if self.configure_optimizer is not None:
            self.configure_optimizer(self)
            
    
    def __repr__(self):
        if self.criteria is not None:
            cri = f"{self.criteria}:{self.best_score:0.4f}"
        else:
            cri = f"best_score:{self.best_score}"
        return f"<FitLoop :: epoch_num:{self.epoch_num} {cri} at {hex(id(self))}>"
    
    def __call__(self, x):
        with torch.no_grad():
            return self.model.eval()(x)
            
            
    # ---------------------------------------------------------------------
    """
    SECTION: 1
    
    Helper functions used in `__loop`
    """
    
    def __call_batch_step(self, state:LoopState, track_batch_metrics:bool) -> None:
        phase = state.phase
        step_funcs = [self.train_step, self.valid_step, self.test_step]
        step_funcs = {s:f for s,f in zip(self._sets, step_funcs)}
        step_func = step_funcs[phase]
        
        if step_func is None :
            if state.phase is self._TR:
                raise AttributeError(f"{phase}_step not assigned")
            else:
                return
            
        rdict = step_func(state)
        if isinstance(rdict,dict):
            state._append_batch_step(rdict)
            if track_batch_metrics:
                self.metrics._append(phase,'batch_step',rdict)
        
    def __call_epoch_start_step(self, state:LoopState) -> None:
        phase = state.phase
        step_funcs = [self.train_epoch_start,self.valid_epoch_start,self.test_epoch_start]
        step_funcs = {s:f for s,f in zip(self._sets, step_funcs)}
        step_func = step_funcs[phase]
        
        if step_func is None:
            return
        rdict = step_func(state)
        if isinstance(rdict,dict):
            state._append_epoch_start(rdict)
            self.metrics._append(phase,'epoch_start',rdict)
        
    def __call_epoch_end_step(self, state:LoopState) -> None:
        phase = state.phase
        step_funcs = [self.train_epoch_end,self.valid_epoch_end,self.test_epoch_end]
        step_funcs = {s:f for s,f in zip(self._sets, step_funcs)}
        step_func = step_funcs[state.phase]
        
        if step_func is None:
            return 
        rdict = step_func(state)
        if isinstance(rdict,dict):
            state._append_epoch_end(rdict)
            self.metrics._append(phase,'epoch_end',rdict)
        
    def __get_dl(self, is_test:bool, use_test_dl:Optional[bool]=None,
                train_dl:Optional[DataLoader]=None,
                valid_dl:Optional[DataLoader]=None,
                test_dl:Optional[DataLoader]=None
                )-> Dict[str,DataLoader]:
        if is_test:
            if test_dl is not None:
                return {self._TE:test_dl}
            
            if use_test_dl is not None and not use_test_dl:
                te_dl = valid_dl if valid_dl is not None else self.valid_dl
                if te_dl is None:
                    raise AttributeError("valid_dl not assigned")
                return {self._TE:te_dl}
            
            elif self.test_dl is None:
                raise AttributeError("test_dl not assigned")
            return {self._TE:self.test_dl}
        
        va_dl = valid_dl if valid_dl is not None else self.valid_dl
        tr_dl = train_dl if train_dl is not None else self.train_dl
        
        if tr_dl is None:
            raise AttributeError("train_dl not assigned, please use the train_dl kwarg")
        if  va_dl is not None:
            return {self._TR:tr_dl, self._VA:va_dl}
        else:
            return {self._TR:tr_dl}
    
    def __profile_time(self, t1, t2, name, is_test):
        if t1 is None or t2 is None:
            return
        else:
            t = t2 - t1
            a,*b =  name.split('_')
            if len(b) == 0:
                a += "_t" if is_test else ""
                if a not in self.time_profile:
                    self.time_profile[a] = []
                self.time_profile[a].append(t)
            else:
                b = '_'.join(b)
                b += "_t" if is_test else ""
                if a not in self.time_profile:
                    self.time_profile[a] = {}
                if b not in self.time_profile[a]:
                    self.time_profile[a][b] = []
                self.time_profile[a][b].append(t)
                
    def _print_time_profile(self):
        if len(self.time_profile) == 0:
            logging.error("please run FitLoop.run_profiler(print_outcome=False) first")
        else:
            print("AVERAGE TIMES")
            for i,m in enumerate(self.time_profile):
                if isinstance(self.time_profile[m], list):
                    temp = torch.tensor(self.time_profile[m]).mean().item()
                    prf = f"{i+1}. {m}:".ljust(20)
                    print(f"{prf} {ptime(temp)}")
                else:
                    print(f"{i+1}. {m}")
                    
                    for j,n in enumerate(self.time_profile[m]):
                        temp = torch.tensor(self.time_profile[m][n]).mean().item()
                        prf = f"{j+1}. {n}:".ljust(18)
                        print(f"  {prf} {ptime(temp)}")
            
    
    def __profile_other(self,val,name):
        # TODO: Profiler for other metrics: CPU, GPU, RAM usages.
        raise NotImplementedError("..under construction")
        
        
    # ---------------------------------------------------------------------
    """
    SECTION: 2
    
    The main loop function __loop 
    """
    
    def __loop(self, 
            epochs:int=1,  print_every:int=1, 
            steps: Optional[int]=None, load_best:bool=False, 
            profiler:bool=False, is_test:bool=False,
            track_batch_metrics:bool=True, define_all:bool=False,
            continue_loop:int=0, no_print:bool=False, no_cast:bool=False,
            display_metrics:Optional[Union[str,List[str]]]=None, no_float:bool=False,
            no_progress:bool=False,is_sanity_check:bool=False, use_test_dl:Optional[bool]=None,
            train_dl:Optional[DataLoader]=None,
            valid_dl:Optional[DataLoader]=None,
            test_dl:Optional[DataLoader]=None
           ) -> None:
        """
        Runs the training loop for `epochs`
        ----
        PARAMETERS:
         - epochs : should be a non negative integer
         - print_every : if 0 will not print, else will print at given epoch
         - steps : number of batches to run in each phase [train,valid] 
             for check if everything is working.
         - load_best : whether to load the best model after training, works only if validation
             parameters are defined `valid_dl`, `valid_step`, `valid_epoch_end`
         - profiler : whether to keep track of time taken by various sections
         - is_test : whether it is a model testing loop or training/validation loop
         - track_batch_metrics : whether to store the values returned in the batch steps
         - define_all : If True then `torch.set_grad_enabled`, `optimizer.zero_grad` and model mode 
             ie [train,eval] have to be called where required (usually in the `train_step` function).
         - continue_loop : Will ask whether to continue training after `continue_loop` epochs, should
             be a positive integer.
         - no_print : If True will suppress all print statements, can be used when custom logging is
             used in the stage functions.
         - no_cast : True, if data casting has to be manually set in the stage functions
         - display_metrics : List of metrics returned in the epoch_end stage rdict that has to be 
             displayed, if None (default) all the returned metrics are displayed.
         - no_float : True don't apply float conversion to returned metrics.
         - no_progress : False don't show the progress bars.
         - is_sanity_check : For sanity check mode.
         - use_test_dl : For use with sanity check, to use valid dl or test dl.
         - train_dl : Will use this instead of DatLoader passed in the constructor call.
         - valid_dl : Will use this instead of DatLoader passed in the constructor call.
         - test_dl : Will use this instead of DatLoader passed in the constructor call.
        
        """
        time_ = lambda p : time.perf_counter() if p else None
        tpe = lambda : time_(print_every != 0) # Returns the time 
        tpr = lambda : time_(profiler) # Times keeping used by profiler
        
        prof_total = tpr() # ⏳
        total_time_start = tpe()
        
        # INITILIZING VARIABLES -----
        is_train = not(is_test or is_sanity_check or profiler)
        pre = self.preloop is not None
        post = self.postloop is not None
        self.metrics._start_run((is_sanity_check or profiler),is_test, no_float)
        
        # Storage
        prof_time = {}
        dl = self.__get_dl(is_test, use_test_dl, train_dl, valid_dl, test_dl)
        sz = { k : len(dl[k].dataset) for k in dl }
        phases = [ph for ph in dl]
        state = {ph: LoopState(ph,self,no_cast,no_float,is_train, is_test, dl[ph]) for ph in phases}

        # Markers
        self.__save_best()
        
        # TQDM Progressbar
        tot_size = torch.tensor([len(dl[d]) for d in dl]).sum().long().item()
        if steps is not None:
            tot_size = torch.tensor([len(dl[d]) if len(dl[d]) < steps else steps for d in dl])\
                .sum().long().item()
                                 
        l_bar='{desc}: {percentage:3.0f}%|' 
        r_bar='| [ {n_fmt}/{total_fmt} ] :: [ {elapsed} < {remaining} ] :: [ {rate_fmt} ] ' 
        bar_format = f'{l_bar}'+'{bar}'+f'{r_bar}'
        etqdm = lambda e: tqdm(range(e),desc="EPOCH :", disable=no_progress or is_test, \
                               bar_format=bar_format, unit="epoch",dynamic_ncols=True)
        btqdm = lambda : tqdm(range(tot_size),leave=False or is_test,disable=no_progress,\
                               bar_format=bar_format,unit="batch",dynamic_ncols=True)
         
        # PROFILER STATEMENT --------- (probably should not be here)
        if profiler:
            print(f"RUNNING PROFILER: {'TEST' if is_test else 'TRAIN'} LOOP" , ("" if is_test else f"{epochs} EPOCH(s)"))
            for dlo in dl: 
                dlo_b = len(dl[dlo])
                dlo_s = len(dl[dlo].dataset)
                bs = dl[dlo].batch_size
                if steps is not None and dlo_b > steps:
                    dlo_b = steps
                lb = dlo_s % bs
                lb = lb if lb > 0 else bs
                print(f"  {dlo.ljust(5)} dl :: batches: {dlo_b:4} batch_size: {dl[dlo].batch_size:4} last_batch: {lb:4} dataset_size: {dlo_s:6}")
            print()
        
        
        # CONVENIENCE FUNCTIONS ------
        
        # Function to get formatted epochs (from 1 not 0)
        r_just_val = len(str(epochs))*2 + 3
        estr = lambda e: f"[{e + 1}/{epochs}]".rjust(r_just_val)

        # Function to print every `print_every` epochs.
        def eprint(e,st):
            if not no_print:
                if (e == 0) and (print_every != 0):
                    print(st,end="")
                elif (e + 1) % print_every == 0:
                    print(st,end="")

        # Function for phase strings.
        def statstr(phase, epoch_metrics, rjust=True):
            em = []
            for m in epoch_metrics:
                val = epoch_metrics[m]
                if isinstance(val, float):
                    em.append(f"{m}: {val:0.4f}")
                else:
                    em.append(f"{m}: {val}")

            mt = ' | '.join(em)
            st =  f"{phase} :: {mt} \n"
            if rjust:
                return st.rjust(r_just_val + len(st) + 3)
            else:
                return st
            
        # To set is_test
        def _profile_time(t1,t2,name):
            self.__profile_time(t1,t2,name,is_test=is_test)
            
        prof_preloop = tpr()
        pre and self.preloop(locals())
        pre and profiler and _profile_time(prof_preloop, tpr(), 'preloop') # ⏳
            
        profiler and _profile_time(prof_total, tpr(), 'initialize') # ⏳
        
        # EPOCH LOOP - START -----------
        prof_epoch_loop = tpr()
        for e in etqdm(epochs):
            prof_epoch_inner = tpr() # ⏳
            epoch_time_start = tpe()
            
            # UPDATE: epoch_num
            if not is_sanity_check and not profiler and not is_test:
                self.epoch_num += 1
            
            # PHASE LOOP [TRAIN|VALID,TEST] - START
            prof_phase_loop = tpr() # ⏳
            prog_bar_phase = btqdm()
            for phase in phases:
                prof_phase_inner = tpr() # ⏳
                prog_bar_phase.desc = phase.upper().ljust(5)+" :"
                
                # EPOCH START STEP - START 
                prof_epoch_start = tpr() # ⏳
                self.__call_epoch_start_step(state[phase])
                profiler and _profile_time(prof_epoch_start,tpr(),f'{phase}_epoch_start') # ⏳
                # EPOCH START STEP - END 
                
                is_tr = phase == self._TR
                if is_tr:
                    eprint(e,estr(e)+f" - ")
                
                # UPDATE: batch_num, metrics['batch'], epoch_num
                state[phase]._pre_epoch_start_update(e)
                
                
                if not define_all:
                    if is_tr:
                          self.model.train()
                    else:
                          self.model.eval()
                            
                # BATCH LOOP - START 
                prof_batch_loop = tpr() # ⏳
                for step, batch in enumerate(dl[phase]):
                    prof_batch_inner = tpr() # ⏳
                    
                    if steps is not None and step == steps: break
                    
                    # Update LoopState: batch_num, batch and batch_size
                    state[phase]._pre_batch_step_update(batch)
                    
                    # BATCH STEP - START 
                    prof_batch_step = tpr() # ⏳
                    if define_all:
                        self.__call_batch_step(state[phase], track_batch_metrics)
                    else:
                        if isinstance(self.optimizer,list):
                            for opt in self.optimizer:opt.zero_grad()
                        else:
                            self.optimizer.zero_grad()
                        with torch.set_grad_enabled(is_tr):
                            self.__call_batch_step(state[phase], track_batch_metrics)
                    profiler and _profile_time(prof_batch_step,tpr(),f'{phase}_step') # ⏳
                    # BATCH STEP - END 
                    prog_bar_phase.update(1)
                    
                    profiler and _profile_time(prof_batch_inner,tpr(),f'{phase}_batch_inner') # ⏳
                    
                profiler and _profile_time(prof_batch_loop,tpr(),f'{phase}_batch_loop') # ⏳
                # BATCH LOOP - END 
                
                # EPOCH END STEP - START 
                prof_epoch_end = tpr()
                self.__call_epoch_end_step(state[phase])
                profiler and _profile_time(prof_epoch_end,tpr(),f'{phase}_epoch_end') # ⏳
                # EPOCH END STEP - END 
                
                # UPDATE MARKERS
                if not (is_tr or is_test) and self.criteria is not None:
                    score = state[phase]._get_epoch_metric(self.criteria)
                    direc = self.criteria_direction > 0
                    is_better = (score >= self.best_score) if direc else (score <= self.best_score)
                    if is_better:
                        self.best_score = score
                        self.__save_best()

                # PRINT EPOCH[PHASE] METRICS
                epoch_metrics = state[phase]._get_epoch_metrics(display_metrics)
                if len(epoch_metrics):
                    if is_tr or is_test:
                        eprint(e,statstr(phase, epoch_metrics,False))
                    else:
                        eprint(e,statstr(phase, epoch_metrics))
                    
                profiler and _profile_time(prof_phase_inner,tpr(),f'{phase}_phase_inner') # ⏳
                
            profiler and _profile_time(prof_phase_loop,tpr(),f'phase loop') # ⏳
            # PHASE LOOP [TRAIN|VALID,TEST] - END
            
            # PRINT EPOCH TIMES
            epoch_time_end = tpe()
            epoch_time = ftime(epoch_time_start, epoch_time_end)
            epoch_time = f"epoch time: {epoch_time}" + ("\n" if no_progress else "")
            not is_test and eprint(e,epoch_time.rjust(len(epoch_time) + r_just_val + 3)+"\n")
            
            prog_bar_phase.close()
            # CONTINUE LOOP ?
            if continue_loop > 0 and \
                not (is_sanity_check or is_test or profiler) and \
                ((e + 1) % continue_loop == 0) and (e + 1 != epochs):
                cont = input("continue loop ([y]/n): ")
                not no_print and print()
                if cont == 'n':
                    break
            
            profiler and _profile_time(prof_epoch_inner,tpr(),f'epoch_inner') # ⏳

        profiler and _profile_time(prof_epoch_loop,tpr(),f'epoch_loop') # ⏳
        # EPOCH LOOP - END 
        
        prof_postloop = tpr()
        post and self.postloop(locals())
        post and profiler and _profile_time(prof_postloop, tpr(), 'postloop') # ⏳
        
        # PRINT FINAL METRICS
        eprint(0,"-"*r_just_val)
        total_time_end = tpe()
        total_time = ftime(total_time_start,total_time_end)
        eprint(0, f"\ntotal time: {total_time}\n")
        if self.criteria is not None and not is_test and self._VA in dl:
            eprint(0, f"best score: {self.best_score:0.4f}\n")

        # RESTORE BEST MODEL
        prof_restore_model = tpr() # ⏳
        if load_best:
            self.__load_best()
        profiler and _profile_time(prof_restore_model,tpr(),f'restore model') # ⏳

        self.metrics._complete_run((is_sanity_check or profiler),is_test)
        profiler and _profile_time(prof_total,tpr(),f'total',) # ⏳
        # __loop - END 

    
    # ---------------------------------------------------------------------
    """
    SECTION: 3 - A
    
    Loop methods for training and testing of the model.
    """
    
    def fit(self, 
            epochs:int=1, print_every:int=1,
            display_metrics:Optional[Union[str,List[str]]]=None,
            train_dl:Optional[DataLoader]=None,
            valid_dl:Optional[DataLoader]=None,
            track_batch_metrics:bool=True, load_best:bool=True,
            continue_loop:int=0, define_all:bool=False,  
            no_print:bool=False, no_cast:bool=False,
            no_float:bool=False, no_progress:bool=False,
           ) -> None:
        """
        Runs the training loop for `epochs`
        ----
        PARAMETERS:
         - epochs : should be a non negative integer
         - print_every : if 0 will not print, else will print at given epoch
         - display_metrics : List of metrics returned in the epoch_end stage rdict that has to be 
             displayed, if None (default) all the returned metrics are displayed.
         - train_dl : Will use this instead of DatLoader passed in the constructor call.
         - valid_dl : Will use this instead of DatLoader passed in the constructor call.
         - track_batch_metrics : whether to store the values returned in the batch steps
         - load_best : whether to load the best model after training, works only if validation
             parameters are defined `valid_dl`, `valid_step`, `valid_epoch_end`
         -  continue_loop : Will ask whether to continue training after `continue` epochs; should
             be a positive integer.
         - define_all : If True then `torch.set_grad_enabled`, `optimizer.zero_grad` and model mode 
             ie [train,eval] have to be called where required (usually in the `train_step` function).
         - no_print : If True will suppress all print statements, can be used when custom logging is
             used in the stage functions.
         - no_cast : True, if data casting has to be manually set in the stage functions
         - no_float : True, don't apply float conversion to returned metrics.
         - no_progress : True, don't show the progress bar.
        
        """
        self.__loop(epochs=epochs, print_every=print_every,
                    display_metrics=display_metrics, track_batch_metrics=track_batch_metrics,
                    load_best=load_best, continue_loop=continue_loop, define_all=define_all,
                    no_print=no_print, no_cast=no_cast, no_float=no_float, no_progress=no_progress,
                    train_dl=train_dl, valid_dl=valid_dl
                   )
    
    def train(self, *args, **kwargs):
        """
        Alias for FitLoop.fit
        """
        self.fit(*args, **kwargs)
    
    def test(self, test_dl:Optional[DataLoader]=None,
            no_print:bool=False, no_cast:bool=False, no_float:bool=False, 
            ) -> None:
        """
        For model testing. Runs loop for one epoch using test DataLoader and test stage functions.
        ----
        PARAMETERS:
         - test_dl : Will use this instead of DatLoader passed in the constructor call.
         - no_print : If True will suppress all print statements, can be used when custom logging is
             used in the stage functions.
         - no_cast : True, if model and data casting has to be manually set in the stage functions
         - no_float : True don't apply float conversion to returned metrics.
         - no_progress : True, don't show the progress bar.
        """
        if self.test_dl is None and test_dl is None:
            logging.error("no test_dl, test can't run")
        elif self.test_step is  None:
            logging.error("no test_step, test can't run")
        else:
            self.__loop(is_test=True, no_print=no_print, no_cast=no_cast, no_float=no_float, test_dl=test_dl)
        
    """
    SECTION: 3 - B

    Loop methods for (sort of) unit testing and timing of components.
    """
    def _checkup(self)->None:
        """
        Handles switching attributes pertaining to best model 
        during checkup.
        """
        if self._temp is None:
            self._temp = self.best_model_name
            self.best_model_name = self._temp_model_name
            if self.best_model_state_dict is not None:
                self._temp_state_dict = self.best_model_state_dict
        elif self._temp is not None:
            self.del_best_model()
            self.best_model_name = self._temp
            self.best_model_state_dict = self._temp_state_dict
            self._temp = None
            self._temp_state_dict = None
        else:
            logging.error("unreachable statement from [_checkup] has been reached")


    def run_profiler(self,
            epochs:Optional[int]=1, steps: Optional[int]=None, define_all:bool=False,
            no_cast:bool=False, no_float:bool=False, no_progress:bool=False, print_outcome:bool=True,
            train_dl:Optional[DataLoader]=None,
            valid_dl:Optional[DataLoader]=None,
            test_dl:Optional[DataLoader]=None
            ) -> Optional[Dict[str,Union[Dict[str,List[float]],List[float]]]]:
        """
        NOTE: -- Note Very Accurate -- Need to switch profiling method.
        
        Runs the loop in profiler mode, ie run all three (train, valid, test) phases 
        (if set) for given number of epochs and steps and print the average time taken 
        at different stages, loop output is not printed.
        
        Returns the time_profile dict.
        
        Criteria based checkpointing is not run, ie best_model and best_score are not saved.
        Model state is not altered (it's reloaded) if the profiler is not interrupted.
        ----
        PARAMETERS:
         - epochs : should be a non negative integer
         - steps : number of batches to iterate over in each phase [train,valid,test] 
             to check if everything is working as expected, if None then all batches are
             iterated over.
         - define_all : If True then `torch.set_grad_enabled`, `optimizer.zero_grad` and model mode 
             ie [train,eval] have to be called where required (usually in the `train_step` function).
         - no_cast : True, if data casting has to be manually set in the stage functions
         - no_float : True don't apply float conversion to returned metrics.
         - no_progress : True, don't show the progress bar.
         - print_outcome : If `False` won't print profiler outcome, the values are returned as a dict.
         - train_dl : Will use this instead of DatLoader passed in the constructor call.
         - valid_dl : Will use this instead of DatLoader passed in the constructor call.
         - test_dl : Will use this instead of DatLoader passed in the constructor call.
        """
        if not self.save_to_disk:
            logging.warn("save_to_disk=False; precheck save not possible; profiling aborted")
            return
        
        # Precheck save state and change best model name
        self._precheck_save()

        t1 = time.perf_counter()
        try:
            self.__loop(epochs=epochs,steps=steps, define_all=define_all, no_cast=no_cast, 
                        no_float=no_float, no_print=True, no_progress=no_progress, 
                        train_dl=train_dl, valid_dl=valid_dl, profiler=True)
            if (self.test_dl is not None or test_dl is not None) and (self.test_step is not None):
                self.__loop(steps=steps, define_all=define_all, no_cast=no_cast, 
                            no_float=no_float, no_print=True, no_progress=no_progress, profiler=True, 
                            test_dl=test_dl, is_test=True)
        except Exception as e:
            logging.error(f"error occured: {repr(e)}")
        st = ptime(time.perf_counter() - t1)

        # Postcheck load state and delete and change best model name
        self._postcheck_load()

        if print_outcome:
            self._print_time_profile()
            self.time_profile = {}
            print(f"\ntotal time: {st}")
        else:
            time_profile = self.time_profile
            self.time_profile = {}
            return time_profile
    
    def run_sanity_check(self, epochs:int=1, 
            steps:int=3, print_every:int=1, use_test_dl=False,
            display_metrics:Optional[Union[str,List[str]]]=None,
            continue_loop:int=0, define_all:bool=False,  
            no_print:bool=False, no_cast:bool=False,
            no_float:bool=False, no_progress:bool=False,
            train_dl:Optional[DataLoader]=None,
            valid_dl:Optional[DataLoader]=None,
            test_dl:Optional[DataLoader]=None
           ) -> None:
        """
        Runs the loop in sanity check mode, ie all three (train, valid, test) phases 
        (if set) for given number of epochs and steps.
        Criteria based checkpointing is not run, ie best_model and best_score are not saved.
        Model state is not altered (it's reloaded) if the sanity check is not interrupted.
        ----
        PARAMETERS:
         - epochs : should be a non negative integer
         - steps : number of batches to run in each phase [train,valid] 
             for check if everything is working, if None all batches are iterated over.
         - use_test_dl : If False will use the validation DataLoader for the test phase,
             else will use the test DataLoader.
         - print_every : if 0 will not print, else will print at given epoch
         - display_metrics : List of metrics returned in the epoch_end stage rdict that has to be 
             displayed, if None (default) all the returned metrics are displayed.
         -  continue_loop : Will ask whether to continue training after `continue` epochs, should
             be a positive integer.
         - define_all : If True then `torch.set_grad_enabled`, `optimizer.zero_grad` and model mode 
             ie [train,eval] have to be called where required (usually in the `train_step` function).
         - no_print : If True will suppress all print statements, can be used when custom logging is
             used in the stage functions.
         - no_cast : True, if data casting has to be manually set in the stage functions
         - no_float : True don't apply float conversion to returned metrics.
         - no_progress : True, don't show the progress bar.
         - train_dl : Will use this instead of DatLoader passed in the constructor call.
         - valid_dl : Will use this instead of DatLoader passed in the constructor call.
         - test_dl : Will use this instead of DatLoader passed in the constructor call.
        """
        if not self.save_to_disk:
            logging.warn("save_to_disk=False; precheck save not possible; sanity check aborted")
            return
        
        # Precheck save state and change best model name
        self._precheck_save()
        _steps = 'all' if steps is None else steps
        try:
            print(f"RUNNING SANITY CHECK: TRAIN LOOP - {epochs} EPOCH(s), {_steps} STEP(s)")
            self.__loop(epochs=epochs, steps=steps, print_every=print_every, 
                        display_metrics=display_metrics, continue_loop=continue_loop,
                        define_all=define_all, no_print=no_print, no_cast=no_cast, 
                        no_float=no_float, no_progress=no_progress, 
                        train_dl=train_dl,valid_dl=valid_dl,
                        is_sanity_check=True)
            if (self.test_dl is not None or test_dl is not None and not use_test_dl) and (self.test_step is not None):
                print()
                print(f"RUNNING SANITY CHECK: TEST LOOP - {_steps} STEP(s)")
                self.__loop(use_test_dl=use_test_dl, steps=steps, print_every=print_every, 
                            display_metrics=display_metrics, continue_loop=continue_loop,
                            define_all=define_all, no_print=no_print, no_cast=no_cast, 
                            no_float=no_float, no_progress=no_progress, 
                            test_dl=test_dl,
                            is_sanity_check=True, is_test=True)
        except Exception as e:
            logging.error(f"error occured: {repr(e)}")

        # Postcheck load state and delete and change best model name
        self._postcheck_load()
    
    
    # ---------------------------------------------------------------------
    """
    SECTION: 4
    
    Functions to preserve the model state.
    """
    def _is_save_safe(self, path):
        return not path.exists()

    def _get_path(self, name, default_name, path):

        if name is None: name = default_name
        if path is None: path = self.save_path

        path = Path(path)
        if not path.exists():
            path.mkdir(parents=True)

        if Path(name).suffix != ".pt":
            name += ".pt"
        return path/name
    
    # Internal Use for best model checkpointing
    def __save_best(self) -> None:
        """
        For use with model checkpointing using `FitLoop.criteria`.
        """
        if self.save_to_disk:
            self.save_model(self.best_model_name, override=True)
        else:
            state_dict = deepcopy(self._model.state_dict())
            self.best_model_state_dict = state_dict
        
    def __load_best(self):
        """
        For use with model checkpointing using `FitLoop.criteria`.
        """
        if self.save_to_disk:
            self.load_model(self.best_model_name)
        else:
            state_dict = self.best_model_state_dict
            self._model.load_state_dict(state_dict)
            self.best_model_state_dict = None

            if self.configure_optimizer is None:
                logging.warning("please reconfigure FitLoop.optimizer before training")
            else:
                self.configure_optimizer(self)
    
    # API
    def save_model(self, name:Optional[str]=None, path:Optional[Union[str,Path]]=None, override:bool=False):
        """
        Saves the model's state dict.

        *Note:* Doesn't save `requires_grad` attribute of the parameters,
        this is default behaviour, to save gradient requirement use
        `FitLoop.save`
        ----
        PARAMETERS
        name : name of the model to be saved; default=`"model.pt"`
        path : path where to save the model; default=`FitLoop.save_path`
        override : override save warning if file exists.
        """
        if not self.save_to_disk:
            logging.warning("save_to_disk=False; save model aborted")
            return

        path = self._get_path(name, self.model_name, path)

        if not self._is_save_safe(path) and not override:
            logging.warning("save aborted; file exists; to save set override=True")
            return

        torch.save(self.model.state_dict(), path)
    
    def load_model(self, name:Optional[str]=None, path:Optional[Union[str,Path]]=None, configure_optimizer:bool=True, 
            map_location:torch.device=None, override:bool=False):
        """
        Loads the model's state dict.
        ----
        PARAMETERS
        name : name of the model to be loaded; default=`"model.pt"`
        path : path where to save the model; default=`FitLoop.save_path`
        configure_optimizer : calls `FitLoop.configure_optimizer` if `True` to set param_groups
        map_location : kwarg passed to torch.load, defaults to `self.device`
        override : load state despite `save_to_disk=False`
        """
        if not self.save_to_disk or override:
            logging.warning("save_to_disk=False; load model failed")
            return

        path = self._get_path(name, self.model_name, path)
        if map_location is None: map_location = self.device
        self.model.load_state_dict(torch.load(path,map_location=map_location))
        if self.configure_optimizer is None or not configure_optimizer:
            logging.warning("please reconfigure FitLoop.optimizer before training")
        else:
            self.configure_optimizer(self)

        
    # ---------------------------------------------------------------------
    """
    SECTION: 5
    
    Functions to preserve the FitLoop object state so that training can be resumed.
    """
    
    def save(self, name=None, path=None, override=False):
        """
        Saves the state of components in a fitloop, 
        includes: optimizer, model, lr_scheduler states.

        *Note:* Since the state dicts of `optimizer` and `lr_scheduler` are saved
        if either is changed after loading eg: before loading `SGD` then switch to
        `Adam` the load won't be able to revert this change and will try to load
        the new Objects with the state dict. So before loading set the same `optimizer`
        and `lr_scheduler`.
        ----
        PARAMETERS
        name : name of the fitloop state to be saved; default=`"state.pt"`
        path : path where to save the state; default=`FitLoop.save_path`
        override : override save warning if file exists.
        """
        if not self.save_to_disk:
            logging.warning("save_to_disk=False; save failed")
            return
        path = self._get_path(name, self.state_name, path)

        if not self._is_save_safe(path) and not override:
            logging.warning("save aborted; file exists; to save set override=True")
            return

        sd = {}
        sd["model"] = self.model.state_dict()
        sd["unlocked"] = len([*get_layers(self.model,True)])
        sd["best_score"] = self.best_score
        sd["epoch_num"] = self.epoch_num
        sd["best_model_name"] = self.best_model_name
        sd["metrics"] = self.metrics
        if isinstance(self.optimizer,list):
            sd["optimizer"] = []
            for opt in self.optimizer:
                sd["optimizer"].append(opt.state_dict())
        else:
            sd["optimizer"] = self.optimizer.state_dict()
        if self.lr_scheduler is not None:
            if isinstance(self.lr_scheduler,list):
                sd["lr_scheduler"] = []
                for sch in self.lr_scheduler:
                    sd["lr_scheduler"].append(sch.state_dict())
            else:
                sd["lr_scheduler"] = self.lr_scheduler.state_dict()
        else:
            sd["lr_scheduler"] = None
        
        torch.save(sd, path)
            
    def load(self, name=None, path=None, map_location=None, override=False):
        """
        Loads the state of components in a fitloop from path.

        *Note:* Since the state dicts of `optimizer` and `lr_scheduler` are saved
        if either is changed after loading eg: before loading `SGD` then switch to
        `Adam` the load won't be able to revert this change and will try to load
        the new Objects with the state dict. So before loading set the same `optimizer`
        and `lr_scheduler`.
        ----
        PARAMETERS
        name : name of the fitloop state to be loaded; default=`"state.pt"`
        path : path from where to load the state; default=`FitLoop.save_path`
        map_location : kwarg passed to torch.load; default=`FitLoop.device`
        override : load state despite `save_to_disk=False`
        """
        if not self.save_to_disk or override:
            logging.warning("save_to_disk=False; load failed")
            return

        path = self._get_path(name, self.state_name, path)
        if map_location is None: map_location = self.device

        sd = torch.load(path, map_location=map_location)
        unlocked = sd["unlocked"] 
        self.model.load_state_dict(sd["model"])
        self.best_score = sd["best_score"]
        self.epoch_num = sd["epoch_num"]
        self.best_model_name = sd["best_model_name"]
        self.metrics = sd["metrics"]
        try:
            if isinstance(self.optimizer,list):
                for optsd,opt in zip(sd["optimizer"],self.optimizer):
                    opt.load_state_dict(optsd)
            else:
                self.optimizer.load_state_dict(sd["optimizer"])
        except Exception as e:
            logging.error(f"error while loading optimizer state_dict: {repr(e)}")
        if sd["lr_scheduler"] is not None:
            if self.lr_scheduler is not None:
                try:
                    if isinstance(self.lr_scheduler,list):
                        for schsd,sch in zip(sd["lr_scheduler"],self.lr_scheduler):
                            sch.load_state_dict(schsd)
                    else:
                        self.lr_scheduler.load_state_dict(sd["lr_scheduler"])
                except Exception as e:
                    logging.error(f"error while loading lr_scheduler state_dict: {repr(e)}")
            else:
                logging.warning("""lr_scheduler state dict found in saved state
                but FitLoop.lr_scheduler is None, please set lr_scheduler 
                and load again to load lr_scheduler state dict.""")
        self.configure_optimizer(self, unlock=unlocked)
            
    def _precheck_save(self):
        # To be used before checkup runs.
        self.save(self._temp_state_name,override=True)
        self._checkup()
    
    def _postcheck_load(self):
        # To be used after checkup runs.
        self._checkup()
        self.load(self._temp_state_name)
        path = self._get_path(None, self._temp_state_name, None)
        try:
            path.unlink() # Delete temp file
        except:
            pass
        
    
    # ---------------------------------------------------------------------
    """
    SECTION: 6
    
    Functions to delete stored model weights.
    """
    def delete(self, name, path=None):
        """
        Deletes the model/state (.pt files).
        ----
        PARAMETERS
        name : name of the file to be loaded; no defaults to prevent erroneous deletion.
        path : path of the file to be deleted; default=`FitLoop.save_path`
        """
        path = self._get_path(name, self.state_name, path)
        path.unlink()
    
    def del_best_model(self) -> None:
        """
        Deletes the best model state dict from the disk if 
        `save_to_disk` else sets attribute to None
        """
        if self.save_to_disk:
            self.delete(self.best_model_name)
        else:
            self.best_model_state_dict = None
    
    # ---------------------------------------------------------------------
    """
    SECTION: 7
    
    Getters for metrics
    """
    @property
    def M(self):
        return self.metrics
    
    @property
    def train_metrics(self):
        return self.metrics.train
    
    @property
    def valid_metrics(self):
        return self.metrics.valid
    
    @property
    def test_metrics(self):
        return self.metrics.test
    
    @property
    def plot(self):
        return self.metrics.plot