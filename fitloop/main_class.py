import time
import math
import warnings
import torch

from uuid import uuid4
from pathlib import Path
from copy import deepcopy
from tqdm.autonotebook import tqdm

from fitloop.helpers.state import LoopState
from fitloop.helpers.helpers import ftime, ptime
from fitloop.helpers.metrics import MetricsAggregator
from fitloop.helpers.defaults import FitLoopDefaults
from .main_constants import *

class FitLoop:
    """
    FitLoop trains Pytorch models.
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
                 train_step: Callable[[LoopState],Dict[str, Any]]=FitLoopDefaults.train_step,
                 valid_step: Optional[Callable[[LoopState],Dict[str, Any]]]=FitLoopDefaults.valid_step,
                 test_step: Optional[Callable[[LoopState],Dict[str, Any]]]=FitLoopDefaults.test_step,
                 
                 # Epoch Start Step
                 train_epoch_start: Optional[Callable[[LoopState],Dict[str, Any]]]=None,
                 valid_epoch_start: Optional[Callable[[LoopState],Dict[str, Any]]]=None,
                 test_epoch_start: Optional[Callable[[LoopState],Dict[str, Any]]]=None,
                 
                 # Epoch End Step
                 train_epoch_end: Callable[[LoopState],Dict[str, Any]]=FitLoopDefaults.train_epoch_end,
                 valid_epoch_end: Optional[Callable[[LoopState],Dict[str, Any]]]=FitLoopDefaults.valid_epoch_end,
                 test_epoch_end: Optional[Callable[[LoopState],Dict[str, Any]]]=FitLoopDefaults.test_epoch_end,
                 
                 # Other Stage Functions
                 preloop: Optional[Callable[[dict],None]]=None,
                 postloop: Optional[Callable[[dict],None]]=None,
                 
                 # Other Args
                 lr_scheduler: Optional[Union[LRScheduler, Any, List[Union[LRScheduler,Any]]]]=None,
                 device: torch.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), 
                 configure_optimizer:Callable[[object],None]=None,
                 dtype: torch.dtype=torch.float32,
                 
                 # Model Evaluation
                 criteria: Optional[str]=None,
                 criteria_direction: int=1,
                 
                 # Model Preservation
                 save_to_disk: bool=False,
                 save_path: str="models",
                 pretrained_model_name: Optional[str]=None,
                 best_model_name: Optional[str]=None,
                ) -> None:
        """
        FitLoop constructor
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
            - save_to_disk : True then save pretrained and best_model to the disk, else it is 
                stored as an attribute.
            - save_path : location where the initial and pretrained models are to be saved
            - pretrained_model_name : Name to save the pretrained model by
            - best_model_name : Name to save the best model by
        """
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
        if pretrained_model_name is None:
            u = str(uuid4()).split('-')[1]
            pretrained_model_name = f"pretrained_{u}.pt"
        if best_model_name is None:
            u = str(uuid4()).split('-')[1]
            best_model_name = f"best_{u}.pt"
        self.pretrained_model_name = pretrained_model_name
        self.best_model_name = best_model_name
        self.save_to_disk = save_to_disk
        self.save_path = Path(save_path)
        if self.save_to_disk and not self.save_path.exists():
            self.save_path.mkdir()
        
        # INITIALIZE NON ARGS
        self.best_model_state_dict = None
        self.pretrained_model_state_dict = None
        self.epoch_num = 0
        self.best_score = self.criteria_direction * float('-inf')
        self.time_profile = {}
        self.metrics = MetricsAggregator()
        
        # Change criteria if defaults are being used
        if self.valid_step is FitLoopDefaults.valid_step:
            self.criteria = FitLoopDefaults.criteria
            
        # Basic Blocks - Calling model setter
        self.model = model
    
    @property
    def model(self):
        return self._model
    
    @model.setter
    def model(self, model):
        self._model = model.to(device=self.device, dtype=self.dtype)
        self.__save_model(self._PR)
    
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
        
        if step_func is None:
            raise AttributeError(f"{phase}_step not assigned")
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
            return None
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
            raise AttributeError(f"{phase}_end_step not assigned")
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
                
    def print_time_profile(self):
        if len(self.time_profile) == 0:
            print("please run FitLoop.run_profiler(print_outcome=False) first")
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
        print("NOT IMPLEMENTED YET")
        pass
        
        
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
        self.__save_model(self._BS)
        
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
         
        # PROFILER STATEMENT ---------
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
            mt = ' | '.join([f"{m}: {epoch_metrics[m]:0.4f}"for m in epoch_metrics])
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
                if not (is_tr or is_test or profiler or is_sanity_check) and self.criteria is not None:
                    score = state[phase]._get_epoch_metric(self.criteria)
                    direc = self.criteria_direction > 0
                    is_better = (score > self.best_score) if direc else (score < self.best_score)
                    if is_better:
                        self.best_score = score
                        self.__save_model(self._BS)

                # PRINT EPOCH[PHASE] METRICS
                epoch_metrics = state[phase]._get_epoch_metrics(display_metrics)
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
        if load_best or profiler or is_sanity_check:
            self.__load_model(self._BS)
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
        self.__loop(is_test=True, no_print=no_print, no_cast=no_cast, no_float=no_float, test_dl=test_dl)
        
    """
    SECTION: 3 - B
    
    Loop methods for (sort of) unit testing and timing of components.
    """
    def run_profiler(self,
            epochs:Optional[int]=1, steps: Optional[int]=None, define_all:bool=False,
            no_cast:bool=False, no_float:bool=False, no_progress:bool=False, print_outcome:bool=True,
            train_dl:Optional[DataLoader]=None,
            valid_dl:Optional[DataLoader]=None,
            test_dl:Optional[DataLoader]=None
            ) -> Dict[str,Union[Dict[str,List[float]],List[float]]]:
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
         - print_outcome : If False won't print profiler outcome, can be accesed from FitLoop.time_profile
         - train_dl : Will use this instead of DatLoader passed in the constructor call.
         - valid_dl : Will use this instead of DatLoader passed in the constructor call.
         - test_dl : Will use this instead of DatLoader passed in the constructor call.
        """
        t1 = time.perf_counter()
        self.__loop(epochs=epochs,steps=steps, define_all=define_all, no_cast=no_cast, 
                    no_float=no_float, no_print=True, no_progress=no_progress, 
                    train_dl=train_dl, valid_dl=valid_dl, profiler=True)
        if self.test_dl is not None or test_dl is not None:
            self.__loop(steps=steps, define_all=define_all, no_cast=no_cast, 
                        no_float=no_float, no_print=True, no_progress=no_progress, profiler=True, 
                        test_dl=test_dl, is_test=True)
        st = ptime(time.perf_counter() - t1)
        if print_outcome:
            self.print_time_profile()
            time_profile = self.time_profile
            self.time_profile = {}
            print(f"\ntotal time: {st}")
            return time_profile
        else:
            return self.time_profile
    
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
    
        
        print(f"RUNNING SANITY CHECK: TRAIN LOOP - {epochs} EPOCH(s), {steps} STEP(s)")
        self.__loop(epochs=epochs, steps=steps, print_every=print_every, 
                    display_metrics=display_metrics, continue_loop=continue_loop,
                    define_all=define_all, no_print=no_print, no_cast=no_cast, 
                    no_float=no_float, no_progress=no_progress, 
                    train_dl=train_dl,valid_dl=valid_dl,
                    is_sanity_check=True)
        if self.test_dl is not None or test_dl is not None:
            print()
            print(f"RUNNING SANITY CHECK: TEST LOOP - {steps} STEP(s)")
            self.__loop(use_test_dl=use_test_dl, steps=steps, print_every=print_every, 
                        display_metrics=display_metrics, continue_loop=continue_loop,
                        define_all=define_all, no_print=no_print, no_cast=no_cast, 
                        no_float=no_float, no_progress=no_progress, 
                        test_dl=test_dl,
                        is_sanity_check=True, is_test=True)
    
    
    # ---------------------------------------------------------------------
    """
    SECTION: 4
    
    Functions to preserve the model state.
    """
    
    def __save_model(self, typ:str) -> None:
        """
        Save model to object or to the disk.
        """
        name = self.best_model_name if typ == self._BS else self.pretrained_model_name
        path = self.save_path/ name
        state_dict = deepcopy(self._model.state_dict())
        if self.save_to_disk:
            torch.save(state_dict, path)
            
        elif typ == self._BS:
            self.best_model_state_dict = state_dict
        elif typ == self._PR:
            self.pretrained_model_state_dict = state_dict
        else:
            logging.warning("model save failed")
        
    def __load_model(self, typ:str):
        """
        Load model from the object or from the disk.
        """
        name = self.best_model_name if typ == self._BS else self.pretrained_model_name
        path = self.save_path/ name
        if self.save_to_disk:
            state_dict = torch.load(path, map_location=self.device)
        elif typ == self._BS:
            state_dict = self.best_model_state_dict
        else:
            state_dict = self.pretrained_model_state_dict
        self._model.load_state_dict(state_dict)
        if self.configure_optimizer is None:
            print("please reconfigure FitLoop.optimizer before training")
        else:
            self.configure_optimizer(self)
    
    def reset(self, reset_model:bool=True) -> None:
        """
        Resets FitLoop to initial state.
        Parameters reset:
            - model, to pretrained state if `reset_model`
            - epoch_num, to 0
            - best_score to ∓inf
        FitLoop.optimizer param groups will have to be set again
        """
        if reset_model:
            self.__load_model(self._PR)
        self.epoch_num = 0
        self.best_score = self.criteria_direction * float('-inf')
        self.metrics = MetricsAggregator()
        
        
    # ---------------------------------------------------------------------
    """
    SECTION: 5
    
    Functions to preserve the FitLoop object state so that training can be resumed.
    """
    
    def save(self, path, only_model=False):
        """
        TODO : save the FitLoop state, if only_model then save only model.
        """
        print("NOT IMPLEMENTED YET")
        pass
    
    def load(self, path):
        """
        TODO : load the FitLoop state, if only model then load the model 
            state dict.
        """
        print("NOT IMPLEMENTED YET")
        pass
    
    
    # ---------------------------------------------------------------------
    """
    SECTION: 6
    
    Functions to delete stored model weights.
    """
    
    def del_pretrained_model(self) -> None:
        """
        Deletes the pretrianed model state dict from the disk if 
        `save_to_disk` else states attribute to None
        """
        if self.save_to_disk:
            (self.save_path/self.pretrained_model_name).unlink()
        else:
            self.pretrained_model_state_dict = None
        
    def del_best_model(self) -> None:
        """
        Deletes the best model state dict from the disk if 
        `save_to_disk` else states attribute to None
        """
        if self.save_to_disk:
            (self.save_path/self.best_model_name).unlink()
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
