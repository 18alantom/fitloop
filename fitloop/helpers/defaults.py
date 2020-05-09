from .state import LoopState
from .constants import Dict

class FitLoopDefaults:
    """
    Some default functions to be used with FitLoop,
    allows fit loop to work without having to define 
    anything.
    
    Doesn't use any lr_schedulers and just a single optimizer
    """
    batch_step_criteria = "running_correct"
    batch_step_loss = "running_loss"
    criteria = "accuracy"
    loss = "loss"
    
    """
    Section 1. BATCH STEP FUNCTIONS
    """
    @staticmethod
    def _common_batch_step(state: LoopState) -> Dict[str, float]:
        X, y = state.batch
        y_ = state.model(X)
        loss = state.loss_function(y_, y)

        return loss, {
            FitLoopDefaults.batch_step_criteria:
            (y_.argmax(dim=1) == y).sum().float().item(),
            FitLoopDefaults.batch_step_loss: (loss.item() * state.batch_size)
        }

    @staticmethod
    def train_step(state: LoopState) -> Dict[str, float]:
        loss, rdict = FitLoopDefaults._common_batch_step(state)
        loss.backward()
        state.optimizer.step()
        return rdict

    @staticmethod
    def valid_step(state: LoopState) -> Dict[str, float]:
        _, rdict = FitLoopDefaults._common_batch_step(state)
        return rdict

    @staticmethod
    def test_step(state: LoopState) -> Dict[str, float]:
        _, rdict = FitLoopDefaults._common_batch_step(state)
        return rdict

    
    """
    Section 2. EPOCH END STEPS FUNCTIONS
    """
    @staticmethod
    def _common_epoch_end(state: LoopState) -> Dict[str, float]:
        return {
            FitLoopDefaults.criteria: state[FitLoopDefaults.batch_step_criteria]\
                .sum().float().item() / state.size,
            FitLoopDefaults.loss:state[FitLoopDefaults.batch_step_loss]\
                .sum().float().item() / state.size
        }

    @staticmethod
    def train_epoch_end(state: LoopState) -> Dict[str, float]:
        return FitLoopDefaults._common_epoch_end(state)

    @staticmethod
    def valid_epoch_end(state: LoopState) -> Dict[str, float]:
        return FitLoopDefaults._common_epoch_end(state)

    @staticmethod
    def test_epoch_end(state: LoopState) -> Dict[str, float]:
        return FitLoopDefaults._common_epoch_end(state)