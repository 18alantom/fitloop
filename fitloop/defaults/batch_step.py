from fitloop.helpers.state import LoopState
from fitloop.helpers.constants import Dict

class DefaultBatchStep:
    """
    Default functions for the batch step.
    """
    batch_step_criteria = "running_correct"
    batch_step_loss = "running_loss"
    """
    Section : BATCH STEP FUNCTIONS

    """
    @staticmethod
    def _common_batch_step(state: LoopState) -> Dict[str, float]:
        X, y = state.batch
        y_ = state.model(X)
        loss = state.loss_function(y_, y)

        return loss, {
            DefaultBatchStep.batch_step_criteria:
            (y_.argmax(dim=1) == y).sum().float().item(),
            DefaultBatchStep.batch_step_loss: (loss.item() * state.batch_size)
        }

    @staticmethod
    def train_step(state: LoopState) -> Dict[str, float]:
        loss, rdict = DefaultBatchStep._common_batch_step(state)
        loss.backward()
        state.optimizer.step()
        if state.lr_scheduler is not None:
            state.lr_scheduler.step()
        return rdict

    @staticmethod
    def valid_step(state: LoopState) -> Dict[str, float]:
        _, rdict = DefaultBatchStep._common_batch_step(state)
        return rdict

    @staticmethod
    def test_step(state: LoopState) -> Dict[str, float]:
        _, rdict = DefaultBatchStep._common_batch_step(state)
        return rdict