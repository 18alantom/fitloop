from fitloop.helpers.state import LoopState
from fitloop.helpers.constants import Dict

class DefaultEpochEnd:
    """
    Default functions for the epoch_end step.
    """
    batch_step_criteria = "running_correct"
    batch_step_loss = "running_loss"
    criteria = "accuracy"
    loss = "loss"
    """
    Section : EPOCH END STEPS FUNCTIONS
    """
    @staticmethod
    def _common_epoch_end(state: LoopState) -> Dict[str, float]:
        return {
            DefaultEpochEnd.criteria: state[DefaultEpochEnd.batch_step_criteria]\
                .sum().float().item() / state.size,
            DefaultEpochEnd.loss:state[DefaultEpochEnd.batch_step_loss]\
                .sum().float().item() / state.size
        }

    @staticmethod
    def train_epoch_end(state: LoopState) -> Dict[str, float]:
        return DefaultEpochEnd._common_epoch_end(state)

    @staticmethod
    def valid_epoch_end(state: LoopState) -> Dict[str, float]:
        return DefaultEpochEnd._common_epoch_end(state)

    @staticmethod
    def test_epoch_end(state: LoopState) -> Dict[str, float]:
        return DefaultEpochEnd._common_epoch_end(state)