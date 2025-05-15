from abc import ABC, abstractmethod

class BaseScheduledOptim(ABC):
    """Abstract base class for scheduled optimizers."""
    def __init__(self, optimizer):
        self._optimizer = optimizer
        self.n_steps = 0

    def step(self):
        self.n_steps += 1
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        self._optimizer.zero_grad()

    def get_lr(self):
        return self._optimizer.param_groups[0]['lr']

    def get_step(self):
        return self.n_steps

    def get_optim(self):
        return self._optimizer

    @abstractmethod
    def _update_learning_rate(self):
        pass