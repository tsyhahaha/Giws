from giws.optim.base import BaseScheduledOptim
import math

class CustomScheduledOptim(BaseScheduledOptim):
    """Ref: Attention is all you need."""
    def __init__(self, optimizer, lr_mul, d_model, n_warmup_steps):
        super().__init__(optimizer)
        self.lr_mul = lr_mul
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps

    def _get_lr_scale(self):
        return (self.d_model ** -0.5) * min(
            self.n_steps ** -0.5,
            self.n_steps * self.n_warmup_steps ** -1.5
        )

    def _update_learning_rate(self):
        lr = self.lr_mul * self._get_lr_scale()
        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr


class CosineAnnealingScheduledOptim(BaseScheduledOptim):
    def __init__(self, optimizer, total_steps, min_lr=1e-6):
        super().__init__(optimizer)
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]

    def _update_learning_rate(self):
        for i, param_group in enumerate(self._optimizer.param_groups):
            base_lr = self.base_lrs[i]
            lr = self.min_lr + 0.5 * (base_lr - self.min_lr) * \
                (1 + math.cos(math.pi * self.n_steps / self.total_steps))
            param_group['lr'] = lr


class StepLRScheduledOptim(BaseScheduledOptim):
    def __init__(self, optimizer, step_size, gamma=0.1):
        super().__init__(optimizer)
        self.step_size = step_size
        self.gamma = gamma

    def _update_learning_rate(self):
        if self.n_steps % self.step_size == 0:
            for param_group in self._optimizer.param_groups:
                param_group['lr'] *= self.gamma


class LinearScheduledOptim(BaseScheduledOptim):
    def __init__(self, optimizer, total_steps, start_lr, end_lr):
        super().__init__(optimizer)
        self.total_steps = total_steps
        self.start_lr = start_lr
        self.end_lr = end_lr

    def _update_learning_rate(self):
        progress = min(1.0, self.n_steps / self.total_steps)
        lr = self.start_lr + progress * (self.end_lr - self.start_lr)
        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr
