from giws.optim.base import BaseScheduledOptim
import math

class CustomScheduledOptim(BaseScheduledOptim):
    """Ref: Attention is all you need."""
    def __init__(self, optimizer, lr_mul, d_model, warmup_steps):
        super().__init__(optimizer)
        self.lr_mul = lr_mul
        self.d_model = d_model
        self.n_warmup_steps = warmup_steps

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
    def __init__(self, optimizer, total_steps, min_lr=1e-6, is_cycle=False):
        super().__init__(optimizer)
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.is_cycle = is_cycle
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]

    def _update_learning_rate(self):
        if self.n_steps > self.total_steps and self.is_cycle:
            return
        for i, param_group in enumerate(self._optimizer.param_groups):
            base_lr = self.base_lrs[i]
            lr = self.min_lr + 0.5 * (base_lr - self.min_lr) * \
                (1 + math.cos(math.pi * self.n_steps / self.total_steps))
            param_group['lr'] = lr


class CosineAnnealingWarmRestartsScheduledOptim(BaseScheduledOptim):
    def __init__(self, optimizer, T_0, T_mult=1, min_lr=1e-6):
        super().__init__(optimizer)
        self.T_0 = T_0
        self.T_mult = T_mult
        self.min_lr = min_lr
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]

        self.T_i = T_0
        self.T_cur = 0

    def _update_learning_rate(self):
        for i, param_group in enumerate(self._optimizer.param_groups):
            base_lr = self.base_lrs[i]
            # cosine scheduler
            lr = self.min_lr + 0.5 * (base_lr - self.min_lr) * \
                (1 + math.cos(math.pi * self.T_cur / self.T_i))
            param_group['lr'] = lr

        # 每步更新状态
        self.T_cur += 1
        if self.T_cur >= self.T_i:
            self.T_cur = 0
            self.T_i *= self.T_mult



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


class ConstantScheduledOptim(BaseScheduledOptim):
    def __init__(self, optimizer, factor=0.3333333333333333, total_iters=0,):
        super().__init__(optimizer)
        self.lr = self._optimizer.param_groups[0]['lr']
        self.total_iters = total_iters
        self.factor = factor

    def _update_learning_rate(self):
        if self.n_steps >= self.total_iters:
            lr = self.lr
        else:
            lr = self.factor * self.lr
            
        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr