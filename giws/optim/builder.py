

import inspect
from giws.optim.scheduled_optim import *

SCHEDULED_OPTIMIZERS = {
    "custom": CustomScheduledOptim,
    "cosine": CosineAnnealingScheduledOptim,
    "step": StepLRScheduledOptim,
    "linear": LinearScheduledOptim,
    "constant": ConstantScheduledOptim,
    "cosine_restart": CosineAnnealingWarmRestartsScheduledOptim
}

def build_scheduled_optim(name, optimizer, **kwargs):
    cls = SCHEDULED_OPTIMIZERS.get(name.lower())
    if cls is None:
        raise ValueError(f"Unknown scheduler name: {name}, please choose from {SCHEDULED_OPTIMIZERS.keys()}")

    # 获取调度器构造函数的参数
    sig = inspect.signature(cls.__init__)
    valid_keys = set(sig.parameters.keys()) - {"self", "optimizer"}
    
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}

    return cls(optimizer, **filtered_kwargs)
