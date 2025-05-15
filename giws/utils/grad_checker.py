import torch
import logging

class GradientChecker:
    def __init__(self, model, verbose=True, raise_on_nan=False):
        self.model = model
        self.verbose = verbose
        self.raise_on_nan = raise_on_nan
        self.logger = logging.getLogger(__name__)

    def check_gradients(self):
        has_invalid = False
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                if not torch.isfinite(param.grad).all():
                    has_invalid = True
                    if self.verbose:
                        self.logger.warning(
                            f"[GradientChecker] Invalid gradient in `{name}` "
                            f"(min={param.grad.min().item()}, max={param.grad.max().item()})"
                        )
        if has_invalid and self.raise_on_nan:
            raise RuntimeError("[GradientChecker] Found NaN or Inf in gradients.")
        return not has_invalid
