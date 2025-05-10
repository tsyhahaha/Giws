import torch

class GradientChecker:
    def __init__(self, model):
        self.model = model

        # 注册反向传播钩子 register_full_backward_hook
        self.hooks = []
        self.register_hooks(model)

    def check_gradients(self, module, grad_input, grad_output):
        # 在这里检查梯度，记录包含 NaN 的位置
        for idx, grad in enumerate(grad_input):
            if grad is not None and torch.isnan(grad).any():
                print(f"NaN gradient detected in module {module.__class__.__name__}, input index {idx}")

    def register_hooks(self, module):
        hook = module.register_full_backward_hook(self.check_gradients)
        self.hooks.append(hook)
         
        # 递归注册所有子模块的反向传播钩子
        for child in module.children():
            self.register_hooks(child)

    def close(self):
        # 移除钩子
        for hook in self.hooks:
            hook.remove()
