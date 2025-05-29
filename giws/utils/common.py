import os
import torch
from functools import partial

def get_save_func(args, model):
    def save_checkpoint(args, model, cur_epoch, cur_step, best_indicator, best=False):
        ckpt_dir = os.path.join(args.output_dir, 'checkpoints')
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        
        save_info = dict(
            model = model.state_dict(),
            cfg=args.model,
            train_steps=cur_step,
            best_indicator=best_indicator
        )
        if not best:
            ckpt_file = os.path.join(ckpt_dir, f'checkpoint_epoch_{cur_epoch}.ckpt')
        else:
            save_info['best_indicator'] = best_indicator
            ckpt_file = os.path.join(ckpt_dir, f'best_checkpoint_epoch_{cur_epoch}.ckpt')
        torch.save(save_info, ckpt_file)
    return partial(save_checkpoint, args=args, model=model)