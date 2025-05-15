import os
import logging
import time
import hydra
from omegaconf import DictConfig, OmegaConf
from datetime import datetime

import torch

from giws.trainer import (
    train_func_minst, 
    train_func_vit, 
    train_func_lstm, 
    train_func_transformer,
)
from giws.utils import ddp_utils

logger = None

class WorkerLogFilter(logger.Filter):
    def __init__(self, rank=-1):
        super().__init__()
        self._rank = rank

    def filter(self, record):
        if self._rank != -1:
            record.msg = f'Rank {self._rank} | {record.msg}'
        return True

def setup(cfg):
    ddp_utils.setup_ddp()
    ddp_utils.setup_seed(cfg.seed)
    world_rank = ddp_utils.get_world_rank()
    local_rank = ddp_utils.get_local_rank()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg.output_dir = os.path.join(cfg.output_dir, timestamp)
    os.makedirs(os.path.abspath(os.path.join(cfg.output_dir, 'checkpoints')), exist_ok=True)
    os.makedirs(os.path.abspath(os.path.join(cfg.output_dir, 'logs')), exist_ok=True)

    current_timestamp = time.time()
    local_time = time.localtime(current_timestamp)
    formatted_time = time.strftime('%m-%d_%H:%M', local_time)
    log_file = os.path.abspath(os.path.join(cfg.output_dir, 'logs', f'{world_rank}.log'))

    level = logger.DEBUG if cfg.verbose else logger.INFO
    # fmt = f'%(asctime)-15s [%(levelname)s] (%(filename)s:%(lineno)d) Rank {world_rank} | %(message)s'
    fmt = f'%(asctime)-15s [%(levelname)s] Rank {world_rank} | %(message)s'

    def _handler_apply(h):
        h.setLevel(level)
        h.setFormatter(logger.Formatter(fmt, datefmt='%Y-%m-%d %H:%M:%S'))
        #h.addFilter(WorkerLogFilter(world_rank),)
        return h

    handlers = [
        logger.FileHandler(log_file),
    ]
    if world_rank == 0:
        handlers.append(logger.StreamHandler())
    

    handlers = list(map(_handler_apply, handlers))

    for handler in logger.root.handlers[:]:
        logger.root.removeHandler(handler)

    logger.basicConfig(
        format=fmt,
        level=level,
        handlers=handlers)
    
    OmegaConf.save(cfg, os.path.join(cfg.output_dir, 'config.yaml'))

    logger.info('-----------------')
    logger.info(f'Arguments: {cfg}')
    logger.info('-----------------')

    logger.info(f'torch.distributed.init_process_group: world_rank={world_rank}, local_rank={local_rank}')
    logger = logging.getLogger(__name__)

def cleanup(cfg):
    torch.distributed.destroy_process_group()

@hydra.main(version_base=None, config_path="config", config_name="base")
def main(cfg : DictConfig):
    setup(cfg)

    target = cfg.get('target', None)
    if target == 'minst':
        train_func_minst(cfg)
    elif target == "vit":
        train_func_vit(cfg)
    elif target == "lstm":
        train_func_lstm(cfg)
    elif target == "transformer":
        train_func_transformer(cfg)
    else:
        raise NotImplementedError(f'train mode {target} not implemented')
    
    cleanup(cfg)

if __name__ == '__main__':
    main()
