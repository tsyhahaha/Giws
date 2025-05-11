import os
import logging
import time
import hydra
from omegaconf import DictConfig, OmegaConf
from datetime import datetime

import torch

from giws.inferer import inference_func_lstm

logger = None

class WorkerLogFilter(logging.Filter):
    def __init__(self, rank=-1):
        super().__init__()
        self._rank = rank

    def filter(self, record):
        if self._rank != -1:
            record.msg = f'Rank {self._rank} | {record.msg}'
        return True

def setup(cfg):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg.output_dir = os.path.join(cfg.output_dir, timestamp)
    os.makedirs(os.path.abspath(os.path.join(cfg.output_dir, 'checkpoints')), exist_ok=True)
    os.makedirs(os.path.abspath(os.path.join(cfg.output_dir, 'logs')), exist_ok=True)

    current_timestamp = time.time()
    local_time = time.localtime(current_timestamp)
    formatted_time = time.strftime('%m-%d_%H:%M', local_time)
    log_file = os.path.abspath(os.path.join(cfg.output_dir, 'logs', f'{formatted_time}.log'))

    level = logging.DEBUG if cfg.verbose else logging.INFO
    # fmt = f'%(asctime)-15s [%(levelname)s] (%(filename)s:%(lineno)d) Rank {world_rank} | %(message)s'
    fmt = f'%(asctime)-15s [%(levelname)s] | %(message)s'

    def _handler_apply(h):
        h.setLevel(level)
        h.setFormatter(logging.Formatter(fmt, datefmt='%Y-%m-%d %H:%M:%S'))
        #h.addFilter(WorkerLogFilter(world_rank),)
        return h

    handlers = [
        logging.StreamHandler(),
        logging.FileHandler(log_file),
        ]

    handlers = list(map(_handler_apply, handlers))

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        format=fmt,
        level=level,
        handlers=handlers)
    
    OmegaConf.save(cfg, os.path.join(cfg.output_dir, 'config.yaml'))

    logging.info('-----------------')
    logging.info(f'Arguments: {cfg}')
    logging.info('-----------------')

    logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="config", config_name="base")
def main(cfg : DictConfig):
    setup(cfg)

    target = cfg.get('target', None)
    if target == "lstm":
        inference_func_lstm(cfg)
    else:
        raise NotImplementedError(f'train mode {target} not implemented')
    

if __name__ == '__main__':
    main()
