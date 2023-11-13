import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import pdb
import os
import time
import copy
import random
from .model import utils
from .model import hook
import argparse
import logging

from peft import get_peft_config, get_peft_model, LoraConfig

from model.model import TwitterClassifier
from model.dataset import GTDataset
from eval_CFIT import eval
""" 
def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_batch_size', type=int, default=128)
    parser.add_argument('--test_batch_size', type=int, default=128)
    parser.add_argument('--train_data_folder', type=str)
    parser.add_argument('--test_data_folder', type=str)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--accumulate-step', type=int, default=4)
    parser.add_argument('--eval-step', type=int, default=1000)
    parser.add_argument('--save-step', type=int, default=1000)
    parser.add_argument('-lr', '--learning-rate', type=float, default=1e-3)
    parser.add_argument('--output-dir', type=str, default='./output')
    parser.add_argument('--eval', action='store_true', help='whether to eval model')
    parser.add_argument('--amp', action='store_true', help='whether to use auto mixed pricision')
    # parser.add_argument('')
    return parser.parse_args()
"""

def setup(args):
    os.makedirs(os.path.abspath(os.path.join(args.output_dir, 'checkpoints')), exist_ok=True)
    os.makedirs(os.path.abspath(os.path.join(args.output_dir, 'logs')), exist_ok=True)

    log_file = os.path.abspath(os.path.join(args.output_dir, 'logs', f'train.log'))

    level = logging.INFO
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

    logging.info('----------------------------------------')
    logging.info(f'Arguments: {args}')
    logging.info(f'Logical batch size {args.accumulate_step * args.train_batch_size}')
    logging.info('----------------------------------------')

    device = "cuda" if torch.cuda.is_available else "cpu"
    args.device = device

def setup_model(args):
    if args.apply_lora:
        model = TwitterClassifier(input_size=args.input_size)
        lora_config = args.lora_config
        peft_config = LoraConfig(
                    r=lora_config.get('lora_r', 4), 
                    lora_alpha=lora_config.get('lora_alpha', 32),
                    lora_dropout=lora_config.get('lora_dropout', 0.1),
                    bias=lora_config.get('lora_bias', 'none'),
                    target_modules=['query', 'value'],
                    modules_to_save=['classifier']
                )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        model.to(args.device)
    else:
        model = TwitterClassifier(input_size=args.input_size).to(args.device)
    logging.info('Model setup finish')
    return model

def setup_dataset(args, processer):
    img_folder, text_folder, label_folder = tuple([os.path.join(args.train_data_folder, i) for i in ['img', 'text', 'label']])
    train_dataset = GTDataset(img_folder, text_folder, label_folder, processer)
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=0, pin_memory=True)
    if args.eval:
        img_folder, text_folder, label_folder = tuple([os.path.join(args.test_data_folder, i) for i in ['img', 'text', 'label']])
        test_dataset = GTDataset(img_folder, text_folder, label_folder, processer)
        
        # shuffle to prevent bias
        split_index = len(train_dataset.data)
        all_data = copy.deepcopy(train_dataset.data) + copy.deepcopy(test_dataset.data)
        random.shuffle(all_data)
        train_dataset.data = all_data[:split_index]
        test_dataset.data = all_data[split_index:]

        test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=True, num_workers=0, pin_memory=True)
    else:
        test_dataloader = None
    logging.info('Dataloader setup finish')
    return train_dataloader, test_dataloader

def train(args):
    model = setup_model(args)
    clip_model, preprocess = utils.get_clip(device=args.device)
    train_dataloader, test_dataloader = setup_dataset(args, preprocess)

    # loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    max_grad_norm = args.get('max_grad_norm', 1.0)
    cur_step = 0
    batch_start_time = time.time()
    all_batch_length = len(train_dataloader) // args.accumulate_step
    batch_loss = []

    accumulate_step = args.get('accumulate_step', 4)
    amp = args.get('amp', False)

    def save_checkpoint(iter):
        ckpt_dir = os.path.join(args.output_dir, 'checkpoints')
        if not os.path.exists(ckpt_dir):
            os.path.makedirs(ckpt_dir)
        
        ckpt_file = os.path.join(ckpt_dir, f'checkpoint_step_{iter}.ckpt')

        torch.save(dict(
            model = model.state_dict(),
            cfg=args,
            train_steps = cur_step), ckpt_file
        )

    # save initial model
    save_checkpoint(0)
    if amp:
        scaler = torch.cuda.amp.GradScaler()

    for epoch in range(args.epochs):
        optimizer.zero_grad()
        for batch, (image, text, label) in enumerate(train_dataloader):
            it = (batch + 1) % args.accumulate_step

            # propagation
            # output = model(image.to(args.device), text.to(args.device))
            if amp:
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    output = model(image.to(args.device), text.to(args.device))
                    loss = criterion(output, label.to(args.device)) / accumulate_step
                
                scaler.scale(loss).backward()
            else:
                output = model(image.to(args.device), text.to(args.device))
                loss = criterion(output, label.to(args.device)) / accumulate_step
                loss.backward()
            batch_loss.append(loss.item())
            # torch.cuda.empty_cache()
            
            # gradient accumulation
            if it == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                jt = (batch + 1) // accumulate_step    # real batch
                logging.info(f'optim step = {cur_step+1} loss = {round(sum(batch_loss)/len(batch_loss), 4)}')
                batch_end_time = time.time()
                logging.info(f'Epoch [{epoch+1}|{args.epochs}] Batch [{jt}/{all_batch_length}] time {round(batch_end_time - batch_start_time, 4)} s.')

                if amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                cur_step += 1
                optimizer.zero_grad()

                if cur_step % args.save_step == 0:
                    save_checkpoint(cur_step)

                if args.eval and cur_step % args.eval_step == 0:
                    logging.info('Begin to eval......')
                    result_file = os.path.join(args.output_dir, f'result_{cur_step}.json')
                    ave_accuracy = eval(model, args.test_data_folder, args.test_batch_size, result_file, dataloader=test_dataloader, is_return=True)
                    logging.info(f'Eval finished. Average Accuracy: {round(ave_accuracy, 4)}')

                batch_start_time = time.time()
                batch_loss = []
                # pdb.set_trace()
    model.gradient_checker.close()



