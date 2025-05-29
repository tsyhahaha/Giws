import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

import os
import time
import logging

import numpy as np

from giws.models import PTBModel
from giws.data import load_ptb_data, PTBDataset
from giws.optim import (
    build_scheduled_optim
)
from giws.utils import (
    dispatch_clip_grad,
    get_save_func,
)

logger = logging.getLogger(__name__)

def setup_model(args):
    model = PTBModel(**args.model)
    model.to(args.gpu_id)
    logger.info(model)
    logger.info('Model setup finish')
    return model.train()

def setup_dataset(args):
    data_path = args.get("data_path", None)
    assert data_path is not None, "Please specify the data path in the configuration file"

    # get tokenized data for training and validation
    train_data, valid_data, word2idx, idx2word = load_ptb_data(args.data_path)

    train_dataset = PTBDataset(train_data, args.batch_size, args.seq_len)

    if args.eval:
        test_dataset = PTBDataset(valid_data, args.batch_size, args.seq_len)
    else:
        test_dataset = None

    train_loader = DataLoader(train_dataset, batch_size=1, pin_memory=True,)
    test_loader = DataLoader(test_dataset, batch_size=1, pin_memory=True) if test_dataset else None

    logger.info(f"LM Vocabulary size: {len(word2idx)}")
    logger.info(f'Dataloader setup finish: train {len(train_dataset)}\t \
                 eval {len(test_dataset) if test_dataset else 0}')
    return train_loader, test_loader, word2idx, idx2word

@torch.no_grad()
def test(model, device, test_dataloader):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for input_ids, labels in test_dataloader:
        input_ids = input_ids[0].long().transpose(1, 0).contiguous().to(device)
        labels = labels[0].long().transpose(1, 0).contiguous().to(device)

        logits, _ = model(input_ids)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            reduction='sum'
        )
        total_loss += loss.item()
        total_tokens += labels.numel()

    avg_loss = total_loss / total_tokens
    ppl = torch.exp(torch.tensor(avg_loss))
    return avg_loss, ppl.item()


def train_func(args):
    device = args.gpu_id
    model = setup_model(args)
    train_dataloader, test_dataloader, word2idx, idx2word = setup_dataset(args)

    # loss and optimizer
    optimizer = torch.optim.AdamW(model.parameters(), args.lr)
    scheduled_optim = build_scheduled_optim(args.schedule_type, optimizer, **args.optim)
    max_grad_norm = args.get('clip_grad_value', 1.0)
    all_batch_length = len(train_dataloader) 

    # initial vars
    best_indicator = 0
    save_checkpoint = get_save_func(args, model)
    if device == 0:
        save_checkpoint(cur_step=0, cur_epoch=0, best_indicator=best_indicator)

    for epoch in range(1, args.epochs+1):
        for batch, (input_ids, label) in enumerate(train_dataloader):
            scheduled_optim.zero_grad()
            batch_start_time = time.time()

            input_ids = input_ids[0].long().transpose(1,0).contiguous().to(device)
            label = label[0].long().transpose(1,0).contiguous().to(device)
            # forward/backward propagation
            logits, (h_t, c_t) = model(input_ids)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), label.view(-1))
            loss.backward()
            
            # gradients clip
            if args.get("clip_grad", False):
                dispatch_clip_grad(model.parameters(), value=max_grad_norm, mode=args.clip_mode)

            scheduled_optim.step()
        
            batch_end_time = time.time()
            logger.info(f'optim step = {scheduled_optim.get_step()} '
                        f'loss = {round(loss.item(), 4)} '
                        f'lr = {scheduled_optim.get_lr()}')
            logger.info(f'Epoch [{epoch}/{args.epochs}] Batch [{batch+1}/{all_batch_length}] time {round(batch_end_time - batch_start_time, 4)} s.')

        # save checkpoints
        if epoch % args.save_interval == 0 or epoch == args.epochs:
            save_checkpoint(cur_step=scheduled_optim.get_step(),
                            cur_epoch=epoch, best_indicator=best_indicator)
        # eval by epoch interval
        if args.eval and (epoch % args.eval_interval == 0 and \
                test_dataloader is not None \
                or epoch == args.epochs):
            logger.info(f'Epoch [{epoch}/{args.epochs}] Begin to test......')
            ave_loss, ppl = test(model, device, test_dataloader)
            logger.info(f"Test finished: test_loss = {ave_loss}, ppl = {ppl}")

            if ppl < best_indicator:    # the lower ppl is, the better.
                best_indicator = ppl
                save_checkpoint(cur_step=scheduled_optim.get_step(),
                            cur_epoch=epoch, best_indicator=best_indicator, best=True)
            
            model.train()
                

