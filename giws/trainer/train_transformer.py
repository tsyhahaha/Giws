import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.amp import GradScaler, autocast
from contextlib import nullcontext


import os
import time
import logging
from functools import partial

import giws.utils.ddp_utils as ddp_utils
from giws.utils import get_save_func
from giws.optim import (
    CustomScheduledOptim,
    ConstantScheduledOptim,
    build_scheduled_optim,
)
from giws.utils import(
    GradientChecker,
    dispatch_clip_grad,
)
from giws.models import Transformer
from giws.data import TranslationDataset

logger = logging.getLogger(__name__)

def cal_performance(pred, gold, trg_pad_idx, smoothing=False):
    ''' Apply label smoothing if needed '''

    pred = pred.view(-1, pred.size(-1))

    loss = F.cross_entropy(
                    pred,
                    gold,
                    ignore_index=trg_pad_idx,
                    label_smoothing=0.1 if smoothing else 0.0,
                )
    
    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
    non_pad_mask = gold.ne(trg_pad_idx)
    n_correct = pred.eq(gold).masked_select(non_pad_mask).sum().item()
    n_word = non_pad_mask.sum().item()

    return loss, n_correct, n_word


def patch_trg(trg):
    trg, gold = trg[:, :-1], trg[:, 1:].contiguous().view(-1)
    return trg, gold

def setup_model(args):
    device = ddp_utils.get_device(args.gpu_list) if args.use_gpu else 'cpu'
    
    extra_cfg = None
    if args.get('model_path', None) is not None:
        save_info = torch.load('args.model_path', map_location='cpu')
        model = Transformer(
            **save_info['cfg'],
            max_length=args.max_len,
            device=device,
        )
        model.load_state_dict(save_info["model"])
        extra_cfg = dict(
            cur_step = save_info['train_steps'],
            best_indicator = save_info['best_indicator'],
        )
    else:
        model = Transformer(
            **args.model, 
            max_length=args.max_len,
            device=device,
        )

    model.to(device)
    logger.info(model)
    logger.info('Model setup finish')
    return model, extra_cfg

def collate_fn(batch):
    processed_batch = {}
    for key in batch[0].keys():
        items = [item[key] for item in batch if item[key] is not None]
        if items:
            processed_batch[key] = torch.stack(items)
        else:
            processed_batch[key] = None
    return processed_batch

def setup_dataset(args):
    train_dataset = TranslationDataset(
        chinese_file=os.path.join(args.data_path, 'chinese.txt'),
        english_file=os.path.join(args.data_path, 'english.txt'),
        chinese_vocab_file=os.path.join(args.data_path, 'chinese_vocab.json'),
        english_vocab_file=os.path.join(args.data_path, 'english_vocab.json'),
        max_len=args.max_len,
    )

    # distributed sampler
    sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=False)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=2 * ddp_utils.get_world_size(),
        pin_memory=True,
        collate_fn=collate_fn,
        shuffle=False,
    )

    if args.eval and ddp_utils.get_local_rank() == 0:
        test_dataset = TranslationDataset(
            chinese_file=os.path.join(args.eval_data_path, 'cn.txt'),
            english_file=os.path.join(args.eval_data_path, 'en.txt'),
            chinese_vocab_file=os.path.join(args.data_path, 'chinese_vocab.json'),
            english_vocab_file=os.path.join(args.data_path, 'english_vocab.json'),
            max_len=args.max_len,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            num_workers=2,
            pin_memory=True,
            collate_fn=collate_fn,
        )
    else:
        test_dataset = None
        test_loader = None

    logger.info(f'Dataloader setup finish: train {len(train_dataset)}\t \
                 test {len(test_dataset) if test_dataset else 0}')
    return train_dataloader, test_loader

@torch.no_grad()
def test(model, validation_data, device, pad_idx=[0,0]):
    ''' Epoch operation in evaluation phase '''
    model.eval()
    total_loss, n_word_total, n_word_correct = 0, 0, 0

    for batch in validation_data:
        # prepare data
        src_seq = batch['src'].to(device)
        trg_seq, gold = map(lambda x: x.to(device), patch_trg(batch['trg']))

        # forward
        pred = model(src_seq, trg_seq)
        loss, n_correct, n_word = cal_performance(
            pred, gold, pad_idx[1], smoothing=False)

        # note keeping
        n_word_total += n_word
        n_word_correct += n_correct
        total_loss += loss.item()

    loss_per_word = total_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    ppl = torch.exp(torch.tensor(loss_per_word))
    return loss_per_word, ppl.item(), accuracy


def train_func(args):
    # model setup
    device = ddp_utils.get_device(args.gpu_list) if args.use_gpu else 'cpu'
    model, extra_cfg = setup_model(args)    # extra_cfg: cfg for warm start
    model.train()

    # dataset setup
    train_dataloader, test_dataloader = setup_dataset(args)
    word2idx = train_dataloader.dataset.get_word2idx(target='trg')

    # loss and optimizer
    amp_enabled = args.get("amp_enabled", False)
    scaler = GradScaler(enabled=amp_enabled)
    context = autocast('cuda') if amp_enabled  else nullcontext()
    optimizer = optim.AdamW(model.parameters(), **args.optim)
    scheduled_optim = build_scheduled_optim(
        args.scheduler_type, 
        optimizer, d_model = args.model.embed_dim,
        **args.scheduler,
    )
    
    max_grad_norm = args.get('clip_grad_value', 1.0)
    all_batch_length = len(train_dataloader)

    # initialization
    best_indicator = 0 if extra_cfg is None else extra_cfg['best_indicator']
    save_checkpoint = get_save_func(args, model)
    
    gradient_checker = GradientChecker(model, verbose=True, raise_on_nan=False)
    if device == 0:
        save_checkpoint(cur_step=0, cur_epoch=0, best_indicator=best_indicator)


    # begin training
    for epoch in range(1, args.epochs+1):
        for batch, encoded_input in enumerate(train_dataloader):
            scheduled_optim.zero_grad()
            batch_start_time = time.time()

            # forward/backward propagation
            with context:
                encoded_input = {k: v.to(device) for k,v in encoded_input.items() if v is not None}
                output = model(
                    src=encoded_input['src'],   # [batch_size, max_seq_len]
                    trg=encoded_input['trg'][:, :-1],   # [batch_size, max_seq_len]
                    use_efficient_attn=args.use_efficient_attn,
                )
                loss = F.cross_entropy(
                    output.view(-1, output.size(-1)),
                    encoded_input['trg'][:, 1:].contiguous().view(-1),
                    ignore_index=word2idx['<pad>'],
                    label_smoothing=args.get('label_smoothing', 0.0),
                )
            scaler.scale(loss).backward()

            # gradients clip
            if args.get("clip_grad", False):
                scaler.unscale_(scheduled_optim.get_optim())
                dispatch_clip_grad(model.parameters(), max_grad_norm)

            # gradients checker
            grad_valid = gradient_checker.check_gradients()
            grad_valid_tensor = torch.tensor(int(grad_valid), device=device)
            dist.all_reduce(grad_valid_tensor, op=dist.ReduceOp.MIN)
            if grad_valid_tensor.item() == 0:
                logger.warning("Skipping step due to invalid gradient")
                scheduled_optim.zero_grad()
                continue


            # gradients update
            scaler.step(scheduled_optim.get_optim())
            scaler.update()
            scheduled_optim.step()
            batch_end_time = time.time()
            # logger information
            logger.info(f'optim step = {scheduled_optim.get_step()} '
                        f'lr = {scheduled_optim.get_lr()} '
                        f'loss = {round(loss.item(), 4)}')
            logger.info(f'Epoch [{epoch}/{args.epochs}] '
                        f'Batch [{batch+1}/{all_batch_length}] '
                        f'time {round(batch_end_time - batch_start_time, 4)} s.')
            
        # save checkpoints by interval
        if (epoch % args.save_interval == 0 or epoch == args.epochs) and device == 0:
            save_checkpoint(cur_epoch=epoch, cur_step=scheduled_optim.get_step(), best_indicator=best_indicator)
        
        # eval by interval
        if (args.eval and epoch % args.eval_interval == 0) or epoch == args.epochs:
            if device == 0:
                logger.info(f'Epoch [{epoch}/{args.epochs}] Beginning to test......')
                val_loss, ppl, acc = test(model, test_dataloader, device)
                logger.info(f'Epoch [{epoch}/{args.epochs}] Test finished, '
                            f'ppl = {round(ppl,4)}, '
                            f'test_loss = {round(val_loss, 4)}, '
                            f'acc = {round(acc, 3)}')

                if acc > best_indicator:
                    best_indicator = acc
                    save_checkpoint(cur_epoch=epoch, cur_step=scheduled_optim.get_step(), 
                                    best_indicator=best_indicator, best=True)
                    
            model.train()

        
                



