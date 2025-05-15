import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.amp import GradScaler, autocast
from contextlib import nullcontext

from nltk.translate.bleu_score import corpus_bleu

import os
import time
import logging
from functools import partial

import giws.utils as ddp_utils
from giws.optim import ScheduledOptim, dispatch_clip_grad
from giws.models import Transformer
from giws.data import TranslationDataset

def cal_performance(pred, gold, trg_pad_idx, smoothing=False):
    ''' Apply label smoothing if needed '''

    pred = pred.view(-1, pred.size(-1))

    loss = loss = F.cross_entropy(
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
    model = Transformer(
        **args.model, 
        max_length=args.max_len,
        device=device,
    )
    model.to(device)
    logging.info(model)
    logging.info('Model setup finish')
    return model

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
    sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=True)
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

    logging.info(f'Dataloader setup finish: train {len(train_dataset)}\t test {len(test_dataset) if test_dataset else 0}')
    return train_dataloader, test_loader


def test(model, validation_data, device, pad_idx=[0,0]):
    ''' Epoch operation in evaluation phase '''
    model.eval()
    total_loss, n_word_total, n_word_correct = 0, 0, 0

    with torch.no_grad():
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
    return loss_per_word, accuracy


def train_func(args):
    device = ddp_utils.get_device(args.gpu_list) if args.use_gpu else 'cpu'
    model = setup_model(args)
    model.train()
    train_dataloader, test_dataloader = setup_dataset(args)
    word2idx = train_dataloader.dataset.get_word2idx(target='trg')

    amp_enabled = args.get("amp_enabled", False)

    # loss and optimizer
    scaler = GradScaler(enabled=amp_enabled)
    context = autocast('cuda') if amp_enabled  else nullcontext()
    scheduled_optim = ScheduledOptim(optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-09), 
                               lr_mul=args.get('lr_mul', 1.),
                               d_model=args.model.embed_dim,
                                n_warmup_steps=args.warmup_steps)
    max_grad_norm = args.get('clip_grad_value', 1.0)
    all_batch_length = len(train_dataloader)

    # initial vars
    cur_step = 0
    best_indicator = 0

    def save_checkpoint(iter, best=False):
        ckpt_dir = os.path.join(args.output_dir, 'checkpoints')
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        
        save_info = dict(
            model = model.state_dict(),
            cfg=args,
            train_steps=cur_step,
            best_indicator=best_indicator
        )
        if not best:
            ckpt_file = os.path.join(ckpt_dir, f'checkpoint_step_{iter}.ckpt')
        else:
            save_info['best_indicator'] = best_indicator
            ckpt_file = os.path.join(ckpt_dir, f'best_checkpoint.ckpt')
        torch.save(save_info, ckpt_file)

    # save initial model
    if device == 0:
        save_checkpoint(0)

    for epoch in range(args.epochs):
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

            # gradients clip
            scaler.scale(loss).backward()
            if args.get("clip_grad", False):
                scaler.unscale_(scheduled_optim.get_optim())
                dispatch_clip_grad(model.parameters(), max_grad_norm)
            
            scaler.step(scheduled_optim.get_optim())
            scaler.update()
            scheduled_optim.step_and_update_lr()
            batch_end_time = time.time()

            # logging information
            torch.distributed.barrier()
            lr = scheduled_optim.get_lr()
            logging.info(f'optim step = {cur_step+1} lr = {lr} loss = {round(loss.item(), 4)}')
            logging.info(f'Epoch [{epoch+1}/{args.epochs}] Batch [{batch+1}/{all_batch_length}] time {round(batch_end_time - batch_start_time, 4)} s.')
            cur_step += 1
            
        # save checkpoints
        if (epoch % args.save_interval == 0 or epoch == args.epochs) and device == 0:
            save_checkpoint(cur_step)
        # eval by epoch interval
        if (args.eval and epoch % args.eval_interval == 0) or epoch == args.epochs - 1:
            if device == 0:
                logging.info(f'Epoch [{epoch+1}/{args.epochs}] Beginning to test......')
                val_loss, acc = test(model, test_dataloader, device)
                logging.info(f'Epoch [{epoch+1}/{args.epochs}] Test finished, loss: {val_loss}, acc: {acc}')

                if acc > best_indicator:
                    best_indicator = acc
                    save_checkpoint(cur_step, best=True)

        
                



