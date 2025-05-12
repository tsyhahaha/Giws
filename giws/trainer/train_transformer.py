import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import GradScaler, autocast
from contextlib import nullcontext

from nltk.translate.bleu_score import sentence_bleu, corpus_bleu

import os
import time
import logging
from functools import partial

import giws.utils as ddp_utils
from giws.models import Transformer
from giws.data import TranslationDataset
from giws.trainer import dispatch_clip_grad

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
    return model.train()

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

def test(args, model, device, test_dataloader, trg_word2idx):
    model.eval()
    total_samples = 0
    total_bleu = 0.0
    trg_idx2word = {v: k for k, v in trg_word2idx.items()}
    
    # 准备特殊token的索引
    pad_idx = trg_word2idx['<pad>']
    eos_idx = trg_word2idx['<eos>']
    
    # 存储所有参考和预测句子
    all_references = []
    all_hypotheses = []
    
    with torch.no_grad():
        for batch in test_dataloader:
            src = batch['src'].to(device)
            tgt = batch['trg'].to(device)
            
            outputs = model(src, tgt[:, :-1], 
                            use_efficient_attn=args.use_efficient_attn)
            predictions = outputs.argmax(dim=-1)
            for i in range(predictions.size(0)):
                ref_indices = []
                for idx in tgt[i].tolist():
                    if idx == eos_idx:
                        break
                    if idx != pad_idx:
                        ref_indices.append(idx)
                ref_sentence = [trg_idx2word[idx] for idx in ref_indices]
                
                hyp_indices = []
                for idx in predictions[i].tolist():
                    if idx == eos_idx:
                        break
                    if idx != pad_idx:
                        hyp_indices.append(idx)
                hyp_sentence = [trg_idx2word[idx] for idx in hyp_indices]
                
                all_references.append([ref_sentence])  # 注意BLEU需要references是列表的列表
                all_hypotheses.append(hyp_sentence)
    
    # 计算BLEU分数 (使用简化版的BLEU计算，不引入外部库)
    bleu_scores = corpus_bleu(all_references, all_hypotheses, weights=(0.25, 0.25, 0.25, 0.25))
    
    model.train()
    return bleu_scores

def train_func(args):
    device = ddp_utils.get_device(args.gpu_list) if args.use_gpu else 'cpu'
    model = setup_model(args)
    train_dataloader, test_dataloader = setup_dataset(args)
    word2idx = train_dataloader.dataset.get_word2idx(target='trg')

    amp_enabled = args.get("amp_enabled", False)

    # loss and optimizer
    scaler = GradScaler(enabled=amp_enabled)
    context = autocast('cuda') if amp_enabled  else nullcontext()
    optimizer = optim.AdamW(model.parameters(), args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_dataloader))
    max_grad_norm = args.get('clip_grad_value', 1.0)
    all_batch_length = len(train_dataloader) 

    # initial vars
    cur_step = 0
    best_acc = 0

    def save_checkpoint(iter, best=False):
        ckpt_dir = os.path.join(args.output_dir, 'checkpoints')
        if not os.path.exists(ckpt_dir):
            os.path.makedirs(ckpt_dir)
        
        save_info = dict(
            model = model.state_dict(),
            cfg=args,
            train_steps=cur_step,
            best_acc=best_acc
        )
        if not best:
            ckpt_file = os.path.join(ckpt_dir, f'checkpoint_step_{iter}.ckpt')
        else:
            save_info['best_acc'] = best_acc
            ckpt_file = os.path.join(ckpt_dir, f'best_checkpoint.ckpt')
        torch.save(save_info, ckpt_file)

    # save initial model
    save_checkpoint(0)

    for epoch in range(args.epochs):
        optimizer.zero_grad()
        for batch, encoded_input in enumerate(train_dataloader):
            batch_start_time = time.time()

            # forward/backward propagation
            with context:
                encoded_input = {k: v.to(device) for k,v in encoded_input.items() if v is not None}
                output = model(
                    src=encoded_input['src'],   # [batch_size, max_seq_len]
                    trg=encoded_input['trg'][:, :-1],   # [batch_size, max_seq_len]
                    use_efficient_attn=args.use_efficient_attn
                )
                loss = F.cross_entropy(
                    output.view(-1, output.size(-1)),
                    encoded_input['trg'][:, 1:].contiguous().view(-1),
                    ignore_index=word2idx['<pad>']  # 如果使用padding_idx=0
                )

            # gradients clip
            scaler.scale(loss).backward()
            if args.get("clip_grad", False):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
        
            batch_end_time = time.time()
            logging.info(f'optim step = {cur_step+1} loss = {round(loss.item(), 4)}')
            logging.info(f'Epoch [{epoch+1}/{args.epochs}] Batch [{batch+1}/{all_batch_length}] time {round(batch_end_time - batch_start_time, 4)} s.')
            cur_step += 1
            
        # save checkpoints
        if epoch % args.save_interval == 0 or epoch == args.epochs:
            save_checkpoint(cur_step)
        # eval by epoch interval
        if (args.eval and epoch % args.eval_interval == 0 and device == 0) or epoch == args.epochs - 1:
            logging.info(f'Epoch [{epoch+1}/{args.epochs}] Beginning to test......')
            bleu_score = test(args, model, device, test_dataloader, word2idx)
            logging.info(f'Epoch [{epoch+1}/{args.epochs}] Test finished, bleu score: {bleu_score}')

        
                



