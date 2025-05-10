import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import pdb
import os
import time
import copy
import random

import argparse
import logging

from peft import get_peft_config, get_peft_model, LoraConfig

import giws.utils as ddp_utils
from giws.CFIT.model.model import TwitterClassifier
from giws.CFIT.model.dataset import GTDataset
from giws.CFIT.scripts.eval_CFIT import eval
from giws.CFIT.model import utils, hook

def setup_model(args):
    model = TwitterClassifier(
            name=args.get('name','clip'),
            img_input_size=args.img_input_size, 
            text_input_size=args.text_input_size, 
            use_cross_attn=args.get('use_cross_attn', False))
    if args.apply_lora:
        lora_config = args.lora_config
        peft_config = LoraConfig(
                    r=lora_config.get('lora_r', 4), 
                    lora_alpha=lora_config.get('lora_alpha', 32),
                    lora_dropout=lora_config.get('lora_dropout', 0.1),
                    bias=lora_config.get('lora_bias', 'none'),
                    target_modules=['query', 'value'],
                    modules_to_save=lora_config.get('modules_to_save', 'classifier')
                )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    model.to(ddp_utils.get_device(args.gpu_list))
    logging.info(model)
    logging.info('Model setup finish')
    return model.train()

def setup_dataset(args, processer):
    img_folder, text_folder, label_folder = tuple([os.path.join(args.train_data_folder, i) for i in ['img', 'text', 'label']])
    train_dataset = GTDataset(img_folder, text_folder, label_folder, processer)
    
    if args.eval:
        img_folder, text_folder, label_folder = tuple([os.path.join(args.test_data_folder, i) for i in ['img', 'text', 'label']])
        test_dataset = GTDataset(img_folder, text_folder, label_folder, processer)
        
        # shuffle to prevent bias
        if args.shuffle:
            all_data = copy.deepcopy(train_dataset.data) + copy.deepcopy(test_dataset.data)
            split_index = int(len(all_data) * 0.8)
            random.shuffle(all_data)
            train_dataset.data = all_data[:split_index]
            test_dataset.data = all_data[split_index:]

        # k-fold cross-validation
        if args.k_fold:
            all_data = copy.deepcopy(train_dataset.data) + copy.deepcopy(test_dataset.data)
            random.shuffle(all_data)
            j = len(all_data) // args.folds
            i = args.fold_index if args.fold_index < args.folds and args.fold_index >=0 else args.folds - 1
            train_dataset.data = all_data[:i*j] + all_data[i*j+j:]
            test_dataset.data = all_data[i*j:i*j+j]

        test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=True, num_workers=0, pin_memory=True)
    else:
        test_dataloader = None
    # distributed sampler
    sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=True)
    train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.train_batch_size,
            sampler=sampler,
            num_workers=2 * ddp_utils.get_world_size(),
            pin_memory=True
    )
    logging.info(f'Dataloader setup finish: train {train_dataset.get_class_num()}, test {test_dataset.get_class_num()}')
    return train_dataloader, test_dataloader

def train(args):
    device = ddp_utils.get_device(args.gpu_list)
    model = setup_model(args)
    clip_model, preprocess = utils.get_clip(device=device)
    train_dataloader, test_dataloader = setup_dataset(args, preprocess)

    # training trick setting
    accumulate_step = args.get('accumulate_step', 4)
    amp = args.get('amp', False)
    
    # loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    max_grad_norm = args.get('max_grad_norm', 1.0)
    cur_step = 0
    batch_start_time = time.time()
    all_batch_length = len(train_dataloader) // accumulate_step
    batch_loss = []

    # eval setting
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
    if amp:
        scaler = torch.cuda.amp.GradScaler()

    for epoch in range(args.epochs):
        optimizer.zero_grad()
        for batch, (image, text, label) in enumerate(train_dataloader):
            it = (batch + 1) % accumulate_step

            # propagation
            if amp:
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    output = model(image.to(device), text.to(device))
                    loss = criterion(output, label.to(device)) / accumulate_step
                
                scaler.scale(loss).backward()
            else:
                output = model(image.to(device), text.to(device))
                loss = criterion(output, label.to(device)) / accumulate_step
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

                if args.eval and cur_step % args.eval_step == 0 and device==0:
                    logging.info('Begin to eval......')
                    result_file = os.path.join(args.output_dir, f'result_{cur_step}.json')
                    ave_accuracy, confusion_matrix, precision, recall, f1_score = eval(model, args.test_data_folder, args.test_batch_size, result_file, dataloader=test_dataloader, is_return=True)
                    logging.info(f'Eval finished. Average Accuracy: {round(ave_accuracy, 4)}, Confusion Matrix:{confusion_matrix}')
                    logging.info(f'precision:{precision}, recall:{recall}, f1_score:{f1_score}')
                    # save the best eval ckpt
                    if ave_accuracy > best_acc:
                        save_checkpoint(cur_step, best=True)
                        best_acc = ave_accuracy


                batch_start_time = time.time()
                batch_loss = []
                # pdb.set_trace()
    model.gradient_checker.close()



