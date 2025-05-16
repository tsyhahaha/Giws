import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ConstantLR
from torch.utils.data import DataLoader, TensorDataset, random_split

import os
import time
import logging

import numpy as np

from giws.models import PoetryModel
from giws.utils import dispatch_clip_grad

logger = logging.getLogger(__name__)

def setup_model(args):
    model = PoetryModel(**args.model)
    model.to(args.gpu_id)
    logger.info(model)
    logger.info('Model setup finish')
    return model.train()

def setup_dataset(args):
    data_path = args.get("data_path", None)
    assert data_path is not None, "Please specify the data path in the configuration file"

    datas = np.load(data_path, allow_pickle=True)
    data = torch.from_numpy(datas['data'])

    ix2word = datas['ix2word'].item()
    word2ix = datas['word2ix'].item()

    logger.info(f"Vocab size: {len(word2ix)}")

    dataset = TensorDataset(data)

    train_ratio = args.get("train_ratio", 0.9)
    train_size = int(len(dataset) * train_ratio)
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=2)

    test_loader = DataLoader(test_dataset,
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=2)

    return train_loader, test_loader

@torch.no_grad()
def test(model, device, test_dataloader):
    model.eval()
    total_correct = 0
    total_count = 0

    for batch in test_dataloader:
        data = batch[0].long().transpose(1, 0).contiguous().to(device)  # [seq_len, batch_size]
        input_ids, target = data[:-1, :], data[1:, :]   # Predict next token

        output = model(input_ids)   # output: [seq_len-1, batch_size, vocab_size]
        logits = output[0] if isinstance(output, tuple) else output # unpack if needed

        pred = logits.argmax(dim=-1)    # [seq_len-1, batch_size]

        correct = (pred == target).sum().item()
        total = target.numel()

        total_correct += correct
        total_count += total

    accuracy = total_correct / total_count
    logger.info(f"Test Accuracy: {round(accuracy * 100, 3)}%")

    return accuracy


def train_func(args):
    device = args.gpu_id
    model = setup_model(args)
    train_dataloader, test_dataloader = setup_dataset(args)

    # loss and optimizer
    optimizer = optim.AdamW(model.parameters(), args.lr)
    scheduler = ConstantLR(optimizer, factor=1.)
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

    for epoch in range(1, args.epochs+1):
        optimizer.zero_grad()
        for batch, data in enumerate(train_dataloader):
            batch_start_time = time.time()
            data = data[0].long().transpose(1,0).contiguous().to(device)
            input_ids, target = data[:-1,:], data[1:,:]
            # forward/backward propagation
            output = model(input_ids)
            loss = F.cross_entropy(output[0], target.view(-1))
            loss.backward()
            
            # gradients clip
            if args.get("clip_grad", False):
                dispatch_clip_grad(model.parameters(), value=max_grad_norm, mode=args.clip_mode)

            optimizer.step()
            optimizer.zero_grad()
        
            batch_end_time = time.time()
            logger.info(f'optim step = {cur_step+1} loss = {round(loss.item(), 4)}')
            logger.info(f'Epoch [{epoch}/{args.epochs}] Batch [{batch+1}/{all_batch_length}] time {round(batch_end_time - batch_start_time, 4)} s.')
            cur_step += 1
        scheduler.step()

        # save checkpoints
        if epoch % args.save_interval == 0 or epoch == args.epochs:
            save_checkpoint(cur_step)
        # eval by epoch interval
        if (args.eval and epoch % args.eval_interval == 0 and \
                test_dataloader is not None) \
                or epoch == args.epochs:
            logger.info(f'Epoch [{epoch}/{args.epochs}] Begin to test......')
            ave_accuracy = test(model, device, test_dataloader)
            if ave_accuracy > best_acc:
                save_checkpoint(cur_step, best=True)
                best_acc = ave_accuracy
                

