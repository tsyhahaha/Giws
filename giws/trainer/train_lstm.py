import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

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

    train_loader = DataLoader(data, 
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=2)


    return train_loader, None

def test(model, device, test_loader):
    if test_loader is None:
        return np.nan
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')
    return accuracy

def train_func(args):
    device = args.gpu_id
    model = setup_model(args)
    train_dataloader, test_dataloader = setup_dataset(args)

    # loss and optimizer
    optimizer = optim.AdamW(model.parameters(), args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
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
        for batch, data in enumerate(train_dataloader):
            batch_start_time = time.time()
            data = data.long().transpose(1,0).contiguous()
            data = data.to(device)
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
            logger.info(f'Epoch [{epoch+1}/{args.epochs}] Batch [{batch+1}/{all_batch_length}] time {round(batch_end_time - batch_start_time, 4)} s.')
            cur_step += 1
        scheduler.step()

        # save checkpoints
        if epoch % args.save_interval == 0 or epoch == args.epochs:
            save_checkpoint(cur_step)
        # eval by epoch interval
        if (args.eval and epoch % args.eval_interval == 0 and device==0) or epoch == args.epochs - 1:
            logger.info(f'Epoch [{epoch+1}/{args.epochs}] Begin to test......')
            ave_accuracy = test(model, device, test_dataloader)
            if ave_accuracy > best_acc:
                save_checkpoint(cur_step, best=True)
                best_acc = ave_accuracy
                

