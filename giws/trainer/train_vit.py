import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import GradScaler, autocast
from contextlib import nullcontext

from torchvision import datasets, transforms

import os
import time
import logging

from giws.models import ViT
from giws.optim import dispatch_clip_grad


def setup_model(args):
    model = ViT(**args.model)
    model.to(args.gpu_id)
    logging.info(model)
    logging.info('Model setup finish')
    return model.train()



def get_train_transforms(args):
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.Resize(args.model.image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean = (0.4914, 0.4822, 0.4465),
                             std = (0.2023, 0.1994, 0.2010))
    ])

def get_test_transforms(args):  
    return transforms.Compose([
        transforms.Resize(args.model.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean = (0.4914, 0.4822, 0.4465),
                             std = (0.2023, 0.1994, 0.2010))
    ])

def setup_dataset(args):
    data_path = args.get("data_path", "../data")
    train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(data_path, train=True, download=True,
                       transform=get_train_transforms(args)),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True
        )
    
    if args.eval:
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(data_path, train=False,
                    transform=get_test_transforms(args)),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
    else:
        test_loader = None
        
    logging.info(f'Dataloader setup finish: train {len(train_loader)}, test {len(test_loader)}')
    return train_loader, test_loader

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    logging.info(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')
    return accuracy

def train_func(args):
    device = args.gpu_id
    model = setup_model(args)
    train_dataloader, test_dataloader = setup_dataset(args)

    amp_enabled = args.get("amp_enabled", False)

    # loss and optimizer
    scaler = GradScaler(enabled=amp_enabled)
    context = autocast('cuda') if amp_enabled  else nullcontext()
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
        for batch, (image, label) in enumerate(train_dataloader):
            batch_start_time = time.time()

            # forward/backward propagation
            with context:
                output = model(image.to(device))
                loss = F.nll_loss(output, label.to(device))

            scaler.scale(loss).backward()
            # gradients clip
            if args.get("clip_grad", False):
                dispatch_clip_grad(model.parameters(), value=max_grad_norm, mode=args.clip_mode)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
            batch_end_time = time.time()
            logging.info(f'optim step = {cur_step+1} loss = {round(loss.item(), 4)}')
            logging.info(f'Epoch [{epoch+1}/{args.epochs}] Batch [{batch+1}/{all_batch_length}] time {round(batch_end_time - batch_start_time, 4)} s.')
            cur_step += 1
        scheduler.step()

        # save checkpoints
        if epoch % args.save_interval == 0 or epoch == args.epochs:
            save_checkpoint(cur_step)
        # eval by epoch interval
        if (args.eval and epoch % args.eval_interval == 0 and device==0) or epoch == args.epochs - 1:
            logging.info(f'Epoch [{epoch+1}/{args.epochs}] Begin to test......')
            ave_accuracy = test(model, device, test_dataloader)
            if ave_accuracy > best_acc:
                save_checkpoint(cur_step, best=True)
                best_acc = ave_accuracy
                

