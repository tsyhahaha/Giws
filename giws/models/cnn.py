import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler
import warnings
warnings.filterwarnings("ignore")

class MINSTNet(nn.Module):
    def __init__(self):
        super(MINSTNet, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.25)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128*7*7, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)

def get_transforms():
    return transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1))
    ])

def get_test_transforms():  
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

def train(args, model, device, train_loader, optimizer, epoch, scaler):
    model.train()
    total_loss = 0
    start_time = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad(set_to_none=True)
        
        with autocast(enabled=args.amp, device_type='cuda'):
            output = model(data)
            loss = F.nll_loss(output, target)
        
        scaler.scale(loss).backward()
        if args.clip_grad:                    # 梯度裁剪
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        if batch_idx % args.log_interval == 0:
            avg_loss = total_loss / (batch_idx + 1)
            elapsed_time = time.time() - start_time
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {avg_loss:.6f}\tTime Interval: {elapsed_time:.2f}s')
            start_time = time.time()
    return total_loss / len(train_loader)

def test(model, device, test_loader):
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

def main():
    parser = argparse.ArgumentParser(description='Optimized MNIST Training')
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=3e-4)         # 更合适的初始学习率
    parser.add_argument('--wd', type=float, default=1e-4)         # 权重衰减
    parser.add_argument('--amp', action='store_true')             # 混合精度训练
    parser.add_argument('--clip-grad', action='store_true')      # 梯度裁剪
    parser.add_argument('--log-interval', type=int, default=10)
    parser.add_argument('--test-batch-size', type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, 
                      transform=get_transforms(),  # 应用数据增强
                      download=True),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False,
                      transform=get_test_transforms()),
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # 初始化模型和优化器
    model = MINISTNet().to(device)
    
    optimizer = optim.AdamW(model.parameters(), 
                           lr=args.lr,)
                           # weight_decay=args.wd)
    
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    scaler = GradScaler(enabled=args.amp)
    
    best_acc = 0
    for epoch in range(1, args.epochs + 1):
        train_loss = train(args, model, device, train_loader, optimizer, epoch, scaler)
        scheduler.step()
        
        acc = test(model, device, test_loader)
        
        # 保存最佳模型
        if acc > best_acc:
            best_acc = acc
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'best_acc': best_acc
            }, "best_model.pth")
    
    print(f"Best Test Accuracy: {best_acc:.2f}%")

if __name__ == '__main__':
    main()
    # python train.py --amp --clip-grad --batch-size 512