import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
from typing import Tuple, Optional


class AverageMeter:
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / max(1, self.count)


@torch.no_grad()
def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1, 5)) -> Tuple[float, float]:
    """Compute top-k accuracy (supports fewer than k classes)."""
    maxk = max(min(k, output.size(1)) for k in topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        k = min(k, output.size(1))
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append((correct_k.mul_(100.0 / batch_size)).item())
    return tuple(res)

def cifar10_loaders(data_dir: str, batch_size: int = 128, workers: int = 4, input_size: int = 224):
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(input_size, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.Resize(input_size + 32),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_set = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_tf)
    test_set  = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_tf)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    return train_loader, test_loader

def build_resnet18(num_classes=10, pretrained=True):
    """Torchvision resnet18 adapted for CIFAR-10."""
    try:
        # torchvision>=0.13 style
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
    except Exception:
        model = models.resnet18(pretrained=pretrained)

    # Replace final FC for 10 classes
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

def build_vgg19(num_classes = 10,pretrained = True):
    """Building vgg19 model adapted for CIFAR-10"""
    try:
        weights = models.VGG19_Weights.DEFAULT if pretrained else None
        model = models.VGG19(weights=weights)
    except Exception:
        model = models.resnet18(pretrained = pretrained)
    
    #Replace final FC for 10 classes
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features = in_features,out_features = num_classes)

    return model

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    loss_meter = AverageMeter()
    top1_meter = AverageMeter()
    top5_meter = AverageMeter()

    for images, targets in tqdm(loader, desc="Training", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        top1, top5 = accuracy(outputs.detach(), targets)
        loss_meter.update(loss.item(), images.size(0))
        top1_meter.update(top1, images.size(0))
        top5_meter.update(top5, images.size(0))

    return loss_meter.avg, top1_meter.avg, top5_meter.avg

@torch.no_grad()
def evaluate(model, loader, criterion, device, desc="eval"):
    model.eval()
    loss_meter = AverageMeter()
    top1_meter = AverageMeter()
    top5_meter = AverageMeter()

    for images, targets in tqdm(loader, desc=f"Evaluating-{desc}", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        outputs = model(images)
        loss = criterion(outputs, targets).item()
        top1, top5 = accuracy(outputs, targets)
        loss_meter.update(loss, images.size(0))
        top1_meter.update(top1, images.size(0))
        top5_meter.update(top5, images.size(0))

    print(f"[{desc}] Loss: {loss_meter.avg:.4f} | Top1: {top1_meter.avg:.2f}% | Top5: {top5_meter.avg:.2f}%")
    return loss_meter.avg, top1_meter.avg, top5_meter.avg

