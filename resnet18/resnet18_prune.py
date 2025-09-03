import argparse
import os
import time
from pathlib import Path
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm

# ---- Optional pruning (install: pip install torch-pruning)
try:
    import torch_pruning as tp
except Exception:
    tp = None

# ---- Optional Vitis-AI quantizer (available inside Vitis-AI Docker)
try:
    from pytorch_nndct import torch_quantizer
except Exception:
    torch_quantizer = None

# ------------------------
# Utilities
# ------------------------
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

# ------------------------
# Data
# ------------------------

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

# ------------------------
# Model
# ------------------------

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

# ------------------------
# Train / Eval
# ------------------------

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

# ------------------------
# Pruning (Taylor importance)
# ------------------------

def taylor_prune(model: nn.Module,
                 example_inputs: torch.Tensor,
                 train_loader: DataLoader,
                 criterion: nn.Module,
                 device: torch.device,
                 pruning_ratio: float = 0.5,
                 iter_steps: int = 5,
                 round_to: int = 8,
                 ignored_layers: Optional[list] = None,
                 finetune_epochs: int = 0,
                 lr: float = 1e-3) -> nn.Module:
    assert tp is not None, "torch-pruning is not installed. pip install torch-pruning"

    model.to(device)
    model.train()

    # Importance: TaylorExpansion (requires gradients)
    imp = tp.importance.TaylorImportance()

    # Ignore classifier
    if ignored_layers is None:
        ignored_layers = []
        if hasattr(model, 'fc') and isinstance(model.fc, nn.Linear):
            ignored_layers.append(model.fc)

    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs.to(device),
        importance=imp,
        iterative_steps=iter_steps,
        pruning_ratio=pruning_ratio,
        ignored_layers=ignored_layers,
        round_to=round_to,
    )

    base_macs, base_params = tp.utils.count_ops_and_params(model, example_inputs.to(device))
    print(f"[Pruning] Baseline: MACs={base_macs/1e6:.2f}M | Params={base_params/1e6:.2f}M")

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)

    for i in range(iter_steps):
        # one small batch to obtain gradients for Taylor criterion
        images, targets = next(iter(train_loader))
        images = images.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()  # gradients needed by TaylorImportance

        pruner.step()  # actually remove channels
        macs, params = tp.utils.count_ops_and_params(model, example_inputs.to(device))
        print(f"[Pruning] Step {i+1}/{iter_steps}: MACs={macs/1e6:.2f}M | Params={params/1e6:.2f}M")

        # optional quick fine-tune between steps to stabilize
        if finetune_epochs > 0:
            for e in range(finetune_epochs):
                tr_loss, tr_t1, tr_t5 = train_one_epoch(model, train_loader, criterion, optimizer, device)
                print(f"   ↳ fine-tune e{e+1}/{finetune_epochs} | loss={tr_loss:.4f} | top1={tr_t1:.2f} | top5={tr_t5:.2f}")

    return model

# ------------------------
# Quantization (PTQ) with Vitis AI (pytorch_nndct)
# ------------------------

def vitis_ai_ptq(model: nn.Module,
                 calib_loader: DataLoader,
                 test_loader: DataLoader,
                 example_inputs: torch.Tensor,
                 device: torch.device,
                 export_dir: str = "quantize_result",
                 deploy: bool = False,
                 calib_batches: int = 200):
    assert torch_quantizer is not None, "pytorch_nndct (Vitis-AI) is not installed. Use the Vitis-AI PyTorch Docker."

    model.to(device).eval()
    example_inputs = example_inputs.to(device)

    # 1) Calibration
    print("[VAI PTQ] Calibration …")
    quantizer = torch_quantizer("calib", model, (example_inputs,), output_dir=export_dir)
    quant_model = quantizer.quant_model
    seen = 0
    with torch.no_grad():
        for images, _ in tqdm(calib_loader, desc="Calibration", leave=False):
            images = images.to(device)
            _ = quant_model(images)
            seen += 1
            if seen >= calib_batches:  # limit for speed
                break
    # export calibration results
    quantizer.export_quant_config()

    # 2) Evaluate INT8 accuracy (simulation)
    print("[VAI PTQ] Evaluating quantized model (simulated int8) …")
    criterion = nn.CrossEntropyLoss().to(device)
    q_loss, q_top1, q_top5 = evaluate(quant_model, test_loader, criterion, device, desc="quant-sim")

    # 3) Export xmodel (test/deploy stage). Batch must be 1.
    print("[VAI PTQ] Exporting xmodel …")
    quantizer = torch_quantizer("test", model, (example_inputs,), output_dir=export_dir)
    quantizer.export_xmodel(output_dir=export_dir, deploy_check=deploy)
    print(f"[VAI PTQ] xmodel exported to: {export_dir}")

    return {"quant_top1": q_top1, "quant_top5": q_top5}

# ------------------------
# CLI
# ------------------------

def is_full_model_file(p: Path) -> bool:
    try:
        obj = torch.load(str(p), map_location="cpu")
        return isinstance(obj, nn.Module)
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description="ResNet18 pruning→quantization on CIFAR-10")
    parser.add_argument('--data-dir', type=str, default='./data')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--input-size', type=int, default=224)

    parser.add_argument('--resume', type=str, default=None, help='float checkpoint (state_dict) or a full model .pth after pruning')

    parser.add_argument('--prune', action='store_true', help='run Taylor-importance pruning')
    parser.add_argument('--prune-ratio', type=float, default=0.5)
    parser.add_argument('--iter-steps', type=int, default=5)
    parser.add_argument('--round-to', type=int, default=8)
    parser.add_argument('--finetune-epochs', type=int, default=0)
    parser.add_argument('--finetune-lr', type=float, default=1e-3)

    parser.add_argument('--quantize', action='store_true', help='run Vitis-AI PTQ & export xmodel')
    parser.add_argument('--export-dir', type=str, default='./quantize_result')
    parser.add_argument('--deploy', action='store_true', help='enable deploy_check during export_xmodel')
    parser.add_argument('--calib-batches', type=int, default=200)
    parser.add_argument('--finetune-model',action = 'store_true',default = None,help = 'float checkpoint full model .pth after pruning')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data
    train_loader, test_loader = cifar10_loaders(args.data_dir, args.batch_size, args.workers, args.input_size)

    # Model build or resume
    criterion = nn.CrossEntropyLoss().to(device)

    if (args.resume is None) and (args.finetune_model is None):
        # fresh / float training
        model = build_resnet18(num_classes=10, pretrained=True)
        model.to(device)
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        best_top1 = 0.0
        for epoch in range(args.epochs):
            t0 = time.time()
            tr_loss, tr_top1, tr_top5 = train_one_epoch(model, train_loader, criterion, optimizer, device)
            va_loss, va_top1, va_top5 = evaluate(model, test_loader, criterion, device, desc=f"float@e{epoch+1}")
            sched.step()
            if va_top1 > best_top1:
                best_top1 = va_top1
                torch.save({'state_dict': model.state_dict()}, 'best_float.pth')
            print(f"[Epoch {epoch+1}/{args.epochs}] train: loss={tr_loss:.4f} top1={tr_top1:.2f} | val: top1={va_top1:.2f} | time={(time.time()-t0):.1f}s")
        print("Saved float checkpoint: best_float.pth")
    else:
        # Load checkpoint
        resume_path = Path(args.resume)
        if is_full_model_file(resume_path):
            print(f"Loading full model object from: {resume_path}")
            model = torch.load(str(resume_path), map_location='cpu')
            model.to(device)
        else:
            print(f"Loading state_dict from: {resume_path}")
            model = build_resnet18(num_classes=10, pretrained=True)
            ckpt = torch.load(str(resume_path), map_location='cpu')
            model.load_state_dict(ckpt['state_dict'])
            model.to(device)
        
        if args.finetune_model:
            pruned_path = Path(args.resume)
        if is_full_model_file(pruned_path):
            print(f"Loading state_dict from: {pruned_path}")
            print(f"Finetuning pruned model....")
            model = torch.load(str(resume_path), map_location='cpu')
            model.to(device)
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
            sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
            va_loss, va_top1, va_top5 = evaluate(model, test_loader, criterion, device, desc="loaded")
            best_top1 = va_top1
            
            for epoch in range(args.epochs):
                t0 = time.time()
                tr_loss, tr_top1, tr_top5 = train_one_epoch(model, train_loader, criterion, optimizer, device)
                va_loss, va_top1, va_top5 = evaluate(model, test_loader, criterion, device, desc=f"float@e{epoch+1}")
                sched.step()
                if va_top1 > best_top1:
                    best_top1 = va_top1
                torch.save(model, 'best_pruned.pth')
                print(f"[Epoch {epoch+1}/{args.epochs}] train: loss={tr_loss:.4f} top1={tr_top1:.2f} | val: top1={va_top1:.2f} | time={(time.time()-t0):.1f}s")

        else:
            raise TypeError(f"{pruned_path} must be full model. Pass pruned model .pth file not state_dict")


        print("Saved pruned checkpoint: best_pruned.pth")
    


    example_inputs = torch.randn(1, 3, args.input_size, args.input_size)
    if args.prune:
            model = taylor_prune(model,
                                example_inputs,
                                train_loader,
                                criterion,
                                device,
                                pruning_ratio=args.prune_ratio,
                                iter_steps=args.iter_steps,
                                round_to=args.round_to,
                                finetune_epochs=args.finetune_epochs,
                                lr=args.finetune_lr)
            # Save entire model (structure changed)
            torch.save(model, 'best_pruned.pth')
            print("Saved pruned model: best_pruned.pth")
            evaluate(model, test_loader, criterion, device, desc="pruned")

    if args.finetune_model:
        pruned_path = Path(args.finetune_model)
        if is_full_model_file(pruned_path):
            print(f"Loading state_dict from: {pruned_path}")
            
            model = torch.load(str(resume_path), map_location='cpu')
            model.to(device)
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
            sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
            va_loss, va_top1, va_top5 = evaluate(model, test_loader, criterion, device, desc="loaded")
            best_top1 = va_top1
            
            for epoch in range(args.epochs):
                t0 = time.time()
                tr_loss, tr_top1, tr_top5 = train_one_epoch(model, train_loader, criterion, optimizer, device)
                va_loss, va_top1, va_top5 = evaluate(model, test_loader, criterion, device, desc=f"float@e{epoch+1}")
                sched.step()
                if va_top1 > best_top1:
                    best_top1 = va_top1
                torch.save(model, 'best_pruned.pth')
                print(f"[Epoch {epoch+1}/{args.epochs}] train: loss={tr_loss:.4f} top1={tr_top1:.2f} | val: top1={va_top1:.2f} | time={(time.time()-t0):.1f}s")

        else:
            raise TypeError(f"{pruned_path} must be full model. Pass pruned model .pth file not state_dict")


        print("Saved pruned checkpoint: best_pruned.pth")



    # Quantization (PTQ)
    if args.quantize:
        stats = vitis_ai_ptq(model, train_loader, test_loader, example_inputs, device,
                             export_dir=args.export_dir, deploy=args.deploy, calib_batches=args.calib_batches)
        print(f"Quantized (sim) Top1={stats['quant_top1']:.2f} | Top5={stats['quant_top5']:.2f}")

        # After export, compile with VAI_C outside Python, e.g.:
        #   vai_c_xir -x {args.export_dir}/ResNet_int.xmodel -a <arch.json> -o ./compiled -n resnet18_pruned

if __name__ == '__main__':
    main()
