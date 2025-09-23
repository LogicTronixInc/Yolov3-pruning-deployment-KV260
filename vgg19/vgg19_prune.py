#torch essentials
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

#other utils
from typing import Tuple, Optional
from pathlib import Path
import sys
import time
import argparse
import tqdm

from cls_net_utils import build_resnet18,accuracy,cifar10_loaders,train_one_epoch,evaluate

try:
    import torch_pruning as tp
except Exception:
    tp = None

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

if __name__ == '__main__':
    main()

