import torch
import torch_pruning as tp
from typing import Tuple, Optional
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.cuda.amp as amp
from torch import optim
import torch.distributed as dist
import torch.multiprocessing as mp
from tqdm import tqdm
from pathlib import Path
import sys
import os
import argparse
from typing import Tuple, Optional
import platform
from pruning_utils import collect_ignored_convs,validate,load_model,get_dataloader
import numpy as np
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils import YOLOv3Loss,Evaluator,set_lr,ModelEMA
from model import do_sigmoid

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


OS_SYSTEM = platform.system()


def setup(rank,world_size):
    if OS_SYSTEM == "Linux":
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12345"
        dist.init_process_group("nccl",rank = rank,world_size = world_size)

def cleanup():
    if OS_SYSTEM == "Linux":
        dist.destroy_process_group()

def train_one_epoch(args,model, loader,ema, criterion, optimizer, device,scaler,epochs):
    model.train()
    multipart_loss_meter = AverageMeter()
    obj_loss_meter = AverageMeter()
    noobj_loss_meter = AverageMeter()
    txty_loss_meter = AverageMeter()
    twth_loss_meter = AverageMeter()
    cls_loss_meter = AverageMeter()
    optimizer.zero_grad()
    losses = defaultdict(float)
    loss_type = ['multipart','obj','noobj','txty','twth','cls']

    for i,(_,images, targets,_) in enumerate(tqdm(loader, desc="Training", leave=False)):
        ni = i + len(loader) * (epochs - 1)
        
        if ni <= args.nw:
            args.grad_accumulate = max(1,np.interp(ni,[0,args.nw],[1,args.nominal_batch_size / args.batch_size]).round())
            set_lr(optimizer,args.base_lr * pow(ni/(args.nw),4))

        images = images.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        with amp.autocast(enabled = not args.no_amp):
            outputs = model(images)
            loss = criterion(outputs, targets)
        scaler.scale((loss[0] / args.grad_accumulate) * args.world_size).backward()

        if ni - args.last_opt_step >= args.grad_accumulate:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if ema is not None:
                ema.update(model)
            args.last_opt_step = ni
        

        multipart_loss_meter.update(loss[0].item(),images.size(0))
        obj_loss_meter.update(loss[1].item(),images.size(0))
        noobj_loss_meter.update(loss[2].item(),images.size(0))
        txty_loss_meter.update(loss[3].item(),images.size(0))
        twth_loss_meter.update(loss[4].item(),images.size(0))
        cls_loss_meter.update(loss[5].item(),images.size(0))
        
        del images,outputs
        torch.cuda.empty_cache()

    
    loss_avg =  [multipart_loss_meter.avg,obj_loss_meter.avg,noobj_loss_meter.avg,txty_loss_meter.avg,twth_loss_meter.avg,cls_loss_meter.avg]
    loss_str = f"[Train-Epoch:{epochs:03d}]"
    
    for loss_name,loss_value in zip(loss_type,loss_avg):
        losses[loss_name] = loss_value
        loss_str += f"{loss_name}:{losses[loss_name]:.4f}"
        print(f"{loss_name}: {losses[loss_name]:.4f}")

    return loss_str

@torch.no_grad()
def calc_mAP(args,model,val_loader,anchors,dpu:bool = True):
    model.eval()
    args.mAP_filepath = Path(val_loader.dataset.dataset.mAP_filepath)
    args.exp_path = Path(args.exp_path)
    os.makedirs(args.exp_path, exist_ok=True)
    evaluator = Evaluator(args.mAP_filepath)
    mAP_dict,eval_text = validate(args,anchors = anchors, dataloader = val_loader,model = model,evaluator = evaluator,save_result = True,dpu  = True, save_filename = "Pruned_map.txt")

    return mAP_dict,eval_text

@torch.no_grad()
def evaluate(args, model, loader, criterion,anchors, device, desc="eval"):
    model.eval()
    multipart_loss_meter = AverageMeter()
    obj_loss_meter = AverageMeter()
    noobj_loss_meter = AverageMeter()
    txty_loss_meter = AverageMeter()
    twth_loss_meter = AverageMeter()
    cls_loss_meter = AverageMeter()

    for _,images, targets,_ in tqdm(loader, desc=f"Evaluating-{desc}", leave=False):
        images = images.to(device, non_blocking=True)
        out = model(images)
        preds0 = do_sigmoid(out[0])
        preds1 = do_sigmoid(out[1])
        preds2 = do_sigmoid(out[2])
        outputs = (preds0,preds1,preds2)

        loss = criterion(outputs, targets)
        multipart_loss_meter.update(loss[0].item(),images.size(0))
        obj_loss_meter.update(loss[1].item(),images.size(0))
        noobj_loss_meter.update(loss[2],images.size(0))
        txty_loss_meter.update(loss[3],images.size(0))
        twth_loss_meter.update(loss[4],images.size(0))
        cls_loss_meter.update(loss[5],images.size(0))
        # loss_meter.update(loss, images.size(0))

    mAP_dict,eval_text = calc_mAP(args,model,val_loader = loader,anchors = anchors,dpu = True)
    print(f"[{desc}]\nMultipart_Loss: {multipart_loss_meter.avg:.4f} | Object Loss:{obj_loss_meter.avg:.4f} | No Object Loss:{noobj_loss_meter.avg:.4f} | txty Loss:{txty_loss_meter.avg:.4f} twth Loss:{twth_loss_meter.avg:.4f} | cls loss:{cls_loss_meter.avg:.4f}\nmAP:{eval_text}")
    return [multipart_loss_meter.avg,obj_loss_meter.avg,noobj_loss_meter.avg,txty_loss_meter.avg,twth_loss_meter.avg,cls_loss_meter.avg],mAP_dict


def taylor_prune(args,
                 model: nn.Module,
                 example_inputs: torch.Tensor,
                 train_loader: DataLoader,
                 test_loader:DataLoader,
                 criterion: nn.Module,
                 device: torch.device,
                 pruning_ratio: float = 0.45,
                 iter_steps: int = 5,
                 round_to: int = 16,
                 ignored_layers: Optional[list] = None,
                 finetune_epochs: int = 0,
                 lr: float = 1e-3) -> nn.Module:
    assert tp is not None, "torch-pruning is not installed. pip install torch-pruning"
    
    model.to(device)
    model.train()

    # Importance: TaylorExpansion (requires gradients)
    imp = tp.importance.TaylorImportance()

    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs.to(device),
        importance = imp,
        iterative_steps = iter_steps,
        pruning_ratio = pruning_ratio,
        ignored_layers = ignored_layers,
        round_to = round_to
    )

    base_macs,base_params = tp.utils.count_ops_and_params(model,example_inputs.to(device))
    print(f"[Pruning] Baseline: MACs={base_macs/1e6:.2f}M | Params={base_params/1e6:.2f}M")

    optimizer = optim.SGD(model.parameters(),lr = lr,momentum = 0.9, weight_decay = 1e-4)

    for i in range(iter_steps):
        # --- collect Taylor grads on a small batch
        _, images, targets, _ = next(iter(train_loader))
        images = images.to(device, non_blocking=True)
 
        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss[0].backward()

        # --- prune
        pruner.step()
        base_macs,base_params = tp.utils.count_ops_and_params(model,example_inputs.to(device))
        print(f"[Pruning] Baseline: MACs={base_macs/1e6:.2f}M | Params={base_params/1e6:.2f}M")

        # --- rebuild optimizer after structural change
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
        optimizer.zero_grad(set_to_none=True)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=finetune_epochs, eta_min=0.001 * 0.1
                )

        # --- optional BN recalibration
        with torch.no_grad():
            model.train()
            for _b, (_, imgs, _, _) in zip(range(100), train_loader):
                imgs = imgs.to(device, non_blocking=True)
                _ = model(imgs)

        # --- short fine-tune: NO warm-up here
        if finetune_epochs > 0:
            # fix accumulation and disable warm-up in your train loop
            steady_acc = max(round(args.nominal_batch_size / args.batch_size), 1)
            args.grad_accumulate = steady_acc
            args.nw = -1
            args.last_opt_step = -1
            scaler = amp.GradScaler(enabled = not args.no_amp)
            ema = ModelEMA(model = model)
            best_map = 0
            for e in range(finetune_epochs):
                tr_str = train_one_epoch(args,model, train_loader, ema, criterion, optimizer, device,scaler,epochs = e + 1)  # train loop should respect args.nw == -1
                scheduler.step()
                print()
                print(f" ↳ fine-tune {e+1}/{finetune_epochs} | {tr_str}")
                        
                loss,mAP_dict = evaluate(args = args,model = model,loader = test_loader,criterion=criterion,anchors = model.anchors,device = device,desc = "eval")
                print(loss)
                if mAP_dict['all']['mAP_50'] > best_map:
                    best_map = mAP_dict['all']['mAP_50']
                    torch.save(model,'./yolov3-pruned-best.pth')
    return model


if __name__ == "__main__":
    torch.cuda.empty_cache()
    anchors = [
        [0.248,      0.7237237 ],
        [0.36144578, 0.53      ],
        [0.42,       0.9306667 ],
        [0.456,      0.6858006 ],
        [0.488,      0.8168168 ],
        [0.6636637,  0.274     ],
        [0.806,      0.648     ],
        [0.8605263,  0.8736842 ],
        [0.944,      0.5733333 ]
        ]

    parser = argparse.ArgumentParser(description = "YOLOv3 Pruning")


    parser.add_argument('--model',
                        type = str,
                        required=False,
                        help = "Path to .pt file.")
    parser.add_argument('--data',
                        type = str,
                        help = "Path to .yaml files for data")

    parser.add_argument('--img-size',
                        type = int,
                        default = 416,
                        help = "Required Image Size to the model")
    parser.add_argument('--exp-path',
                        type = str,
                        default = './mAP_results')
    parser.add_argument('--conf-thres',
                        type = float,
                        default = 0.3,
                        help = "confidence threshold for calculating mAP")
    parser.add_argument('--nms-thres',
                        type = float,
                        default = 0.6,
                        help = "nms threshold for calculating mAP")
    parser.add_argument('--base-lr', 
                        type = float,
                        default = 0.001, 
                        help = "Base Learning rate"
                        )
    parser.add_argument('--momentum',
                        type = float,
                        default = 0.9,
                        help = "momentum for optimizer")
    parser.add_argument("--lr-decay", 
                        nargs="+", 
                        default=[150, 200], 
                        type=int, 
                        help="Epoch to learning rate decay")

    parser.add_argument('--weight-decay',
                        type = float,
                        default = 0.0005,
                        help = 'Weight Decay')

    parser.add_argument('--batch-size',
                        type = int,
                        default = 8,
                        help = 'batch_size')

    parser.add_argument('--train-base',
                        action='store_true',
                        help = "preTrain the base model")
    parser.add_argument('--pretrain-epochs',
                        type = int,
                        default = 15,
                        help = "Pretrain epochs")
    parser.add_argument('--prune',
                        action = 'store_true',
                        help = "prune the model")
    parser.add_argument('--prune-steps',
                        type = int,
                        default = 3,
                        help = "Number of iteration for pruning")
    parser.add_argument('--finetune-steps',
                        type = int,
                        default = 3,
                        help = "Number of finetuning steps after each pruning step")
    parser.add_argument('--world-size',
                        type = int,
                        default = 1,
                        help = "Number of devices available")
    parser.add_argument('--no-amp',
                        action = "store_true",
                        help = "Don't use grad_scaler and AMP")

    parser.add_argument('--post-train-pruned',
                        action = 'store_true',
                        help = "path to pruned model")

    args,_ = parser.parse_known_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = load_model(mode = "dpu",input_size = 416,num_classes = 20,model_type = "base",anchors = anchors,device = device,model_path = args.model).to(device)
    example_inputs = torch.randn(1,3,416,416).to(device)
    base_macs,base_params = tp.utils.count_ops_and_params(model,example_inputs.to(device))
    print(f"[Pruning] Baseline: MACs={base_macs/1e6:.2f}M | Params={base_params/1e6:.2f}M")
    ignored_layers = collect_ignored_convs(model,keep_stem = False,keep_stage_entry = False,keep_stage_exit= False)    
    
    train_loader = get_dataloader(voc_path = args.data,batch_size = 8,same_subset = False, subset_length = 10000, train = True,mAP_filename='eval_train_10000.json')
    test_loader = get_dataloader(voc_path = args.data,batch_size = 8,same_subset = False,subset_length = 1000, train = False,mAP_filename = 'eval_test_1000.json')

    #Train for 15 epochs before pruning:
    criterion = YOLOv3Loss(input_size=416,num_classes = 20,anchors = model.anchors)
    loss,mAP_dict = evaluate(args = args,model = model,loader = test_loader,criterion=criterion,anchors = model.anchors,device = device,desc = "eval")
    print(loss)
    print(mAP_dict['all']['mAP_50'])

    if args.train_base:
        args.nominal_batch_size = 64
        args.last_opt_step = -1
        args.grad_accumulate = max(round(args.nominal_batch_size / args.batch_size), 1)
        args.nw = -1
        
        optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.pretrain_epochs, eta_min=args.base_lr * 0.1
                )
        scaler = amp.GradScaler(enabled = not args.no_amp)
        ema = ModelEMA(model = model)
        for e in range(args.pretrain_epochs):
            loss_str = train_one_epoch(args,model,train_loader,ema,criterion,optimizer,device,scaler,e + 1)
            scheduler.step()
            print(f"Epoch{e + 1}/{args.pretrain_epochs} Loss: {loss_str}")
            loss,mAP_dict = evaluate(args = args,model = model,loader = test_loader,criterion=criterion,anchors = model.anchors,device = device,desc = "eval")
            print(loss)
            print(mAP_dict['all']['mAP_50'])

    if args.prune:
        example_inputs = torch.randn(1,3,416,416).to(device)
        model = taylor_prune(
            args,
            model,
            example_inputs,
            train_loader,
            test_loader,
            criterion,
            device,
            pruning_ratio = 0.2,
            iter_steps = args.prune_steps,
            ignored_layers = ignored_layers,
            finetune_epochs = args.finetune_steps,
            lr = args.base_lr
        )

    print(args.post_train_pruned)
    if args.post_train_pruned:
        model  = torch.load('/home/logictronix01/saurav/YOLOv3/VainF_pruning/yolov3-pruned-best.pth',map_location = 'cpu',weights_only = False).to(device)
        base_macs,base_params = tp.utils.count_ops_and_params(model,example_inputs.to(device))
        print(f"[Pruning] Baseline: MACs={base_macs/1e6:.2f}M | Params={base_params/1e6:.2f}M")
        
        loss,mAP_dict = evaluate(args = args,model = model,loader = test_loader,criterion=criterion,anchors = model.anchors,device = device,desc = "eval")
        print(loss)
        
        args.nominal_batch_size = 64
        args.last_opt_step = -1
        args.grad_accumulate = max(round(args.nominal_batch_size / args.batch_size), 1)
        args.warmup = 5
        args.nw = -1
        # max(round(args.warmup * len(train_loader)), 100)
       
        optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.pretrain_epochs, eta_min=args.base_lr * 0.1
                )
        scaler = amp.GradScaler(enabled = not args.no_amp)
        ema = ModelEMA(model = model)
        best_map = 0

        for e in range(args.pretrain_epochs):
            loss_str = train_one_epoch(args,model,train_loader,ema,criterion,optimizer,device,scaler,e + 1)
            scheduler.step()
            print(f"Epoch{e + 1}/{args.pretrain_epochs} Loss: {loss_str}")
            loss,mAP_dict = evaluate(args = args,model = model,loader = test_loader,criterion=criterion,anchors = model.anchors,device = device,desc = "eval")
            print(loss)
            if mAP_dict['all']['mAP_50'] > best_map:
                best_map = mAP_dict['all']['mAP_50']
                torch.save(model,'./yolov3-pruned-best.pth')

        


        






    # # Unit test: train_one_epoch function
    # optimizer = optim.SGD(model.parameters(),lr = args.base_lr,momentum = args.momentum,weight_decay = args.weight_decay)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones = args.lr_decay,gamma = 0.1)

    # # loss = train_one_epoch(model,train_loader,criterion,optimizer,device)
    # # print(loss)
    # example_inputs = torch.randn(1,3,416,416).to(device)
    # model = taylor_prune(model,example_inputs,train_loader,criterion,device=device,ignored_layers = ignored_layers,finetune_epochs = 3,round_to = 8)
        
    # loss,mAP_dict = evaluate(args = args,model = model,loader = test_loader,criterion=criterion,anchors = model.anchors,device = device,desc = "eval")
    # print(loss)
    # print(mAP_dict['all']['mAP_50'])