import torch
import torch_pruning as tp
from typing import Tuple, Optional
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda import amp
from torch import optim
import torch.distributed as dist
import torch.multiprocessing as mp
from tqdm import tqdm
from pathlib import Path
import sys
import os
import argparse
from typing import Tuple, Optional
from pruning_utils import collect_ignored_convs,validate,load_model,get_dataloader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils import YOLOv3Loss,Evaluator,set_lr
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

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    multipart_loss_meter = AverageMeter()
    obj_loss_meter = AverageMeter()
    noobj_loss_meter = AverageMeter()
    txty_loss_meter = AverageMeter()
    twth_loss_meter = AverageMeter()
    cls_loss_meter = AverageMeter()
    optimizer.zero_grad()

    for (_,images, targets,_) in tqdm(loader, desc="Training", leave=False):

        images = images.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        outputs = model(images)
        loss = criterion(outputs, targets)

        loss[0].backward()
        optimizer.step()

        multipart_loss_meter.update(loss[0].item(),images.size(0))
        obj_loss_meter.update(loss[1].item(),images.size(0))
        noobj_loss_meter.update(loss[2].item(),images.size(0))
        txty_loss_meter.update(loss[3].item(),images.size(0))
        twth_loss_meter.update(loss[4].item(),images.size(0))
        cls_loss_meter.update(loss[5].item(),images.size(0))

    return [multipart_loss_meter.avg,obj_loss_meter.avg,noobj_loss_meter.avg,txty_loss_meter.avg,twth_loss_meter.avg,cls_loss_meter.avg]

@torch.no_grad()
def calc_mAP(args,model,val_loader,anchors,dpu:bool = True):
    model.eval()
    subset_class = val_loader.dataset
    base_dataset_class = val_loader.dataset.dataset
    idx = list(subset_class.indices)

    base_dataset_class.generate_mAP_source(save_dir = Path("./data/eval_src"),mAP_filename =mAP_filename,indices = idx)
    args.mAP_filepath = Path(base_dataset_class.mAP_filepath)
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
        _,images,targets,_ = next(iter(train_loader))
        images = images.to(device)
        
        optimizer.zero_grad(set_to_none = True)
        outputs = model(images)
        loss = criterion(outputs,targets)
        loss[0].backward()
        
        pruner.step()
        macs, params = tp.utils.count_ops_and_params(model, example_inputs.to(device))
        print(f"[Pruning] Step {i+1}/{iter_steps}: MACs={macs/1e6:.2f}M | Params={params/1e6:.2f}M")
        if finetune_epochs > 0:
            for e in range(finetune_epochs):
                tr_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
                print(f"↳ fine-tune e{e+1}/{finetune_epochs} | loss={tr_loss}")
        
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
    help="Epoch to learning rate decay"
    )

    parser.add_argument('--weight-decay',
                        type = float,
                        default = 0.0005,
                        help = 'Weight Decay')



    args,_ = parser.parse_known_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = load_model(mode = "dpu",input_size = 416,num_classes = 20,model_type = "base",anchors = anchors,device = device,model_path = '/home/logictronix01/saurav/YOLOv3/weights/yolov3-base.pt').to(device)
    ignored_layers = collect_ignored_convs(model,keep_stem = False,keep_stage_entry = False,)    

    train_loader = get_dataloader(voc_path = '/home/logictronix01/saurav/YOLOv3/data/voc.yaml',batch_size = 8,same_subset = False, subset_length = 2000, train = True)
    test_loader = get_dataloader(voc_path = '/home/logictronix01/saurav/YOLOv3/data/voc.yaml',batch_size = 8,same_subset = False,subset_length = 100, train = False)

    #Unit Test: evaluate function
    criterion = YOLOv3Loss(input_size=416,num_classes = 20,anchors = model.anchors)    
    loss,mAP_dict = evaluate(args = args,model = model,loader = test_loader,criterion=criterion,anchors = model.anchors,device = device,desc = "eval")
    print(loss)
    print(mAP_dict['all']['mAP_50'])

    # Unit test: train_one_epoch function
    optimizer = optim.SGD(model.parameters(),lr = args.base_lr,momentum = args.momentum,weight_decay = args.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones = args.lr_decay,gamma = 0.1)

    # loss = train_one_epoch(model,train_loader,criterion,optimizer,device)
    # print(loss)
    example_inputs = torch.randn(1,3,416,416).to(device)
    model = taylor_prune(model,example_inputs,train_loader,criterion,device=device,ignored_layers = ignored_layers,finetune_epochs = 3,round_to = 8)
        
    loss,mAP_dict = evaluate(args = args,model = model,loader = test_loader,criterion=criterion,anchors = model.anchors,device = device,desc = "eval")
    print(loss)
    print(mAP_dict['all']['mAP_50'])