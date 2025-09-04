import torch
import torch_pruning as tp
from pruning_utils import collect_ignored_convs,validate,load_model
from typing import Tuple, Optional
from pathlib import Path
import sys
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils import YOLOv3Loss



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
    loss_meter = AverageMeter()

    for images, targets in tqdm(loader, desc="Training", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    return loss_meter.avg

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
    # if ignored_layers is None:
    #     ignore_layers = 

if __name__ == "__main__":
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(mode = "dpu",input_size = 416,num_classes = 20,model_type = "base",anchors = anchors,device = device,model_path = '/home/logictronix01/saurav/YOLOv3/weights/yolov3-base.pt')
    ignored_layers = collect_ignored_convs(model)
    print(model)
