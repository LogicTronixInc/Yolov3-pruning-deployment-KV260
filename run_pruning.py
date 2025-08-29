import argparse,time,os
import torch,torch.nn as nn
import torch.nn.functional as F
import torch_pruning as tp
from pathlib import Path
import sys
from torch.utils.data import DataLoader,Subset
from ptflops import get_model_complexity_info
from pruning_utils import count_params,fps_report,build_ratio_dict_for_heads,head_out_channels,get_head_3x3_convs,round_layers_to_multiple,fuse_model,count_bn,collect_ignored_convs,bn_recalibration
from tqdm import tqdm
from benchmark_utilities import script_model
import psutil
from benchmark_model import _benchmark_fps,benchmark_fps_saved_pruned,benchmark_fps_saved_torchscript
from script_model import script_model

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from inference import preprocess,post_process
from model import Conv,ResBlock,build_backbone,FPN,YOLOv3,YoloHead,DetectLayer
from dataloader import Dataset,BasicTransform

# @torch.no_grad()
# def benchmark_fps(model, input_size=608, runs=50, warmup=10, device="cpu"):
#     model.eval().to(device)
#     x = torch.randn(1, 3, input_size, input_size, device=device)
#     # warmup
#     for _ in range(warmup):
#         _ = model(x)
#     torch.cuda.synchronize() if device.startswith("cuda") else None
#     t0 = time.time()
#     for _ in range(runs):
#         _ = model(x)
#     torch.cuda.synchronize() if device.startswith("cuda") else None
#     dt = time.time() - t0
#     fps = runs / dt
#     return fps


# Here pruning of the model takes place. 
# MACs and Nparams are calculated pre and post pruning the model.
# Sparsity is also calculated post pruning model.

#Builds the model from checkpoint path provided to the script.
def build_model(input_size, num_classes, anchors, model_type, ckpt_path: str):
    model = YOLOv3(input_size=input_size,
                   num_classes=num_classes,
                   anchors=anchors,
                   model_type=model_type,
                   pretrained=False)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt["model_state"] if "model_state" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(sd, strict=True)
    print(f"[load] missing={len(missing)} unexpected={len(unexpected)}")
    return model
    
def channel_prune(model, input_size=416, target_sparsity=0.5, stepwise=True):
    model.eval()
    example_inputs = torch.randn(1, 3, input_size, input_size)

    importance = tp.importance.BNScaleImportance()  
    ignored_layers = collect_ignored_convs(model)
    print(len(ignored_layers))

    ratio_dict = build_ratio_dict_for_heads(model,head_ratio = 0.50)

    for p in model.parameters():
        p.requires_grad_(True)

    with torch.enable_grad():
        pruner = tp.pruner.MagnitudePruner(
            model,
            example_inputs,
            importance=importance,
            iterative_steps=10 if stepwise else 1,
            pruning_ratio=target_sparsity,
            pruning_ratio_dict= ratio_dict,
            global_pruning=True,
            ignored_layers=ignored_layers,
    )
        print(f"[prune] target global channel sparsity = {target_sparsity}")
        pruner.step()
    return model

def save(model, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for attr in [
        "start_flops_count","stop_flops_count","reset_flops_count",
        "compute_average_flops_cost","add_flops_counting_methods"
    ]:
        if hasattr(model, attr):
            delattr(model, attr)
    torch.save(model, os.path.join(out_dir, "yolov3_pruned_full.pt"))  # easiest
    
    
def main():
    phys = psutil.cpu_count(logical=False) or os.cpu_count() or 1
    torch.set_num_interop_threads(1)   # set once at program start
    torch.set_num_threads(phys)  
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="path to trained yolov3 .pt")
    ap.add_argument("--input_size", type=int, default=416)
    ap.add_argument("--num_classes", type=int, default=20)
    ap.add_argument("--model_type", type=str, default="base")  # matches your naming
    ap.add_argument("--sparsity", type=float, default=0.4)
    ap.add_argument("--out", type=str, default="pruned_export")
    args = ap.parse_args()

    # Example COCO/VOC anchors parser (expect 9 numbers*2)
    anchors = [[0.248, 0.7237237 ],
        [0.36144578, 0.53      ],
        [0.42,       0.9306667 ],
        [0.456,      0.6858006 ],
        [0.488,      0.8168168 ],
        [0.6636637,  0.274     ],
        [0.806,      0.648     ],
        [0.8605263,  0.8736842 ],
        [0.944,      0.5733333 ]]

    model = build_model(args.input_size, args.num_classes, anchors, args.model_type, args.ckpt)
    if hasattr(model, "set_grid_xy"):
        model.set_grid_xy(args.input_size)

    pre_macs,pre_params,pre_fps = _benchmark_fps(model,input_size = args.input_size,runs = 20,warmup = 5, device = 'cpu')
    
    print(f"MACS:{pre_macs/1e9:.2f}")
    print(f"Nparam:{pre_params/1e6:.2f}")
    print(f"FPS:{pre_fps}")

    scripted = script_model(model,input_size = 416,filename = "./torchscript/yolo_unpruned_scripted.pt",channels_last = True)

    print(type(scripted))

    _,_,scr_fps = benchmark_fps_saved_torchscript('./torchscript/yolo_unpruned_scripted.pt')

    fps_report(pre_fps,scr_fps)

    input_sig = torch.randn(1,3,args.input_size,args.input_size)

    vals = head_out_channels(model)

    # print(f"Preprune Head channels Counts:")    
    # for key,values in vals.items():
    #     print(f"{key}: {values}")

    pruned = channel_prune(
            model,                          
            input_size=args.input_size,
            target_sparsity=args.sparsity, 
            stepwise=True
        )
    
    vals_after = head_out_channels(pruned)
    # print(f"Postprune Head Channels Counts:")
    # for key,values in vals_after.items():
    #     print(f"{key}: {values}")

    head_layers = get_head_3x3_convs(model= pruned)
    _ = round_layers_to_multiple(pruned,input_sig,head_layers,k = 8)
    val_after_k = head_out_channels(pruned)

    # print(f"Post Vector Friendly Channels:")
    # for key,values in val_after_k.items():
    #     print(f"{key}: {values}")

    val_dataset = Dataset('/home/saurav/Desktop/Internship/ML-Internship-Saurav-Paudel/Paper_Implementation/ObjectDetection/UniYOLO/data/voc.yaml',phase = 'val')
    k = 300
    val_transformer = BasicTransform(input_size=args.input_size)
    val_dataset.load_transformer(transformer=val_transformer)
    subset_idx = list(range(min(k,len(val_dataset))))
    val_subset = Subset(val_dataset,subset_idx)
    
    val_loader = DataLoader(dataset=val_subset, collate_fn=Dataset.collate_fn, batch_size=8, shuffle=False, pin_memory=True, num_workers=1)

    # start_time = time.time()
    # bn_recalibration(pruned, val_loader, iters=200)
    # print(f"200 forward passes for batch size of {batch_size}:{time.time() - start_time()}")

    # print("BN Before:",count_bn(pruned))
    # print(pruned.backbone.res_block1)
    pruned.eval()
    fuse_model(pruned)

    # print("BN After:",count_bn(pruned))
    
    save(pruned,out_dir="./pruned_models")

    post_macs,post_params,post_fps = benchmark_fps_saved_pruned(model_path = '/home/saurav/Desktop/Internship/ML-Internship-Saurav-Paudel/Paper_Implementation/ObjectDetection/UniYOLO/torch_pruning/pruned_models/yolov3_pruned_full.pt',warmup=20)
    fps_report(pre_fps,post_fps)

    scripted_model = script_model(pruned,filename = '/home/saurav/Desktop/Internship/ML-Internship-Saurav-Paudel/Paper_Implementation/ObjectDetection/UniYOLO/torch_pruning/torchscript/yolo_scripted.pt',input_size = args.input_size,channels_last = True,device = "cpu")

    print(type(scripted_model))

    torch_scipt_path = "/home/saurav/Desktop/Internship/ML-Internship-Saurav-Paudel/Paper_Implementation/ObjectDetection/UniYOLO/torch_pruning/torchscript/yolo_scripted.pt"

    macs,params,script_fps = benchmark_fps_saved_torchscript(torch_scipt_path)

    fps_report(pre_fps,script_fps)


if __name__ == "__main__":
    main()


