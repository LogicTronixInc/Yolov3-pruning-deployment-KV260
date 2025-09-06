import torch.nn as nn
import torch
import torch_pruning as tp
import json
from tqdm import tqdm
import numpy as np
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from model import YOLOv3_DPU,YOLOv3,BoxDecoder,do_sigmoid
from dataloader import Dataset,BasicTransform,AugmentTransform
from torch.utils.data import DataLoader,Subset
from utils import YOLOv3Loss,scale_coords,transform_xcycwh_to_x1y1x2y2,transform_x1y1x2y2_to_x1y1wh,filter_confidence,run_NMS,to_image


def count_params(m):
    return sum(p.numel() for p in m.parameters())

def fps_report(fps_before: float, fps_after: float):
    delta = fps_after - fps_before
    speedup = fps_after / fps_before
    pct_fps = (speedup - 1.0) * 100.0
    lat_before = 1000.0 / fps_before
    lat_after  = 1000.0 / fps_after
    pct_lat = (1.0 - lat_after / lat_before) * 100.0
    print(f"FPS: {fps_before:.2f} -> {fps_after:.2f}  |  +{delta:.2f} FPS ({pct_fps:.2f}%), speedup {speedup:.3f}x")
    print(f"Latency: {lat_before:.2f} ms -> {lat_after:.2f} ms  |  {-pct_lat:.2f}% change" if pct_lat < 0
          else f"Latency reduced by {pct_lat:.2f}%")
    

def get_head_3x3_convs(model):
    """Return the three Conv2d modules right before the final 1×1 detect layers."""
    # Your Conv wrapper is: .conv = Sequential[Conv2d, BN, Act] → so index 0 is Conv2d
    names = dict(model.named_modules())
    picks = []
    for k in ["head.detect_s.conv.conv.0", "head.detect_m.conv.conv.0", "head.detect_l.conv.conv.0"]:
        if k in names and isinstance(names[k], nn.Conv2d):
            picks.append(names[k])
        else:
            print(f"[WARN] Could not find {k} — check your module names.")
    return picks

def round_layers_to_multiple(model, example_inputs, layers, k: int = 16):
    """
    For each Conv2d in `layers`, round down out_channels to nearest multiple of k.
    Uses TP DependencyGraph so all linked inputs/BN are sliced consistently.
    """
    # ensure grids match the dummy size if your head caches them
    if hasattr(model, "set_grid_xy"):
        H = example_inputs.shape[-1]
        model.set_grid_xy(H)

    DG = tp.DependencyGraph().build_dependency(model, example_inputs=example_inputs)

    # enable autograd for TP
    for p in model.parameters():
        p.requires_grad_(True)

    changed = {}
    with torch.enable_grad():
        for conv in layers:
            oc = conv.out_channels
            new_oc = (oc // k) * k
            drop = oc - new_oc
            if drop <= 0:
                continue
            # prune the last `drop` filters (any consistent index set is fine here)
            idxs = list(range(oc - drop, oc))
            group = DG.get_pruning_group(conv, tp.prune_conv_out_channels, idxs=idxs)
            if DG.check_pruning_group(group):
                group.prune()
                changed[conv] = (oc, conv.out_channels)
            else:
                print(f"[SKIP] could not prune {conv} to multiple of {k}.")
    return changed

def build_ratio_dict_for_heads(model, head_ratio=0.30):
    ratio_dict = {}
    for conv in get_head_3x3_convs(model):
        ratio_dict[conv] = head_ratio  # ask the pruner to remove ~30% from each head 3×3
    return ratio_dict

def head_out_channels(model):
    import torch.nn as nn
    names = dict(model.named_modules())
    vals = {}
    for k in ["head.detect_s.conv.conv.0", "head.detect_m.conv.conv.0", "head.detect_l.conv.conv.0"]:
        m = names.get(k, None)
        if isinstance(m, nn.Conv2d):
            vals[k] = m.out_channels
    return vals


def round_layers_to_multiple(model, example_inputs, layers, k: int = 16):
    """
    For each Conv2d in `layers`, round down out_channels to nearest multiple of k.
    Uses TP DependencyGraph so all linked inputs/BN are sliced consistently.
    """
    # ensure grids match the dummy size if your head caches them
    if hasattr(model, "set_grid_xy"):
        H = example_inputs.shape[-1]
        model.set_grid_xy(H)

    DG = tp.DependencyGraph().build_dependency(model, example_inputs=example_inputs)

    # enable autograd for TP
    for p in model.parameters():
        p.requires_grad_(True)

    changed = {}
    with torch.enable_grad():
        for conv in layers:
            oc = conv.out_channels
            new_oc = (oc // k) * k
            drop = oc - new_oc
            if drop <= 0:
                continue
            # prune the last `drop` filters (any consistent index set is fine here)
            idxs = list(range(oc - drop, oc))
            group = DG.get_pruning_group(conv, tp.prune_conv_out_channels, idxs=idxs)
            if DG.check_pruning_group(group):
                group.prune()
                changed[conv] = (oc, conv.out_channels)
            else:
                print(f"[SKIP] could not prune {conv} to multiple of {k}.")
    return changed




def fuse_conv_bn_(conv: nn.Conv2d, bn: nn.BatchNorm2d):
    with torch.no_grad():
        w = conv.weight
        if conv.bias is None:
            conv.bias = nn.Parameter(torch.zeros(w.size(0), device=w.device))
        b = conv.bias

        gamma, beta = bn.weight, bn.bias
        mean, var, eps = bn.running_mean, bn.running_var, bn.eps
        std = (var + eps).sqrt()

        w.mul_(gamma.view(-1,1,1,1) / std.view(-1,1,1,1))
        b.add_((-mean * gamma / std) + beta)

def _fuse_in_sequential(seq: nn.Sequential) -> nn.Sequential:
    """
    Scan an nn.Sequential and fold every Conv+BN *adjacent* pair.
    Returns a new nn.Sequential without any BN next to Conv.
    Idempotent: calling twice is safe.
    """
    layers = list(seq.children())
    new_layers = []
    i = 0
    while i < len(layers):
        a = layers[i]
        if i + 1 < len(layers) and isinstance(a, nn.Conv2d) and isinstance(layers[i+1], nn.BatchNorm2d):
            b = layers[i+1]
            fuse_conv_bn_(a, b)
            new_layers.append(a)           # keep fused conv
            i += 2                         # skip BN
        else:
            new_layers.append(a)
            i += 1
    return nn.Sequential(*new_layers)

def fuse_model(m: nn.Module):
    """
    Recursively:
      - fuse Conv+BN inside any nn.Sequential
      - handle your Conv wrapper by fusing inside its `.conv` Sequential
    """
    for name, child in list(m.named_children()):
        # Case A: your Conv wrapper (units.Conv) that holds a Sequential at .conv
        if hasattr(child, "conv") and isinstance(child.conv, nn.Sequential):
            child.conv = _fuse_in_sequential(child.conv)

        # Case B: plain sequentials anywhere else in the tree
        if isinstance(child, nn.Sequential):
            setattr(m, name, _fuse_in_sequential(child))

        # Recurse down
        fuse_model(child)

# --- sanity helpers ---
def count_bn(model): 
    return sum(isinstance(x, nn.BatchNorm2d) for x in model.modules())

def collect_ignored_convs(model, keep_stem=False, keep_stage_entry=False):
    ignored = set()
    for name, m in model.named_modules():
        if not isinstance(m, nn.Conv2d):
            continue

        if name.endswith(".detect"):
            ignored.add(m)
            continue

        if ".res_block" in name and ".conv2.conv.0" in name:
            ignored.add(m)
            continue

        if keep_stem and name == "backbone.conv1.conv.0":
            ignored.add(m)
            continue

        if keep_stage_entry and (name.endswith(".res_block1.conv.conv.0")
                                 or name.endswith(".res_block2.conv.conv.0")
                                 or name.endswith(".res_block3.conv.conv.0")
                                 or name.endswith(".res_block4.conv.conv.0")
                                 or name.endswith(".res_block5.conv.conv.0")):
            ignored.add(m)
            continue

    return list(ignored)

@torch.no_grad()
def bn_recalibration(model, loader, iters=400, device="cpu", only_bn_train=True, momentum=0.01):
    # Put only BN layers in train (so running stats update), keep rest eval
    if only_bn_train:
        model.eval()
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.train()
    else:
        model.train()

    # Temporarily lower BN momentum for smoother estimates
    orig_mom = []
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            orig_mom.append(m.momentum)
            m.momentum = momentum if m.momentum is None else min(momentum, m.momentum)


    for seen, (_, imgs, _, _) in enumerate(
            tqdm(loader, total=min(iters, len(loader)), desc="BN recal", unit="batch"),
            start=1):
        imgs = imgs.to(device, non_blocking=True)
        _ = model(imgs)
        tqdm.write(f"seen={seen}")   # optional line-print; bar still shows progress
        if seen >= iters:
            break

    # Restore momentum and set eval
    i = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = orig_mom[i]; i += 1
    model.eval()

@torch.no_grad()
def validate(args,anchors, dataloader, model, evaluator, dpu:bool = False,save_result=False,save_filename:str = "predictions.txt"):
    model.eval()
    if not dpu :
        model.module.set_grid_xy(input_size=args.img_size) if hasattr(model, "module") else model.set_grid_xy(input_size=args.img_size)

    with open(args.mAP_filepath, mode="r") as f:
        mAP_json = json.load(f)

    cocoPred = []
    check_images, check_preds, check_results = [], [], []
    imageToid = mAP_json["imageToid"]

    for _, minibatch in enumerate(dataloader):
        filenames, images, shapes = minibatch[0], minibatch[1], minibatch[3]
        predictions = model(images.to("cpu"))
        if dpu:
            decoded_52 = BoxDecoder(predictions[0],torch.tensor(anchors[0:3])).decode_predictions()
            decoded_26 = BoxDecoder(predictions[1],torch.tensor(anchors[3:6])).decode_predictions()
            decoded_13 = BoxDecoder(predictions[2],torch.tensor(anchors[6:9])).decode_predictions()
            predictions = torch.cat((decoded_52,decoded_26,decoded_13),dim = 1)
        # cuda(args.rank, non_blocking=True)

        for j in range(len(filenames)):
            prediction = predictions[j].cpu().numpy()

            prediction[:, 1:5] = transform_xcycwh_to_x1y1x2y2(boxes=prediction[:, 1:5], clip_max=1.0)
            prediction = filter_confidence(prediction=prediction, conf_threshold=args.conf_thres)
            prediction = run_NMS(prediction=prediction, iou_threshold=args.nms_thres)

            if len(check_images) < 5:
                check_images.append(to_image(images[j]))
                check_preds.append(prediction.copy())
                
            if len(prediction) > 0:
                filename = filenames[j]
                shape = shapes[j]
                cls_id = prediction[:, [0]]
                conf = prediction[:, [-1]]
                box_x1y1x2y2 = scale_coords(img1_shape=images.shape[2:], coords=prediction[:, 1:5], img0_shape=shape[:2])
                box_x1y1wh = transform_x1y1x2y2_to_x1y1wh(boxes=box_x1y1x2y2)
                img_id = np.array((imageToid[filename],) * len(cls_id))[:, np.newaxis]
                cocoPred.append(np.concatenate((img_id, box_x1y1wh, conf, cls_id), axis=1))

    del images, predictions
    torch.cuda.empty_cache()

    if len(cocoPred) > 0:
        cocoPred = np.concatenate(cocoPred, axis=0)
        mAP_dict, eval_text = evaluator(predictions=cocoPred)

        if save_result:
            np.savetxt(args.exp_path / save_filename, cocoPred, fmt="%.4f", delimiter=",", header=f"Inference results of [image_id, x1y1wh, score, label]") 
            out_file = args.exp_path / save_filename
            with open(out_file, "a") as f:
                f.write(eval_text)
        return mAP_dict, eval_text
    else:
        return None, None

def get_dataloader(voc_path,batch_size,same_subset = False, subset_length = 8,train = False,input_size = 416,mAP_filename = "eval.json"):
    if not train:
        val_dataset = Dataset(voc_path,phase = 'val')
        if same_subset:
            k = len(val_dataset)
        else:
            k = subset_length
        print(len(val_dataset))
        subset_index = list(range(min(k,len(val_dataset))))
        val_transform = BasicTransform(input_size = input_size)
        val_dataset.load_transformer(val_transform)
        val_subset = Subset(val_dataset,indices = subset_index)
        idx = list(val_subset.indices)
        val_subset.dataset.generate_mAP_source(save_dir = Path("./data/eval_src"),mAP_filename = mAP_filename,indices = idx)
        
        return DataLoader(dataset=val_subset, collate_fn=Dataset.collate_fn, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=1,drop_last=True)
    
    else:
        train_dataset = Dataset(voc_path,phase = "train")
        if same_subset:
            k = len(train_dataset)
        else:
            k = subset_length
        subset_index = list(range(min(k,len(train_dataset))))
        train_transform = AugmentTransform(input_size = input_size,dataset = train_dataset)
        train_dataset.load_transformer(train_transform)
        train_subset = Subset(train_dataset,indices = subset_index)
        return DataLoader(dataset=train_subset, collate_fn=Dataset.collate_fn, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=1,drop_last=True)
    

def load_model(mode:str,device: str,input_size: int, num_classes: int,model_type: str,anchors,model_path:str):
    """
    Initialize YOLO model and load checkpoint.
    mode:"normal" -> uninspected model,some layers might not be supported in the dpu
          "dpu" -> dpu supported model
          "dpu_quantized" -> Quantized Model in DPU.
         
    """
    if mode == "dpu":
        model = YOLOv3_DPU(input_size = input_size, num_classes = num_classes, anchors = anchors, model_type = model_type, pretrained = False).to(device)
        ckpt = torch.load(model_path,map_location = 'cpu',weights_only = False)
        sd = ckpt["model_state"] if "model_state" in ckpt else ckpt
        missing,unexpected = model.load_state_dict(sd,strict = True)
        model.set_grid_xy(input_size = input_size)
        print(f"[load] missing = {len(missing)} unexpected = {len(unexpected)}")
    
    elif mode == "normal":
        model = YOLOv3(input_size = input_size, num_classes = num_classes, anchors = anchors, model_type = model_type, pretrained = False).to(device)
        ckpt = torch.load(model_path,map_location = 'cpu',weights_only = False)
        sd = ckpt["model_state"] if "model_state" in ckpt else ckpt
        missing,unexpected = model.load_state_dict(sd,strict = True)
        model.set_grid_xy(input_size = input_size)
        print(f"[load] missing = {len(missing)} unexpected = {len(unexpected)}")
    elif mode == "dpu_quantized":
        model = torch.jit.load(model_path,map_location=device)
    
    model.to(device).eval()
    model.zero_grad() 
    return model



