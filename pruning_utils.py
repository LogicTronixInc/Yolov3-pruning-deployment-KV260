import torch.nn as nn
import torch
import torch_pruning as tp

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