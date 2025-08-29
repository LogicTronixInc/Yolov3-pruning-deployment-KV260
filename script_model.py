import argparse,time,os
import torch,torch.nn as nn
from ptflops import get_model_complexity_info
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import model

def to_channels_last_safe(model: nn.Module) -> nn.Module:
    """Mark module NHWC; only reorder 4D tensors to channels_last (safe)."""
    model.to(memory_format=torch.channels_last)
    with torch.no_grad():
        for p in model.parameters():
            if p.dim() == 4:
                p.data = p.data.contiguous(memory_format=torch.channels_last)
            else:
                p.data = p.data.contiguous()
        for b in model.buffers():
            if b.dim() == 4:
                b.data = b.contiguous(memory_format=torch.channels_last)
            else:
                b.data = b.contiguous()
    return model

@torch.no_grad()
def script_model(model,input_size,filename:str,channels_last:bool = False,device:str = "cpu"):
    model.eval().to(device)
    ex = torch.randn(1,3,input_size,input_size,device = device)
    if channels_last:
        to_channels_last_safe(model)
        ex = ex.to(memory_format=torch.channels_last)
    scripted = torch.jit.trace(model,ex)
    # scripted = torch.jit.optimize_for_inference(scripted)
    torch.jit.save(scripted,filename)

    return scripted


def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True, help="path to trained yolov3 .pt")
    ap.add_argument("--input_size", type=int, default=416)
    ap.add_argument("--num_classes", type=int, default=20)
    ap.add_argument("--model_type", type=str, default="base")  # matches your naming
    ap.add_argument("--onnx", type=bool, default=False)
    args = ap.parse_args()
    model = torch.load(args.model,map_location="cpu",weights_only=False)

    scripted_model = script_model(model,input_size=args.input_size,channels_last = False,device= args.device)


if __name__ == "__main__":
    main()

        