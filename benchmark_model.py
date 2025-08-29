import argparse,time,os
import torch,torch.nn as nn
from ptflops import get_model_complexity_info
from pathlib import Path
import sys
from script_model import to_channels_last_safe
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import model
import onnx
import onnxruntime as ort
from onnxruntime.quantization import (
    quantize_static,CalibrationDataReader,
    QuantType,CalibrationMethod,QuantFormat
)

@torch.no_grad()
def _benchmark_fps(model, input_size=608, runs=50, warmup=10, device="cpu",channels_last:bool = False,get_params:bool = True):
    model.eval().to(device)
    x = torch.randn(1, 3, input_size, input_size, device=device)
    macs,params = None,None

    if get_params:
        macs,params = get_model_complexity_info(model,(3,input_size,input_size),as_strings = False,print_per_layer_stat=False,verbose = True)
    if channels_last:
        x = x.to(memory_format=torch.channels_last)
    # warmup
    for _ in range(warmup):
        _ = model(x)
    torch.cuda.synchronize() if device.startswith("cuda") else None
    t0 = time.time()
    for _ in range(runs):
        _ = model(x)
    torch.cuda.synchronize() if device.startswith("cuda") else None
    dt = time.time() - t0
    fps = runs / dt
    return macs,params,fps

@torch.no_grad()
def benchmark_fps_saved_pruned(model_path,input_size = 416,runs = 20,warmup = 10,device = "cpu"):
    model = torch.load(model_path,map_location = "cpu",weights_only = False)
    macs,params,fps = _benchmark_fps(model = model,input_size = input_size,runs = runs,warmup = warmup,device = device)
    print(f"MACS: {macs/1e9:.2f}G")
    print(f"Nparam:{params/1e6:.2f}")
    print(f"FPS:{fps}")
    return macs,params,fps


@torch.no_grad()
def benchmark_fps_saved_torchscript(model_path,input_size = 416,runs = 20,warmup = 5,device = "cpu"):
    scripted_model = torch.jit.load(model_path)
    scripted_model = to_channels_last_safe(scripted_model)
    scripted_model = torch.jit.optimize_for_inference(scripted_model)
    _,_,fps = _benchmark_fps(scripted_model,input_size = input_size,runs = runs,warmup=warmup,device = device,channels_last=True,get_params=False)
    
    print(f"FPS:{fps}")

    return _,_,fps

def ort_session(path,intra = 4,inter = 1):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.enable_mem_pattern = True
    so.enable_cpu_mem_arena = True
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    providers = [
        (
            "CPUExecutionProvider", {
                "intra_op_num_threads":intra,
                "inter_op_num_threads":inter
            }
        )
    ]    
    return ort.InferenceSession(path,sess_options = so,providers = providers)

def bench_ort_session(sess: 'ort.InferenceSession', input_size: int = 416,
                      runs: int = 20, warmup: int = 5, input_name: str = "images") -> float:
    import numpy as np
    x = np.random.randn(1, 3, input_size, input_size).astype("float32")
    feed = {input_name: x}
    for _ in range(warmup):
        _ = sess.run(None, feed)
    t0 = time.perf_counter()
    for _ in range(runs):
        _ = sess.run(None, feed)
    return runs / (time.perf_counter() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=False, help="path to trained yolov3 .pt")
    ap.add_argument("--input_size", type=int, default=416)
    ap.add_argument("--mode",type = str, default = 'pruned',help = "`pruned` for prune-only model `prune-script` for pruned-script model `script` for normal script model ")
    ap.add_argument("--num_classes", type=int, default=20)
    ap.add_argument("--model_type", type=str, default="base")  # matches your naming
    args = ap.parse_args()


    if args.mode == 'pruned':
        benchmark_fps_saved_pruned(model_path = args.model,input_size=args.input_size)

    elif args.mode == 'pruned-script':
        benchmark_fps_saved_torchscript(model_path = args.model,input_size=args.input_size)
    
    elif args.mode == 'script':
        benchmark_fps_saved_torchscript(model_path = args.model,input_size=args.input_size)

    elif args.mode == 'onnx':
        session_f32 = ort_session('/home/saurav/Desktop/Internship/ML-Internship-Saurav-Paudel/Paper_Implementation/ObjectDetection/UniYOLO/VainF_pruning/onnx/yolov3.onnx')
        
        # x = np.random.randn(1, 3, 416, 416).astype("float32")
        # feed = {"images":x}

        # output = session_f32.run(None,feed)
        # print(output[0].shape)
        
        
        
        
        fps = bench_ort_session(session_f32,warmup=10,runs = 30)
        print(fps)
        

if __name__ == "__main__":
    main()





