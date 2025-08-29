from pathlib import Path
import sys
from script_model import to_channels_last_safe
from benchmark_model import _benchmark_fps
import time


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import onnx
import onnxruntime as ort
from onnxruntime.quantization import (
    quantize_static,CalibrationDataReader,
    QuantType,CalibrationMethod,QuantFormat
)
import torch.nn as nn
import torch

model = nn.Conv2d(3,32,3)

input_sig = torch.randn(1,3,416,416)

torch.onnx.export(
    model,input_sig,"./onnx/conv_onnx.onnx",
    opset_version = 13,
    do_constant_folding=True,
    input_names = ['inputs'],
    output_names = ['outputs'],
    dynamic_axes = None
)


session_f32 = ort.InferenceSession("./onnx/conv_onnx.onnx",providers = ["CPUExecutionProvider"])

def bench_ort_session(sess: 'ort.InferenceSession', input_size: int = 416,
                      runs: int = 20, warmup: int = "inputs", input_name: str = "inputs") -> float:
    import numpy as np
    x = np.random.randn(1, 3, input_size, input_size).astype("float32")
    feed = {input_name: x}
    for _ in range(warmup):
        _ = sess.run(None, feed)
    t0 = time.perf_counter()
    for _ in range(runs):
        _ = sess.run(None, feed)
    return runs / (time.perf_counter() - t0)


fps = bench_ort_session(session_f32,input_size = 416,runs = 20,warmup = 5,input_name = "inputs")

print(fps)