from pathlib import Path
import sys

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
from model import YOLOv3
from inference import run_inference,load_model
from pruning_utils import fuse_model

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

anchors = [[0.248, 0.7237237 ],
    [0.36144578, 0.53      ],
    [0.42,       0.9306667 ],
    [0.456,      0.6858006 ],
    [0.488,      0.8168168 ],
    [0.6636637,  0.274     ],
    [0.806,      0.648     ],
    [0.8605263,  0.8736842 ],
    [0.944,      0.5733333 ]]

model = build_model(input_size = 416,num_classes = 20,anchors = anchors,model_type='base',ckpt_path='/home/saurav/Desktop/Internship/ML-Internship-Saurav-Paudel/Paper_Implementation/ObjectDetection/UniYOLO/weights/yolov3-base.pt')

fuse_model(model)
print(model)

input_sig = torch.randn(1,3,416,416)
model.eval()
torch.onnx.export(
    model,input_sig,'./onnx/yolov3.onnx',
    opset_version=19,
    do_constant_folding = True,
    dynamo=True,
    optimize = True,
    input_names = ['images'],
    output_names = ['preds'],
    dynamic_axes = None,
    dump_exported_program = True,
    training=torch.onnx.TrainingMode.EVAL
)








# model_script = load_model(version = "V3",device = "cpu",input_size=416,num_classes = 20,model_type = "base",anchors = anchors,pretrained=False)
# run_inference(model,input_size= 416,source = '/home/saurav/Downloads/HotwifeAlice - 2 caged cucks their hotwives and 2 bulls - ManyVi.mp4',device = 'cpu',conf_thresh = 0.25,nms_iou_thresh=0.6)
