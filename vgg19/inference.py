import argparse
import time
from pathlib import Path
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T

import numpy as np

from cls_net_utils import build_vgg19   


CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


def build_transform(input_size: int):
    # CIFAR10 normalization
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2470, 0.2435, 0.2616]

    return T.Compose([
        T.Resize((input_size, input_size)),
        T.ToTensor(),
        T.Normalize(mean, std)
    ])

def load_model(model_path: Path, device: torch.device, num_classes=10):
    try:
        obj = torch.load(str(model_path), map_location="cpu",weights_only=False)
        if isinstance(obj, nn.Module):
            print(f"[Load] Full model detected at {model_path}")
            model = obj
        else:
            print(f"[Load] state_dict detected at {model_path}")
            model = build_vgg19(num_classes=num_classes, pretrained=True)
            state_dict = obj.get("state_dict", obj)
            model.load_state_dict(state_dict)
            
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")

    model.eval().to(device)
    return model


def main():
    parser = argparse.ArgumentParser(description="Single image inference with FPS measurement")
    parser.add_argument("--model", type=str, required=True, help="Path to model (.pth)")
    parser.add_argument("--image", type=str, required=True, help="Path to image file")
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--repeat", type=int, default=100, help="Number of runs for FPS test")
    args = parser.parse_args()

    device = torch.device(args.device)
    model = load_model(Path(args.model), device)

    transform = build_transform(args.input_size)
    img = Image.open(args.image).convert("RGB")
    
    x = transform(img).unsqueeze(0).to(device)

    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(x)

    # Timing loop
    torch.cuda.synchronize() if device.type == "cuda" else None
    start = time.time()
    with torch.no_grad():
        for _ in range(args.repeat):
            logits = model(x)
    torch.cuda.synchronize() if device.type == "cuda" else None
    end = time.time()

    fps = args.repeat / (end - start)

    # Final prediction and image display
    probs = F.softmax(logits, dim=1)[0]
    pred_idx = probs.argmax().item()
    pred_name = CIFAR10_CLASSES[pred_idx]

    print(f"[Result] Predicted class: {pred_name} (idx={pred_idx}, prob={probs[pred_idx]:.4f})")
    print(f"[Perf] FPS: {fps:.2f} (averaged over {args.repeat} runs)")
    

    disp_img = cv2.resize(np.array(img),(224,224))
    
    cv2.putText(disp_img,f"{pred_name}: {probs[pred_idx]:.4f}",(10,30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0,255,0),
                2)
    cv2.putText(disp_img,f"FPS: {fps:.2f}",(10,60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0,255,0),
                2)
    cv2.imshow("Image", disp_img)
    cv2.waitKey((0))


if __name__ == "__main__":
    main()
