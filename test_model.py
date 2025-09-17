import torch

model = torch.load('/home/saurav/Desktop/Internship/ML-Internship-Saurav-Paudel/Paper_Implementation/ObjectDetection/UniYOLO/weights/yolov3-base.pt',map_location='cpu')

print(model.keys())