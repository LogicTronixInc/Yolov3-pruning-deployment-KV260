from cls_net_utils import build_vgg19,cifar10_loaders,train,is_full_model_file
#torch essentials
import torch
import torch.nn as nn

#other utils
from typing import Tuple, Optional
from pathlib import Path
import argparse



def main():
    parser = argparse.ArgumentParser(description="Training Script for VGG19/ResNet Model")
    parser.add_argument('--model',required = False,type = str)
    parser.add_argument("--data-dir", type = str,default = "./data")
    parser.add_argument("--epochs",type = int,default =10)
    parser.add_argument("--batch-size",type = int,default = 128)
    parser.add_argument("--workers",type = int,default = 4)
    parser.add_argument("--lr",type = float,default = 0.1)
    parser.add_argument("--momentum",type = float,default = 0.9)
    parser.add_argument("--weight-decay",type = float,default = 5e-4)
    parser.add_argument("--input-size",type = int,default = 224)
    parser.add_argument("--mode",required=False,type = str, help = "float or pruned")
    parser.add_argument("--finetune-model",action = 'store_true')

    parser.add_argument('--scratch', action='store_true', help='float checkpoint (state_dict) or a full model .pth after pruning')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, test_loader = cifar10_loaders(args.data_dir, args.batch_size, args.workers, args.input_size)

    print(args.finetune_model)
    print(args.scratch)
    if (args.scratch == True) and (args.finetune_model == False):
        # fresh / float training
        mode = "float"
        model = build_vgg19(num_classes=10, pretrained=True)
        model.to(device)
        train(args,model,train_loader,test_loader,device,mode = mode)
        print(f"Saved Float Checkpoint:best_{mode}.pth")

    else:
        resume_path = Path(args.model)
        mode = "pruned"

        print(is_full_model_file(resume_path))

        if is_full_model_file(resume_path):
            print(f"Loading full model object from: {resume_path}")
            model = torch.load(str(resume_path), map_location='cpu')
            model.to(device)
        else:
            print(f"Loading state_dict from: {resume_path}")
            model = build_vgg19(num_classes=10, pretrained=True)
            ckpt = torch.load(str(resume_path), map_location='cpu')
            model.load_state_dict(ckpt['state_dict'])
            model.to(device)
        
        train(args,model,train_loader,test_loader,device,mode = mode)
        print(f"Saved Pruned Checkpoint: best_{mode}.pth")


if __name__ == "__main__":
    main()
