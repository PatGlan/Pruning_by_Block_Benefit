import torch
import argparse
from utilities.args import get_args_parser

from data.datasets import get_dataloader
from models.model_loader import get_student_model

import os

def convert_form_torch_pretrained(args):
    #argument
    #--model-path ../models/checkpoints/torch/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz


    assert ".npz" in args.model_path or args.model_pretrained

    data_loader_train, data_loader_val, dataset_train, dataset_val = get_dataloader(args)
    model, gate_func_added = get_student_model(args)

    # naming
    save_path = os.path.abspath(__file__)
    save_path = os.path.dirname(os.path.dirname(save_path))
    save_path += "/models/checkpoints/torch"
    fileName = "vitB16_i21k_224.pth"
    file_path = os.path.join(save_path, fileName)

    # save model
    torch.save({"model": model.state_dict()}, file_path)

def convert_from_dino(args):
    path_backbone="/home/glandorf/projects/FoundDist/models/checkpoints/dinov2/dinov2_vitb14_pretrain.pth"
    path_head="/home/glandorf/projects/FoundDist/models/checkpoints/dinov2/dinov2_vitb14_linear_head.pth"
    save_path="/home/glandorf/projects/FoundDist/models/checkpoints/dinov2"

    data_loader_train, data_loader_val, dataset_train, dataset_val = get_dataloader(args)
    model, gate_func_added = get_student_model(args)

    cp_backbone = torch.load(path_backbone, map_location='cpu')

    cp_head = torch.load(path_head, map_location='cpu')
    cp_head = {"head." + k:v for k, v in cp_head.items()}

    cp_new = {**cp_backbone, **cp_head}


    model_dino_torchub = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model="dinov2_vitb14")


    #test if checkpints fits to model
    cp_model = model.state_dict()
    #model.load_state_dict(cp_new, strict=True)


    y=1



if __name__ == '__main__':
    parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    convert_type = "dinov2" #["torch, "dino"]
    if convert_type == "torch":
        convert_form_torch_pretrained(args)
    elif convert_type == "dinov2":
        convert_from_dino(args)
    else:
        assert False, "wrong convert type"
    y=1


