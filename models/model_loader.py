
import timm
import torch
import numbers
from timm.models import create_model
#from timm.models import vision_transformer
#import math
import models.arch #as VisionTransformer
import numpy as np
from copy import deepcopy
from torch import nn

from utils.model_utils import get_module

def prepare_checkpoint(cp, model):
    if "model" in cp.keys():
        cp = cp["model"]
    return cp

def get_name_head(model, is_dist_token=False):
    search_string = "head_dist" if is_dist_token else "head"
    return ([None] + [name for name, m in model.named_modules() if name.endswith(search_string) and "backbone" not in name])[-1]

def prepare_model_head(model, args):
    stud_state_dict = model.state_dict()

    num_features = model.num_features if hasattr(model, "num_features") else model.backbone.num_features #* 2
    num_classes = args.nb_classes

    name_head = get_name_head(model, False)
    isIdentitiy = getattr(model, name_head).__class__.__name__ == 'Identity'
    if isIdentitiy or stud_state_dict[name_head + ".weight"].shape[0] != num_classes:
        new_head = nn.Linear(num_features, num_classes)
        setattr(model, name_head, new_head)
        stud_state_dict = model.state_dict()

    name_dist_head = get_name_head(model, True)
    isIdentitiy = getattr(model, name_dist_head).__class__.__name__ == 'Identity' if name_dist_head is not None else False
    if name_dist_head is not None and args.distillation_type != "none" and \
            (isIdentitiy or  stud_state_dict[name_dist_head + ".weight"].shape[0] != num_classes):
        new_dist_head = nn.Linear(num_features, num_classes)
        setattr(model, "head_dist", new_dist_head)


    if any([name.endswith("mask_token") for name, p in model.named_parameters()]):
        del(model.mask_token)

    return model

def add_head_2_checkpoint(model, checkpoint, args):
    stud_state_dict = model.state_dict()

    name_head = get_name_head(model, False)
    if name_head + ".weight" not in checkpoint.keys() or \
            checkpoint[name_head + ".weight"].shape[0] != stud_state_dict[name_head + ".weight"].shape[0]:
        checkpoint[name_head + ".weight"] = torch.ones_like(stud_state_dict[name_head + ".weight"]).normal_(mean=0.0, std=0.01)
        checkpoint[name_head + ".bias"] = torch.zeros_like(stud_state_dict[name_head + ".bias"])

    name_dist_head = get_name_head(model, True)
    if args.distillation_type != "none" and (name_dist_head + ".weight" not in checkpoint.keys() or \
                                                  checkpoint[name_dist_head + ".weight"].shape[0] != stud_state_dict[name_dist_head + ".weight"].shape[0]):
        checkpoint[name_dist_head + ".weight"] = deepcopy(checkpoint[name_dist_head + ".weight"])
        checkpoint[name_dist_head + ".bias"] = deepcopy(checkpoint[name_dist_head + ".weight"])

    #delete mask token
    if any([x.endswith("mask_token") for x in checkpoint.keys()]):
        name_mask_token = [k for k in checkpoint.keys() if k.endswith("mask_token")][-1]
        del(checkpoint[name_mask_token])

    return checkpoint




def interpolate_pos_embedding(checkpoint, backbone):
    # interpolate position embedding
    name_pos_embed = [k for k in checkpoint.keys() if k.endswith("pos_embed")][-1]
    pos_embed_checkpoint = checkpoint[name_pos_embed]
    embedding_size = pos_embed_checkpoint.shape[-1]
    num_patches = backbone.patch_embed.num_patches
    num_extra_tokens = backbone.pos_embed.shape[-2] - num_patches
    # height (== width) for the checkpoint position embedding
    orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
    # height (== width) for the new position embedding
    new_size = int(num_patches ** 0.5)
    # class_token and dist_token are kept unchanged
    extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
    # only the position tokens are interpolated
    pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
    pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
    pos_tokens = torch.nn.functional.interpolate(
        pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
    pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
    new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
    checkpoint[name_pos_embed] = new_pos_embed

    return checkpoint


def _extend_distill_token_pos_embed(checkpoint, model, add_dist_token=False):
    name_pos_embed = [k for k in checkpoint.keys() if k.endswith("pos_embed")][-1]
    cls_embed = checkpoint[name_pos_embed][:, 0:1]
    patch_embed = checkpoint[name_pos_embed][:, 1:]

    '''
    #adjust size pos-embedding if image-size is smaller
    shape_pos_cp = list(checkpoint["pos_embed"].shape)
    shape_pos_cp[1] -= 1
    shape_pos_cp[1] -= shape_pos_cp[1] % 2
    shape_pos_model = list(model.pos_embed.shape)
    shape_pos_model[1] -= 1
    shape_pos_model[1] -= shape_pos_model[1] % 2
    pos_equal_shape = all([x == y for x,y in zip(shape_pos_cp, shape_pos_model)])
    if not pos_equal_shape:
        patch_embed = torch.ones_like(model.pos_embed).normal_(mean=0.0, std=0.01)
        patch_embed = patch_embed[:, 1:, :]
        #patch_embed = patch_embed[:, :shape_pos_model[1], :]
    '''
#
    #copy use cls-token for distill-token
    if add_dist_token:
        checkpoint[name_pos_embed] = torch.cat([cls_embed, cls_embed, patch_embed], dim=1)
    else:
        checkpoint[name_pos_embed] = torch.cat([cls_embed, patch_embed], dim=1)

    return checkpoint

def resize_model_by_checkpoint(model, checkpoint):

    for name, w_cpt in checkpoint.items():
        m = get_module(model, name.split("."))

        if not all([i_cht == i_model for i_cht, i_model in zip(w_cpt.shape, m.shape)]):
            m_parent = get_module(model, name.split(".")[:-1])
            name_child = name.split(".")[-1]
            setattr(m_parent, name_child, nn.Parameter(torch.rand_like(w_cpt)))

            #update normalized shape parameter
            if "norm" in name:
                normalized_shape = w_cpt.shape[0]
                if isinstance(normalized_shape, numbers.Integral):
                    # mypy error: incompatible types in assignment
                    normalized_shape = (normalized_shape,)  # type: ignore[assignment]
                m_parent.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
                y=1

            y=1

    #test new shape model
    for name, w_cpt in checkpoint.items():
        m = get_module(model, name.split("."))
        assert all([i_cht == i_model for i_cht, i_model in zip(w_cpt.shape, m.shape) ])

    return model


def get_student_model(args, logger=None):

    if "dinov2" in args.model: #example: dinov2_vitb14
        student_model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=args.model)
        embed_size = student_model.embed_dim
    else:
        n_cls = args.nb_classes if not args.model_pretrained else 1000
        cpt_path = args.model_path if (args.model_path is not None and".npz" in args.model_path) else ''
        student_model = create_model(
            args.model,
            pretrained=args.model_pretrained,
            checkpoint_path=cpt_path,
            num_classes=n_cls,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            drop_block_rate=None,
            img_size=args.input_size
        )
        embed_size = student_model.embed_dim


    #add num_qkv_embed
    for i, block in enumerate(student_model.blocks):
        block.attn.num_qkv_embed = block.attn.qkv.out_features // 3 // block.attn.num_heads

    student_model = prepare_model_head(student_model, args)

    #get checkpoint
    gate_func_added = False
    if args.model_path is not None and ".npz" not in args.model_path:
        if args.model_path.startswith('https'):
            checkpoint_full = torch.hub.load_state_dict_from_url(
                args.model_path, map_location='cpu', check_hash=True)
        else:
            checkpoint_full = torch.load(args.model_path, map_location='cpu')

        checkpoint = prepare_checkpoint(checkpoint_full, student_model)
        checkpoint = _extend_distill_token_pos_embed(checkpoint, student_model, add_dist_token=False)

        checkpoint = interpolate_pos_embedding(checkpoint, student_model)

        #checkpoint = _extend_distill_token_pos_embed(checkpoint, student_model,add_dist_token=False)
        checkpoint = add_head_2_checkpoint(student_model, checkpoint, args)
        if "dist_token" in student_model.state_dict().keys():
            checkpoint["dist_token"] = torch.zeros(1, 1, embed_size)

        for key in checkpoint.keys():
            if "gate" in key:
                gate_func_added = True

        #update checkpoint for soft-classifier
        student_state_dict = student_model.state_dict()
        checkpoint = {k:v for k,v in checkpoint.items() if "fmHead" not in k or k in student_state_dict.keys()}
        checkpoint = {k:v for k,v in checkpoint.items() if "_gate" not in k or k in student_state_dict.keys()}


        if args.model_resize_by_checkpoint:
            student_model = resize_model_by_checkpoint(student_model, checkpoint)


        student_model.load_state_dict(checkpoint, strict=True)

        if "num_heads" in checkpoint_full.keys(): #to reconstrucmt sparse attention
            for i, block in enumerate(student_model.blocks):
                block.attn.num_heads = checkpoint_full["num_heads"][i]

        if "num_qkv_embed" in checkpoint_full.keys(): #to reconstrucmt sparse attention
            for i, block in enumerate(student_model.blocks):
                block.attn.num_qkv_embed = checkpoint_full["num_qkv_embed"][i]

        if "masks_block" in checkpoint_full.keys() and f"0_attn_input" in checkpoint_full["masks_block"].keys(): #to reconstrucmt input mask of block
            for i, block in enumerate(student_model.blocks):
                #if checkpoint_full["masks_block"][f"{i}_attn_input"] is not None:
                block.attn.input_mask = checkpoint_full["masks_block"][f"{i}_attn_input"]
                if block.attn.input_mask is not None:
                    block.attn.input_mask.cuda()

                block.attn.output_mask = checkpoint_full["masks_block"][f"{i}_attn_output"]
                if block.attn.output_mask is not None:
                    block.attn.output_mask.cuda()

                block.mlp.input_mask = checkpoint_full["masks_block"][f"{i}_mlp_input"]
                if block.mlp.input_mask is not None:
                    block.mlp.input_mask.cuda()

                block.mlp.output_mask = checkpoint_full["masks_block"][f"{i}_mlp_output"]
                if block.mlp.output_mask is not None:
                    block.mlp.output_mask.cuda()
        else:
            for i, block in enumerate(student_model.blocks):
                block.attn.input_mask = None
                block.attn.output_mask = None
                block.mlp.input_mask = None
                block.mlp.output_mask = None

        if logger is not None:
            logger.info(f"loaded model from: {args.model_path}")


    if args.model_freeze:
        for name, param in student_model.named_parameters():
            param.requires_grad = False

    return student_model, gate_func_added

def get_teacher_model(args):

    if args.use_timm_model:
        n_cls = args.nb_classes if not args.model_pretrained else 1000
        teacher_model = create_model(
            args.teacher_model,
            pretrained=args.teacher_model_pretrained,
            num_classes=n_cls,
            img_size=args.input_size
        )
        if args.model_pretrained and args.nb_classes != 1000:
            teacher_model.head = nn.Linear(teacher_model.embed_dim, args.nb_classes)
    else:
        teacher_model = VisionTransformer.__dict__[args.model_arch](img_size=[args.input_size], patch_size=16, in_chans=3,
                                          num_classes=args.nb_classes, embed_dim=768, depth=12,
                                          num_heads=12,
                                          mlp_ratio=4.,
                                          qkv_bias=True)
    #get checkpoint
    if args.teacher_path.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            args.teacher_path, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(args.teacher_path, map_location='cpu')
    checkpoint = prepare_checkpoint(checkpoint)
    #checkpoint = _extend_distill_token_pos_embed(checkpoint, teacher_model, add_dist_token=False)
    checkpoint = interpolate_pos_embedding(checkpoint, teacher_model)
    checkpoint = add_head_2_checkpoint(teacher_model, checkpoint)


    teacher_model.load_state_dict(checkpoint, strict=True)

    return teacher_model