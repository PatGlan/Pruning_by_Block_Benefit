
import timm.models.vision_transformer
import torch
import torch.nn as nn
import numpy as np
import math
from pruner.gating_layer import GateLayer, MaskModule
from pruner.pruner_utils import get_module
from .mask_wrapper import replace_linear

def forward_features_vit_base_patch16_224(self, x):
    B = x.shape[0]
    x = self.patch_embed(x)

    cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
    x = torch.cat((cls_tokens, x), dim=1)
    x = x + self.pos_embed
    x = self.pos_drop(x)

    x_inter = []
    x_inter.append(x)
    for i, blk in enumerate(self.blocks):
        x, x_attn, x_mlp = blk(x)
        x_inter.append(x_attn.to(torch.float))
        x_inter.append(x_mlp.to(torch.float))

    x = self.norm(x)
    return x, x_inter

def forward_vit_base_patch16_224(self, x):
    x, x_inter = self.forward_features(x)
    p = self.head(x[:,0])

    p_inter = None
    if hasattr(self, "fmHead"):
        p_inter = self.fmHead(x_inter)

    return p, p_inter, x_inter

def forward_features_deit_distilled_patch16_224(self, x):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications to add the dist_token
    B = x.shape[0]
    x = self.patch_embed(x)

    cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
    dist_token = self.dist_token.expand(B, -1, -1)
    x = torch.cat((cls_tokens, dist_token, x), dim=1)

    x = x + self.pos_embed
    x = self.pos_drop(x)

    x_inter = []
    x_inter.append(x)
    for blk in self.blocks:
        x, x_attn, x_mlp = blk(x)
        x_inter.append(x_attn.to(torch.float))
        x_inter.append(x_mlp.to(torch.float))

    x = self.norm(x)
    return x, x_inter

def forward_deit_distilled_patch16_224(self, x):
    x, x_inter = self.forward_features(x)
    p = self.head(x[:, 0])
    p_dist = self.head_dist(x[:, 1])

    p_inter = None
    if hasattr(self, "fmHead"):
        p_inter = self.fmHead(x_inter)

    if self.training:
        return p, p_dist, p_inter, x_inter
    else:
        # during inference, return the average of both classifier predictions
        return (p + p_dist) / 2, p_inter, x_inter


def forward_features_dinov2_base_patch16_224(self, x, masks=None):
    '''
    B, nc, w, h = x.shape
    x = self.patch_embed(x)
    if masks is not None:
        x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)

    x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
    x = x + self.interpolate_pos_encoding(x, w, h)

    if self.register_tokens is not None:
        x = torch.cat(
            (
                x[:, :1],
                self.register_tokens.expand(x.shape[0], -1, -1),
                x[:, 1:],
            ),
            dim=1,
        )
    '''

    x = self.prepare_tokens_with_masks(x)

    x_inter = []
    x_inter.append(x)
    for i, blk in enumerate(self.blocks):
        x, x_attn, x_mlp = blk(x)

        x_inter.append(x_attn.to(torch.float))
        x_inter.append(x_mlp.to(torch.float))

    x = self.norm(x)
    return x, x_inter

def forward_dinov2_patch16_224(self, x):
    x, x_inter = self.forward_features(x)
    p = self.head(x[:,0])

    p_inter = None
    if hasattr(self, "fmHead"):
        p_inter = self.fmHead(x_inter)

    return p, p_inter, x_inter


'''
forward function used for Vision Transformer modules
- block
- attention
- mlp
'''
from copy import deepcopy
def forward_block_vit(self, x):
    #attention
    #if detach_graph:
    #    x = x.detach()

    #if not(hasattr(self.attn, "skip_block") and self.attn.skip_block):
    res = x
    if hasattr(self.attn, "input_mask") and self.attn.input_mask is not None:
        x = x[:,:,self.attn.input_mask]
    x_int = self.norm1(x)
    x_int = self.attn(x_int)
    x_int = self.drop_path(x_int)
    if hasattr(self.attn, "output_mask") and self.attn.output_mask is not None:
        #x = res
        #x[:,:,self.attn.output_mask] = x[:,:,self.attn.output_mask] + x_int

        x = torch.zeros_like(res)
        x[:, : , self.attn.output_mask] = x[:, : , self.attn.output_mask] + x_int
        x = res + x
    else:
        x = x_int + res
    feat_attn = x
    #test_attn = feat_attn[0, 0].detach().cpu().numpy()

    #FFN
    #if not (hasattr(self.mlp, "skip_block") and self.mlp.skip_block):
    res = x
    if hasattr(self.mlp, "input_mask") and self.mlp.input_mask is not None:
        x = x[:,:,self.mlp.input_mask]
    x_int = self.norm2(x)
    x_int = self.mlp(x_int)
    x_int = self.drop_path(x_int)
    if hasattr(self.mlp, "output_mask") and self.mlp.output_mask is not None:
        #x = res
        #x[:,:,self.mlp.output_mask] = x[:,:,self.mlp.output_mask] + x_int

        x = torch.zeros_like(res)
        x[:, : , self.mlp.output_mask] = x[:, : , self.mlp.output_mask] + x_int
        x = res + x
    else:
        x = x_int + res
    feat_mlp = x
    #test_mlp = feat_mlp[0, 0].detach().cpu().numpy()

    return x, feat_attn, feat_mlp

def forward_block_nested_vit(self, x):
    #attention
    #if detach_graph:
    #    x = x.detach()
    #if not (hasattr(self.attn, "skip_block") and self.attn.skip_block):
    res = x
    x = self.norm1(x)
    x = self.attn(x)
    x = self.ls1(x)
    x = self.drop_path1(x)
    x = x + res
    feat_attn = x

    #FFN
    #if not (hasattr(self.mlp, "skip_block") and self.mlp.skip_block):
    res = x
    x = self.norm2(x)
    x = self.mlp(x)
    x = self.ls2(x)
    x = self.drop_path2(x)
    x = x + res
    feat_mlp = x

    return x, feat_attn, feat_mlp


def forward_attention_nested_vit(self, x):
    if hasattr(self, "input_mask") and self.input_mask is not None:
        x = x[:,:,self.input_mask]

    if hasattr(self, "attn_in_gate"):
        x = self.attn_in_gate(x)
    if hasattr(self, "attn_in_out_gate"):
        x = self.attn_in_out_gate(x)

    B, N, C = x.shape
    qkv = self.qkv(x)
    qkv = qkv.reshape(B, N, 3, self.num_heads, self.num_qkv_embed)

    if hasattr(self, "attn_qkv_gate"):
        qkv = self.attn_qkv_gate(qkv)

    qkv = qkv.permute(2, 0, 3, 1, 4)

    q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

    attn = (q @ k.transpose(-2, -1))
    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)
    x = (attn @ v).transpose(1, 2)

    x = x.reshape(B, N, int(self.num_heads * self.num_qkv_embed))

    x = self.proj(x)
    if hasattr(self, "attn_out_gate"):
        x = self.attn_out_gate(x)
    if hasattr(self, "attn_in_out_gate"):
        x = self.attn_in_out_gate(x)

    x = self.proj_drop(x)

    return x


def forward_attention(self, x):
    #if hasattr(self, "input_mask") and self.input_mask is not None:
    #    x = x[:,:,self.input_mask]

    if hasattr(self, "attn_in_gate"):
        x = self.attn_in_gate(x)
    if hasattr(self, "attn_in_out_gate"):
        x = self.attn_in_out_gate(x)

    B, N, C = x.shape
    qkv = self.qkv(x)
    qkv = qkv.reshape(B, N, 3, self.num_heads, self.num_qkv_embed)

    if hasattr(self, "attn_qkv_gate"):
        qkv = self.attn_qkv_gate(qkv)

    qkv = qkv.permute(2, 0, 3, 1, 4)

    q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

    attn = (q @ k.transpose(-2, -1)) * self.scale
    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)
    x = (attn @ v).transpose(1, 2)

    x = x.reshape(B, N, int(self.num_heads * self.num_qkv_embed))

    x = self.proj(x)
    if hasattr(self, "attn_out_gate"):
        x = self.attn_out_gate(x)
    if hasattr(self, "attn_in_out_gate"):
        x = self.attn_in_out_gate(x)
    x = self.proj_drop(x)

    return x

def forward_mlp(self, x):
    #if hasattr(self, "input_mask") and self.input_mask is not None:
    #    x = x[:, :, self.input_mask]

    if hasattr(self, "mlp_in_gate"):
        x = self.mlp_in_gate(x)
    if hasattr(self, "mlp_in_out_gate"):
        x = self.mlp_in_out_gate(x)

    x = self.fc1(x)
    x = self.act(x)
    if hasattr(self, "mlp_hid_gate"):
        x = self.mlp_hid_gate(x)

    x = self.drop(x)

    x = self.fc2(x)
    if hasattr(self, "mlp_out_gate"):
        x = self.mlp_out_gate(x)
    if hasattr(self, "mlp_in_out_gate"):
        x = self.mlp_in_out_gate(x)
    x = self.drop(x)
    return x


'''
update forward functions
'''

def get_forward_functions(model_name:str):

    new_forward_fkt = {
        "vit_base_patch16_224": {
                    "blocks.forward": {"blk": forward_block_vit, "attn": forward_attention, "mlp": forward_mlp},
                    "forward_features": forward_features_vit_base_patch16_224,
                    "forward": forward_vit_base_patch16_224,
                    },


        "deit_base_patch16_224": {
            "blocks.forward": {"blk": forward_block_vit, "attn": forward_attention, "mlp": forward_mlp},
            "forward_features": forward_features_vit_base_patch16_224,
            "forward": forward_vit_base_patch16_224,
        },
        "deit_small_patch16_224": {
            "blocks.forward": {"blk": forward_block_vit, "attn": forward_attention, "mlp": forward_mlp},
            "forward_features": forward_features_vit_base_patch16_224,
            "forward": forward_vit_base_patch16_224,
        },
        "deit_tiny_patch16_224": {
            "blocks.forward": {"blk": forward_block_vit, "attn": forward_attention, "mlp": forward_mlp},
            "forward_features": forward_features_vit_base_patch16_224,
            "forward": forward_vit_base_patch16_224,
        },


        "deit_base_distilled_patch16_224": {
                    "blocks.forward": {"blk": forward_block_vit, "attn": forward_attention, "mlp": forward_mlp},
                    "forward_features": forward_features_deit_distilled_patch16_224,
                    "forward": forward_deit_distilled_patch16_224,
                    },
        "deit_small_distilled_patch16_224": {
            "blocks.forward": {"blk": forward_block_vit, "attn": forward_attention, "mlp": forward_mlp},
            "forward_features": forward_features_deit_distilled_patch16_224,
            "forward": forward_deit_distilled_patch16_224,
        },
        "deit_tiny_distilled_patch16_224": {
            "blocks.forward": {"blk": forward_block_vit, "attn": forward_attention, "mlp": forward_mlp},
            "forward_features": forward_features_deit_distilled_patch16_224,
            "forward": forward_deit_distilled_patch16_224,
        },


        "dinov2_vitb14": {
            "blocks.forward": {"blk": forward_block_nested_vit, "attn": forward_attention_nested_vit, "mlp": forward_mlp},
            "forward_features": forward_features_dinov2_base_patch16_224,
            "forward": forward_dinov2_patch16_224,
        },
        "dinov2_vitb14_lc": {
            "blocks.forward": {"blk": forward_block_nested_vit, "attn": forward_attention_nested_vit,
                               "mlp": forward_mlp},
            "forward_features": forward_features_dinov2_base_patch16_224,
            "forward": forward_dinov2_patch16_224,
        },
    }

    return new_forward_fkt[model_name]




def update_model_gate_layer(model, args):
    '''
    def forward_hook(self, input, output):
        input_flat_sign = input[0].flatten(end_dim=1).sign()
        #input_flat_sign = input[0].sign()
        weight_sign = self.weight.sign().transpose(0,1)
        out_sign = torch.matmul(input_flat_sign, weight_sign)
        out_sign[out_sign <= 0] = 0
        out_sign = out_sign.mean(dim=0)
        y=1
    '''

    replace_linear(model, "model")

    # update parameter
    for name, m in model.named_modules():

        if m.__class__.__name__ in ["Attention", "MemEffAttention"]:
            m.kr = 1.0
            #m.skip_block = False
            m.input_mask = None
            m.output_mask = None

            parent = get_module(model, name.split(".")[:-1])
            m.related_block_param = [parent.norm1]

            #input
            embed_features = int(m.qkv.weight.shape[0] / 3)
            m.attn_in_gate = GateLayer(embed_features, [1, 1, -1], args)
            m.attn_in_gate.add_predec_layer(parent.norm1)
            m.attn_in_gate.add_success_layer(m.qkv)
            m.qkv.set_input_mask(m.attn_in_gate, "attnIn")
            parent.norm1.set_output_mask(m.attn_in_gate, "attnIn")

            #output
            m.attn_out_gate = GateLayer(m.proj.out_features, [1, 1, -1], args)
            m.attn_out_gate.add_predec_layer(m.proj)
            m.proj.set_output_mask(m.attn_out_gate, "attnOut")

            #qkv
            qkv_size = int(m.proj.out_features // m.num_heads)
            m.attn_qkv_gate = GateLayer(qkv_size, [1, 1, 1, 1, -1], args)
            m.attn_qkv_gate.add_predec_layer(m.qkv)
            m.attn_qkv_gate.add_success_layer(m.proj)
            m.proj.set_input_mask(m.attn_qkv_gate, "qkv", rep01_normal=m.num_heads)
            m.qkv.set_output_mask(m.attn_qkv_gate, "qkv", rep01_normal=m.num_heads*3)

        elif m.__class__.__name__ == "Mlp":
            m.kr = 1.0
            #m.skip_block = False
            m.input_mask = None
            m.output_mask = None

            parent = get_module(model, name.split(".")[:-1])
            m.related_block_param = [parent.norm2]

            #input
            embed_features = m.fc1.in_features
            m.mlp_in_gate = GateLayer(embed_features, [1, 1, -1], args)
            m.mlp_in_gate.add_predec_layer(parent.norm2)
            m.mlp_in_gate.add_success_layer(m.fc1)
            m.fc1.set_input_mask(m.mlp_in_gate, "mlpIn")
            parent.norm2.set_output_mask(m.mlp_in_gate, "mlpIn")

            #output
            embed_features = m.fc1.in_features
            m.mlp_out_gate = GateLayer(embed_features, [1, 1, -1], args)
            m.mlp_out_gate.add_predec_layer(m.fc2)
            m.fc2.set_output_mask(m.mlp_out_gate, "mlpOut")



            #hidden
            hidden_features = m.fc1.out_features
            m.mlp_hid_gate = GateLayer(hidden_features, [1, 1, -1], args)
            m.mlp_hid_gate.add_predec_layer(m.fc1)
            m.mlp_hid_gate.add_success_layer(m.fc2)
            m.fc1.set_output_mask(m.mlp_hid_gate, "mlpHid")
            m.fc2.set_input_mask(m.mlp_hid_gate, "mlpHid")


def update_model_forward_functions(model, args, name_model:str, add_gate_layer=False):

    for new_fkt_name, new_fkt in get_forward_functions(name_model).items():
        module = model
        for sub_module in new_fkt_name.split(".")[:-1]:
            module = getattr(module, sub_module)
        new_fkt_name = new_fkt_name.split(".")[-1]

        if isinstance(module, torch.nn.Sequential) or isinstance(module, torch.nn.ModuleList):
            for i in range(len(module)):
                new_forward_method = new_fkt["blk"].__get__(module[i], module[i].__class__)
                setattr(module[i], new_fkt_name, new_forward_method)

                #attention
                m_attn = module[i].attn
                new_forward_method = new_fkt["attn"].__get__(m_attn, m_attn.__class__)
                setattr(m_attn, new_fkt_name, new_forward_method)

                #mlp
                m_mlp = module[i].mlp
                new_forward_method = new_fkt["mlp"].__get__(m_mlp, m_mlp.__class__)
                setattr(m_mlp, new_fkt_name, new_forward_method)
        else:
            new_forward_method = new_fkt.__get__(module, module.__class__)
            setattr(module, new_fkt_name, new_forward_method)
    return model

