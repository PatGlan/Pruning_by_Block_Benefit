


from copy import deepcopy
from torch import nn

from .gating_layer import GateLayer
from .mask_wrapper import MaskLinear, MaskNorm


def get_module(obj, names):
    if len(names) == 0:
        return obj
    else:
        return get_module(getattr(obj, names[0]), names[1:])

def _update_parent_properties(model, name_module, remain_in, remain_out):
    parent_name = name_module.split(".")[:-1]
    parent_class = get_module(model, parent_name)

    new_num_heads = None
    if "head" in remain_in.keys():
        new_num_heads = remain_in["head"]
    if "head" in remain_out.keys():
        new_num_heads = remain_out["head"]
    if new_num_heads is not None:
        parent_class.num_heads = new_num_heads

    new_num_qkv_embed = None
    if "qkv" in remain_in.keys():
        new_num_qkv_embed = remain_in["qkv"]
    if "qkv" in remain_out.keys():
        new_num_qkv_embed = remain_out["qkv"]
    if new_num_qkv_embed is not None:
        parent_class.num_qkv_embed = new_num_qkv_embed

    y=1

def freeze_model(model):

    for name, m in model.named_modules():
        if isinstance(m, MaskLinear) and 'blocks' in name:

            remain_in, remain_out = m.prune_param_by_reduce_size()
            _update_parent_properties(model, name, remain_in, remain_out)
            m.freeze()
        if isinstance(m, MaskNorm) and 'blocks' in name:
            remain_in, remain_out = m.prune_param_by_reduce_size()
            _update_parent_properties(model, name, remain_in, remain_out)
            m.freeze()

    list_gate_paths = []
    for name, m in model.named_modules():
        if isinstance(m, GateLayer):
            parent_name = name.split(".")[:-1]
            parent_class = get_module(model, parent_name)
            gate_hard_mask = m.get_hard_mask()
            if not hasattr(parent_class, "skip_block") or not parent_class.skip_block:
                parent_class.skip_block = not gate_hard_mask.any().item()

                if parent_class.skip_block:
                    parent_class.kr = 0.0
                    for m_in_block in parent_class.parameters():
                        m_in_block.requires_grad = False
                    y=1
                    for m_related_block in parent_class.related_block_param:
                        for m_r_b in m_related_block.parameters():
                            m_r_b.requires_grad = False

                    y=1
            if "in_gate" in name:
                parent_class.input_mask = deepcopy(gate_hard_mask)
            #if "proj" in name or "mlp_out_gate" in name:
            if "out_gate" in name:
                parent_class.output_mask = deepcopy(gate_hard_mask)
            if "in_out_gate" in name:
                parent_class.input_mask = deepcopy(gate_hard_mask)
                parent_class.output_mask = deepcopy(gate_hard_mask)
            #m.freeze()
            list_gate_paths.append(name)

    #delete gate-layer
    for gate_path in list_gate_paths:
        parent_class = get_module(model, gate_path.split(".")[:-1])
        delattr(parent_class, gate_path.split(".")[-1])

        y=1

    if hasattr(model, "fmHead"):
        del(model.fmHead)

    y=1


def count_non_block_param(model):
    non_block_param = 0
    non_block_param += model.pos_embed.numel()
    non_block_param += model.cls_token.numel()
    non_block_param += model.patch_embed.proj.weight.numel() + model.patch_embed.proj.bias.numel()
    non_block_param += model.norm.weight.numel() + model.norm.bias.numel()
    non_block_param += model.head.weight.numel() + model.head.bias.numel()
    return non_block_param

def count_parameters_model_hard_masked(model, args):
    param_kept = 0
    param_unpruned = 0

    for name, m in model.named_modules():
        if isinstance(m, MaskLinear) and 'blocks' in name:
            m_param_kept, m_param_unpr = m.get_num_param()
            param_kept = param_kept + m_param_kept
            param_unpruned = param_unpruned + m_param_unpr
        elif "norm" in name and 'blocks' in name:
            num_param = m.weight.numel() + m.bias.numel()
            param_kept = param_kept + num_param
            param_unpruned = param_unpruned + num_param

    if not args.prune_rate_by_block_param_only:
        non_block_param = count_non_block_param(model)
        param_kept = param_kept + non_block_param
        param_unpruned = param_unpruned + non_block_param

    return param_kept, param_unpruned

def count_kept_param_per_block(model):
    #model_tmp = model.module if isinstance(model, DistributedDataParallel) else model
    block_param_kept = [0] * (len(get_module(model, "blocks".split("."))) * 2)
    block_param_unpr = [0] * (len(get_module(model, "blocks".split("."))) * 2)
    subidx_table = {"attn": 0, "mlp": 1}
    for name, m in model.named_modules():
        if isinstance(m, MaskLinear) and 'blocks' in name:
            m_kept_param, param_unpr = m.get_num_param()

            block_idx = int(name.split(".")[1])
            sub_idx = subidx_table[name.split(".")[2]]

            block_param_kept[(2*block_idx) + sub_idx] += m_kept_param
            block_param_unpr[(2*block_idx) + sub_idx] += param_unpr
        elif isinstance(m, nn.Linear) and 'blocks' in name:
            block_idx = int(name.split(".")[1])
            sub_idx = subidx_table[name.split(".")[2]]

            n_param = m.weight.numel() + m.bias.numel()
            block_param_kept[(2*block_idx) + sub_idx] += n_param
            block_param_unpr[(2*block_idx) + sub_idx] += n_param
        elif "norm" in name and 'blocks' in name:
            num_param = m.weight.numel() + m.bias.numel()

            block_idx = int(name.split(".")[1])
            if "norm1" in name:
                sub_idx = subidx_table["attn"]
            elif "norm2" in name:
                sub_idx = subidx_table["mlp"]
            else:
                assert False

            block_param_kept[(2 * block_idx) + sub_idx] += num_param
            block_param_unpr[(2 * block_idx) + sub_idx] += num_param


    return block_param_kept, block_param_unpr

