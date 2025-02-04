import torch


def get_module(obj, names):
    if len(names) == 0:
        #setattr(obj, names[0])
        return obj
    else:
        return get_module(getattr(obj, names[0]), names[1:])

def get_new_attn_dim(model_without_ddp):
    '''
    num_heads = [0] * len(model_without_ddp.blocks)

    found_attn = 0
    for name, m in model_without_ddp.named_modules():
        #print(name)
        if "attn" == name.split(".")[-1]:
            block_idx = int(name.split(".")[1])
            num_heads[block_idx] = m.num_heads
            found_attn += 1

    assert len(list_num_heads) == found_attn
    '''
    list_num_heads = []
    list_qkv_embed = []

    model_blocks = [m for name, m in model_without_ddp.named_modules() if name.endswith("blocks") or name.endswith("layer")][0]
    for block in model_blocks:
        list_num_heads.append(block.attn.num_heads)
        #list_qkv_embed.append(block.attn.qkv.out_features // 3 // block.attn.num_heads)
        list_qkv_embed.append(block.attn.num_qkv_embed)

    return list_num_heads, list_qkv_embed


def get_block_masks(model_without_ddp):

    list_block_masks = {}

    model_blocks = [m for name, m in model_without_ddp.named_modules() if name.endswith("blocks") or name.endswith("layer")][0]
    for i, block in enumerate(model_blocks):
        if hasattr(block.attn, "attn_in_gate"):
            list_block_masks[f"{i}_attn_input"] = block.attn.attn_in_gate.get_hard_mask()
        elif hasattr(block.attn, "input_mask"):
            list_block_masks[f"{i}_attn_input"] = block.attn.input_mask

        if hasattr(block.attn, "attn_out_gate"):
            list_block_masks[f"{i}_attn_output"] = block.attn.attn_out_gate.get_hard_mask()
        elif hasattr(block.attn, "output_mask"):
            list_block_masks[f"{i}_attn_output"] = block.attn.output_mask

        if hasattr(block.mlp, "mlp_in_gate"):
            list_block_masks[f"{i}_mlp_input"] = block.mlp.mlp_in_gate.get_hard_mask()
        elif hasattr(block.mlp, "input_mask"):
            list_block_masks[f"{i}_mlp_input"] = block.mlp.input_mask

        if hasattr(block.mlp, "mlp_out_gate"):
            list_block_masks[f"{i}_mlp_output"] = block.mlp.mlp_out_gate.get_hard_mask()
        elif hasattr(block.mlp, "output_mask"):
            list_block_masks[f"{i}_mlp_output"] = block.mlp.output_mask

        y=1

    return list_block_masks
