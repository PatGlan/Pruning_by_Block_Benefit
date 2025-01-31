
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numbers
import numpy as np
from copy import deepcopy


class MaskNorm(nn.LayerNorm):
    def __init__(self, m):
        super().__init__(m.weight.shape[0])

        self.weight = m.weight
        if m.bias is not None:
            self.bias = m.bias
        self.eps = m.eps


        #self._mask_input = []
        self._mask_output = []
        self.input_shape = None

        self.isFrozen = False

    def freeze(self):
        self.isFrozen = True
        #del(self._mask_input)

        del(self._mask_output)

    '''
    def set_input_mask(self, gate_mask, name_mask, rep01_normal=1, rep02_inter=1):
        mask = gate_mask.get_hard_mask()
        assert len(mask.shape) == 1
        assert mask.shape[0] * rep01_normal * rep02_inter == self.weight.shape[1]
        self._mask_input.append([gate_mask, rep01_normal, rep02_inter, name_mask])
        #self._mask_input_repeat = repeat
    '''


    def set_output_mask(self, gate_mask, name_mask, rep01_normal=1, rep02_inter=1):
        mask = gate_mask.get_hard_mask()
        assert len(mask.shape) == 1
        assert mask.shape[0] * rep01_normal * rep02_inter == self.weight.shape[0]
        self._mask_output.append([gate_mask, rep01_normal, rep02_inter, name_mask])


    def get_input_mask(self, count_remain=False):
        return torch.ones(1, device=self.weight.device)

    def get_output_mask(self,  count_remain=False):
        remain = {}
        mask_out = torch.ones(self.weight.shape[0], device=self.weight.device)
        if not self.isFrozen:
            for _gate_mask_out, rep01_normal, rep02_inter, name in self._mask_output:
                new_out_mask = _gate_mask_out.get_hard_mask()
                new_out_mask_orig_size = new_out_mask.repeat(rep01_normal).repeat_interleave(rep02_inter)
                mask_out = torch.logical_and(mask_out, new_out_mask_orig_size)
                if count_remain:
                    remain[name] = int(new_out_mask.sum().item())

        if count_remain:
            return mask_out, remain
        return mask_out

    def get_num_param(self, count_bias=True):

        if not self.weight.requires_grad:
            param_out_kept = 0.0
        #elif self._mask_output is not None:
        elif hasattr(self, "_mask_output"):
            param_out_kept = self.get_output_mask().sum().item()
        else:
            param_out_kept = self.weight.shape[0]

        num_param_kept = param_out_kept
        num_param_unpruned = self.weight.numel()

        if count_bias:
            num_param_kept += param_out_kept
            num_param_unpruned += self.bias.numel()
        return num_param_kept, num_param_unpruned


    def prune_param_by_reduce_size(self):

        mask_out, remain_out = self.get_output_mask(count_remain=True)
        mask_out = mask_out.type(torch.bool)

        w = self.weight[mask_out]
        b = self.bias[mask_out]
        self.out_features = mask_out.sum().item()

        normalized_shape = mask_out.sum().item()
        if isinstance(normalized_shape, numbers.Integral):
            # mypy error: incompatible types in assignment
            normalized_shape = (normalized_shape,)  # type: ignore[assignment]
        self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]

        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(b)

        return remain_out, remain_out




class MaskLinear(nn.Linear):
    def __init__(self, m):

        super().__init__(m.in_features, m.out_features, m.bias is not None)
        self.weight = m.weight
        if m.bias is not None:
            self.bias = m.bias

        self._mask_input = []
        self._mask_output = []
        self.input_shape = None

        self.apply_mask_in_forward = False #set for sparse training
        self.isFrozen = False
    def freeze(self):
        self.isFrozen = True
        del(self._mask_input)
        del(self._mask_output)

    def set_input_mask(self, gate_mask, name_mask, rep01_normal=1, rep02_inter=1):
        mask = gate_mask.get_hard_mask()
        assert len(mask.shape) == 1
        assert mask.shape[0] * rep01_normal * rep02_inter == self.weight.shape[1]
        self._mask_input.append([gate_mask, rep01_normal, rep02_inter, name_mask])
        #self._mask_input_repeat = repeat

    def set_output_mask(self, gate_mask, name_mask, rep01_normal=1, rep02_inter=1):
        mask = gate_mask.get_hard_mask()
        assert len(mask.shape) == 1
        assert mask.shape[0] * rep01_normal * rep02_inter == self.weight.shape[0]
        self._mask_output.append([gate_mask, rep01_normal, rep02_inter, name_mask])
        #self._mask_output_repeat = repeat



    def get_input_mask(self, count_remain=False):
        remain = {}
        #mask_in = torch.ones_like(self.weight[0, :])
        mask_in = torch.ones(self.weight.shape[1], device=self.weight.device)
        if not self.isFrozen:
            for _gate_mask_in, rep01_normal, rep02_inter, name in self._mask_input:
                new_in_mask = _gate_mask_in.get_hard_mask()
                new_in_mask_orig_size = new_in_mask.repeat(rep01_normal).repeat_interleave(rep02_inter)
                mask_in = torch.logical_and(mask_in, new_in_mask_orig_size)
                if count_remain:
                    remain[name] = int(new_in_mask.sum().item())

        if count_remain:
            return mask_in, remain
        return mask_in
    def get_output_mask(self,  count_remain=False):
        remain = {}
        #mask_out = torch.ones_like(self.weight[:, 0])
        mask_out = torch.ones(self.weight.shape[0], device=self.weight.device)
        if not self.isFrozen:
            for _gate_mask_out, rep01_normal, rep02_inter, name in self._mask_output:
                new_out_mask = _gate_mask_out.get_hard_mask()
                new_out_mask_orig_size = new_out_mask.repeat(rep01_normal).repeat_interleave(rep02_inter)
                mask_out = torch.logical_and(mask_out, new_out_mask_orig_size)
                if count_remain:
                    remain[name] = int(new_out_mask.sum().item())

        if count_remain:
            return mask_out, remain
        return mask_out

    def get_num_param(self, count_bias=True):

        if not self.weight.requires_grad: #layer is disablied
            param_in_kept = 0.0
        elif hasattr(self, "_mask_input"):
            in_mask = self.get_input_mask().detach().cpu().numpy()
            param_in_kept = self.get_input_mask().sum().item()
        else:
            param_in_kept = self.weight.shape[1]

        if not self.weight.requires_grad:
            param_out_kept = 0.0
        #elif self._mask_output is not None:
        elif hasattr(self, "_mask_output"):
            out_mask = self.get_output_mask().detach().cpu().numpy()
            param_out_kept = self.get_output_mask().sum().item()
        else:
            param_out_kept = self.weight.shape[0]

        num_param_kept = param_in_kept * param_out_kept
        num_param_unpruned = self.weight.numel()

        if count_bias:
            num_param_kept += param_out_kept
            num_param_unpruned += self.bias.numel()
        return num_param_kept, num_param_unpruned


    def prune_param_by_reduce_size(self):

        mask_in, remain_in = self.get_input_mask(count_remain=True)
        #assert len([x for x in remain_in.values() if x <=0]) == 0, "zero-dimension must not exist"
        mask_in = mask_in.type(torch.bool)
        w = self.weight[:, mask_in]
        self.in_features = mask_in.sum().item()

        mask_out, remain_out = self.get_output_mask(count_remain=True)
        #assert len([x for x in remain_out.values() if x <= 0]) == 0, "zero-dimension must not exist"
        mask_out = mask_out.type(torch.bool)


        '''
        #todo test
        rand_mat01 = torch.rand(128, 17, self.weight.shape[1], requires_grad=False).cuda()
        rand_mat02 = deepcopy(rand_mat01)

        rand_mat01[:, :, torch.logical_not(mask_in)] = 0.0
        test_out01 = F.linear(rand_mat01, self.weight) #, self.bias)
        #test_out01[:, :, torch.logical_not(mask_out)] = 0.0
        #test_out01 = test_out01[:, :, mask_out]

        w_new = self.weight
        #w_new[:, torch.logical_not(mask_in)] = 0.0
        #w_new[torch.logical_not(mask_out), :] = 0.0
        w_new = w_new[:, mask_in]
        #w_new = w_new[mask_out, :]

        b_new = self.bias
        b_new.requires_grad = False
        b_new = b_new[mask_out]
        #b_new[torch.logical_not(mask_out)] = 0.0
        test_out02 = F.linear(rand_mat01[:, :, mask_in], w_new) #, b_new)


        test_12 = (test_out01-test_out02).abs().detach().cpu().numpy()
        test_12_ = (test_out01-test_out02).abs().sum().item()
        max_diff = np.max(test_12)

        if mask_in.sum().item() != self.weight.shape[1]:
            y=1
        '''

        w = w[mask_out, :]
        b = self.bias[mask_out]
        self.out_features = mask_out.sum().item()

        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(b)

        return remain_in, remain_out

    def forward(self, input: Tensor) -> Tensor:
        self.input_shape = input.shape

        w_mask = self.weight
        b_mask = self.bias


        if self.apply_mask_in_forward:

            mask_in = self.get_input_mask()
            '''
            w_mask.mul_(mask_in)
            '''
            w_mask = w_mask[:, mask_in]
            input = input[:, :, mask_in]


            mask_out = self.get_output_mask()
            w_mask = w_mask[mask_out, :]
            b_mask = b_mask[mask_out]

        return F.linear(input, w_mask, b_mask)



def replace_linear(module, name):
    '''
    Recursively put desired batch norm in nn.module module.

    set module = net to start code.
    '''
    from .mask_wrapper import MaskLinear, MaskNorm

    # go through all attributes of module nn.module (e.g. network or layer) and put batch norms if present
    for attr_str in dir(module):
        target_attr = getattr(module, attr_str)
        if type(target_attr) == nn.Linear and "blocks" in name:
            new_module = MaskLinear(target_attr)
            setattr(module, attr_str, new_module)
        elif type(target_attr) == nn.LayerNorm and "blocks" in name:
            new_module = MaskNorm(target_attr)
            setattr(module, attr_str, new_module)
    # iterate through immediate child modules. Note, the recursion is done by our code no need to use named_modules()
    for name_child, immediate_child_module in module.named_children():
        replace_linear(immediate_child_module, name + "." + name_child)