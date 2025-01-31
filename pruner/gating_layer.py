# Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
# Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
Gating layer for pruning, adapted from [Taylor_pruning](https://github.com/NVlabs/Taylor_pruning)
"""

import torch
from torch.nn.modules import Module
import torch.nn as nn
from functools import reduce


def forward_hook_out(self, input, output):
    self.output = output.detach()

def backward_hook_out(self, grad_input, grad_output):
    self.output_grad = grad_output[0].detach()

def backward_hook_input(self, grad_input, grad_output):
    self.input_grad = grad_input[0].detach()


class MaskModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.m = nn.Parameter(torch.ones(dim))
        self.no_update = True



class GateLayer(nn.Module):
    def __init__(self, features, size_mask, args):
        super(GateLayer, self).__init__()

        self.features = features
        self.size_mask_view = size_mask
        self.input_dims = [i for i, x in enumerate(self.size_mask_view) if x > 0]

        self.isFrozen = False
        self.mask = MaskModule(self.features)

        self.predec_layer = None
        self.successor_layer = None


    def set_extern_mask(self, new_mask):
        self.mask = new_mask

    def add_predec_layer(self, predec_layer):
        self.predec_layer = predec_layer

    def add_success_layer(self, successor_layer):
        self.successor_layer = successor_layer

    def get_param_per_mask_elem(self):
        param_per_mask_elem = 0
        num_mask_elem = self.mask.m.sum()

        if self.predec_layer is not None:
            predec_input_mask = self.predec_layer.get_input_mask().type(torch.int)
            rep = self.predec_layer.weight.shape[0] // self.features
            param_per_mask_elem += predec_input_mask.sum() * rep


        if self.successor_layer is not None:
            successor_out_mask = self.successor_layer.get_output_mask().type(torch.int)
            rep = self.successor_layer.weight.shape[1] // self.features
            param_per_mask_elem += successor_out_mask.sum() * rep

        return param_per_mask_elem, num_mask_elem



    def set_new_mask(self, new_mask):
        if not self.isFrozen:
            assert all([n == o for n, o in zip(new_mask.shape, self.mask.m.shape)])
            assert new_mask.dtype == torch.float
            self.mask.m.data = new_mask.data

    def get_hard_mask(self):
        if hasattr(self, "mask"):
            return (self.mask.m.data >= 0.5)
        return None
    #def reset_weight(self):
    #    if not self.isFrozen:
    #        self.weight.data = torch.ones_like(self.weight.data)

    def forward(self, input):
        if self.isFrozen:
            return input

        #self.input_features = input.sum(dim=self.input_dims)

        if not hasattr(self, "x_elem"):
            shape_dim = [input.shape[i] for i, m in enumerate(self.size_mask_view) if m > 0]
            shape_dim = shape_dim[1:] #ignore batch-dimension
            self.x_elem = reduce(lambda x, y: x*y, shape_dim)

        #input.mul_(self.weight.view(*self.size_mask_view))

        if self.mask is not None:
            input = input * self.mask.m.view(*self.size_mask_view)
            #input.mul_(self.mask.view(*self.size_mask_view))

        #self.output_features = input.sum(dim=self.input_dims)

        return input

    #def extra_repr(self):
    #    return 'in_features={}, out_features={}'.format(
    #        self.input_features, self.output_features is not None
    #    )

    def freeze(self):
        self.isFrozen = True
        #del(self.weight)
        del(self.mask)