

import torch
import math
from torch.nn import functional as F

from pruner.mask_wrapper import MaskLinear
from pruner import GateLayer, count_kept_param_per_block
from pruner.pruner_utils import count_non_block_param
from pruner.prune_scores import *


import torch.distributed as dist
from collections import OrderedDict
from copy import copy


class PruneLoss(torch.nn.Module):

    def __init__(self, base_criterion: torch.nn.Module, model, steps_per_epoch, args, logger=None):
        super().__init__()
        self.base_criterion = base_criterion
        self.logger = logger
        self.model = model

        #prune parameter
        self.f_sp = torch.nn.Softplus()
        self.s_block = {k + "_gate":1.0 for k in ["attn_in", "attn_qkv", "attn_out", "mlp_in", "mlp_hid", "mlp_out"]}
        for scale_str in args.prune_scale_gate_imp.split(","):
            k = scale_str.split(":")[0] + "_gate"
            if k in self.s_block.keys():
                self.s_block[k] = float(scale_str.split(":")[1])


        self.step_start = int(args.prune_start_epoch * steps_per_epoch)
        self.step_sparse = int(args.prune_sparse_epoch * steps_per_epoch)
        if args.prune_finalize_epoch is not None:
            self.step_final = int(args.prune_finalize_epoch * steps_per_epoch)
        else:
            self.step_final = int(args.epochs * steps_per_epoch)
        assert self.step_sparse >= self.step_start, "there must be training steps to prune model"
        assert self.step_final >= self.step_sparse, "there must be training steps to prune model"

        self.update_block_kr = True
        if args.prune_kr_handcraft is not None:
            kr_block = args.prune_kr_handcraft.split(",")
            assert len(kr_block) == 24, "arg.prune-kr-block is defined wrong"
            kr_block = [float(x) for x in kr_block]
            self.kr_block = torch.tensor(kr_block).cuda()
            self.update_block_kr = False

        self.update_intra_block_scores = not args.model_freeze

        self.block_obj = {}
        self.block_gates = {}
        for block_idx in range(len(model.blocks)):
            #add attention
            self.block_obj[f"blocks.{block_idx}.attn"] = model.blocks[block_idx].attn

            attn_gates = {}
            for sub_name, sub_m in model.blocks[block_idx].attn.named_modules():
                if isinstance(sub_m, GateLayer):
                    attn_gates[sub_name] = sub_m
            self.block_gates[f"blocks.{block_idx}.attn"] = attn_gates

            #add mlp
            self.block_obj[f"blocks.{block_idx}.mlp"] = model.blocks[block_idx].mlp

            mlp_gates = {}
            for sub_name, sub_m in model.blocks[block_idx].mlp.named_modules():
                if isinstance(sub_m, GateLayer):
                    mlp_gates[sub_name] = sub_m
            self.block_gates[f"blocks.{block_idx}.mlp"] = mlp_gates

        self.mask_blocks_kept_dense = torch.zeros(len(self.block_obj.keys()), dtype=torch.bool, requires_grad=False).cuda()
        self.mask_block_disable = torch.zeros(len(self.block_obj.keys()), dtype=torch.bool, requires_grad=False).cuda()

        self.num_non_block_param = count_non_block_param(model)
        if args.prune_rate_by_block_param_only:
            self.num_non_block_param = 0

        self.reset_imp_scores()
        self.args = args

    def reset_imp_scores(self):
        if hasattr(self, "imp_scores"):
            self.last_imp_values = {}
            self.last_imp_values["imp_scores"] = self.imp_scores
            self.last_imp_values["imp_cnt"] = self.imp_cnt
            if self.update_block_kr:
                self.last_imp_values["imp_block_cnt"] = self.imp_block_cnt
                self.last_imp_values["imp_block_cls"] = self.imp_block_cls.detach()
                self.last_imp_values["imp_block_patch"] = self.imp_block_patch.detach()

        self.imp_scores = OrderedDict()
        self.imp_cnt = OrderedDict()
        if self.update_block_kr:
            self.imp_block_cnt = torch.zeros(1, requires_grad=False).cuda()
            self.imp_block_cls = torch.zeros(len(self.block_obj.keys()), requires_grad=False).cuda()
            self.imp_block_patch = torch.zeros(len(self.block_obj.keys()), requires_grad=False).cuda()

        #update w_per_layer
        self.remain_w_per_block, self.w_per_block = count_kept_param_per_block(self.model)
        self.remain_w_per_block = torch.tensor(self.remain_w_per_block, dtype=torch.float, requires_grad=False).cuda()
        self.w_per_block = torch.tensor(self.w_per_block, dtype=torch.float, requires_grad=False).cuda()




    '''
    parameter: s_p,  targets,  step, t_p=None, s_x_inter=None, t_x_inter=None
    '''
    def forward(self, *args, **kwargs):

        #base criterion
        if self.args.distillation_type == 'none':
            loss = self.base_criterion(kwargs["s_p"], kwargs["targets"])
        elif self.args.distill_method == 'deit':
            loss = self.base_criterion(kwargs["inputs"], kwargs["s_p"], kwargs["targets"])
        else:
            assert False, "distill method does not exist"

        #loss soft classifier from intermediate results
        if self.update_block_kr and kwargs["s_p_inter"] is not None:
            cls_losses_in = []
            cls_losses_out = []
            patch_losses_in = []
            patch_losses_out = []
            for p_cls_in, p_cls_out, p_patch_in, p_patch_out in \
                    zip(kwargs["s_p_inter"]["p_cls_in"], kwargs["s_p_inter"]["p_cls_out"],
                        kwargs["s_p_inter"]["p_patch_in"], kwargs["s_p_inter"]["p_patch_out"]):
                cls_losses_in.append(self.base_criterion(p_cls_in, kwargs["targets"]).unsqueeze(0))
                cls_losses_out.append(self.base_criterion(p_cls_out, kwargs["targets"]).unsqueeze(0))
                patch_losses_in.append(self.base_criterion(p_patch_in, kwargs["targets"]).unsqueeze(0))
                patch_losses_out.append(self.base_criterion(p_patch_out, kwargs["targets"]).unsqueeze(0))

            cls_losses_in = torch.cat(cls_losses_in)
            cls_losses_out = torch.cat(cls_losses_out)
            patch_losses_in = torch.cat(patch_losses_in)
            patch_losses_out = torch.cat(patch_losses_out)

            loss += cls_losses_in.sum()
            loss += cls_losses_out.sum()
            loss += patch_losses_in.sum()
            loss += patch_losses_out.sum()

            if (kwargs["step"] >= self.step_start):
                new_imp_block_cls = cls_losses_in.detach() - cls_losses_out.detach()
                new_imp_block_patch = patch_losses_in.detach() - patch_losses_out.detach()

                self.imp_block_cls += new_imp_block_cls
                self.imp_block_patch += new_imp_block_patch
                self.imp_block_cnt += 1



        return loss

    def get_kr_block_without_update(self) -> list:
        kr_blocks = []
        for i, (name_block, block) in enumerate(self.block_obj.items()):
            kr_blocks.append(block.kr)
        return kr_blocks

    def get_skip_block_flag(self) -> list:
        skip_block_flags = []
        for i, (name_block, block) in enumerate(self.block_obj.items()):
            skip_block_flags.append(block.skip_block)
        return skip_block_flags

    def transform_imp_block(self, imp_block, cnt_block, step=0, name="xxx"):
        imp_block = imp_block / cnt_block

        if self.logger is not None:
            self.logger.info(f"step: {step} | imp_block_{name}Bt: " + ", ".join(
                ["{:+02.4f}".format(x) for x in imp_block.detach().cpu().tolist()]).replace("+", " "))

        imp_block = imp_block * self.args.pr_block_imp_scale / imp_block.max()
        imp_block = self.f_sp(1.4 * imp_block) - self.f_sp(1.4 * imp_block - self.args.pr_block_imp_scale)

        if self.logger is not None:
            self.logger.info(f"step: {step} | imp_block_{name}At: " + ", ".join(
                ["{:+02.4f}".format(x) for x in imp_block.detach().cpu().tolist()]).replace("+", " "))

        imp_block = imp_block / (self.remain_w_per_block + 1e-6)
        imp_block = imp_block / imp_block.mean()

        if self.logger is not None:
            self.logger.info(f"step: {step} | imp_block_{name}As: " + ", ".join(
                ["{:+02.4f}".format(x) for x in imp_block.detach().cpu().tolist()]).replace("+", " "))
            self.logger.info(f"step: {step} | " + ("*" * 231))

        return imp_block


    def get_imp_block(self, step=None, reset_imp_score=True):
        if self.logger is not None:
            self.logger.info(f"step: {step} | layer_index_imp: " +
                             ", ".join([" {:1.4f}".format(x) for x in range(1, 10)]) + ", " +
                             ", ".join([" {:2.3f}".format(x) for x in range(10, 25)]))
            self.logger.info(f"step: {step} | " + ("*" * 231))

        imp_block_cls = self.transform_imp_block(self.imp_block_cls, self.imp_block_cnt, step, "cls")
        imp_block_patch = self.transform_imp_block(self.imp_block_patch, self.imp_block_cnt, step, "pat")

        #smooth update of imp_blocks
        if hasattr(self, "last_imp_block_cls") and self.update_block_kr:
            gamma = self.args.pr_gamma
            imp_block_cls = gamma * imp_block_cls + (1 - gamma) * self.last_imp_block_cls
            imp_block_patch = gamma * imp_block_patch + (1 - gamma) * self.last_imp_block_patch

        if reset_imp_score:
            self.last_imp_block_cls = imp_block_cls
            self.last_imp_block_patch = imp_block_patch

        imp_block = self.args.pr_alpha * imp_block_cls + (1 - self.args.pr_alpha) * imp_block_patch

        return imp_block



    def get_kr_blockwise(self, step=None, reset_imp_score=True):

        min_kr_block = 0.0
        max_kr_block = 1.0

        kr_model = 1 - (self.args.prune_ratio)
        if self.update_block_kr:
            kr_block = self.get_imp_block(step, reset_imp_score)

            if self.logger is not None:
                self.logger.info(
                    f"step: {step} | imp_block_final: " + ", ".join(
                        ["{:+02.4f}".format(x) for x in kr_block.detach().cpu().tolist()]).replace(
                        "+", " "))

            #fist scaling run
            mask_blocks_kept_dense = self.mask_blocks_kept_dense
            mask_blocks_disabled = torch.logical_or(self.mask_block_disable, (kr_block <= min_kr_block))
            mask_block_prunable = torch.logical_not(torch.logical_or(mask_blocks_disabled, mask_blocks_kept_dense))
            kr_block[mask_blocks_disabled] = min_kr_block
            kr_block[mask_blocks_kept_dense] = max_kr_block

            num_prunable_param_model = kr_model * (self.w_per_block.sum() + self.num_non_block_param) \
                                       - self.num_non_block_param \
                                       - (self.w_per_block[mask_blocks_disabled] * min_kr_block).sum() \
                                       - (self.w_per_block[mask_blocks_kept_dense] * max_kr_block).sum()
            s = num_prunable_param_model / (kr_block[mask_block_prunable] * self.w_per_block[mask_block_prunable]).sum()
            kr_block[mask_block_prunable] *= s

            cnt = 0
            while ((kr_block > max_kr_block).any() or (kr_block < min_kr_block).any()):
                assert cnt < 20, "to much for kr"

                mask_blocks_kept_dense = torch.logical_or(mask_blocks_kept_dense, (kr_block >= max_kr_block))
                mask_blocks_disabled = torch.logical_or(mask_blocks_disabled, (kr_block <= min_kr_block))
                mask_block_prunable = torch.logical_not(torch.logical_or(mask_blocks_disabled, mask_blocks_kept_dense))
                kr_block[mask_blocks_disabled] = min_kr_block
                kr_block[mask_blocks_kept_dense] = max_kr_block
                if not mask_block_prunable.any() and self.logger is not None:
                    self.logger.info("All Blocks fully pruned!")
                    self.logger.info(kr_block.detach().cpu().numpy())
                    break

                num_prunable_param_model = kr_model * (self.w_per_block.sum() + self.num_non_block_param) \
                                           - self.num_non_block_param \
                                           - (self.w_per_block[mask_blocks_disabled] * min_kr_block).sum() \
                                           - (self.w_per_block[mask_blocks_kept_dense] * max_kr_block).sum()
                s = num_prunable_param_model / (
                            kr_block[mask_block_prunable] * self.w_per_block[mask_block_prunable]).sum()
                kr_block[mask_block_prunable] *= s
                cnt += 1

            kr_block = torch.clamp(kr_block, min=min_kr_block, max=max_kr_block)
        else:
            kr_block = self.kr_block

        return kr_block

    def update_kr_block(self, step=None) -> list:

        pr_step = 1.0
        if step is not None:
            pr_step = min(max(step + 1 - self.step_start, 0) / max(float(self.step_sparse - self.step_start), 1e-4), 1.0)

        goal_kr_block = self.get_kr_blockwise(step=step)
        actual_kr_block = []
        for i, (name_block, block) in enumerate(self.block_obj.items()):
            block.kr = 1 - (1 - goal_kr_block[i].item()) * pr_step
            actual_kr_block.append(block.kr)

        return goal_kr_block

    def _get_new_mask(self, vec:torch.tensor, sort_idx:torch.tensor, t_idx:int, step=int):
        if self.step_final > self.step_sparse:
            sharp_ratio = max(min((step - self.step_sparse) / max(float(self.step_final - self.step_sparse), 1e-4), 1.0), 0.0)
        else:
            sharp_ratio = 1.0
        offset_imp_val = sharp_ratio * self.args.pr_tau_min + (1-sharp_ratio) * self.args.pr_tau_max
        offset_mask_val = 0.9

        # calculation by idx
        new_mask = torch.arange(1, sort_idx.shape[0] + 1).to(sort_idx.device)
        offset_idx = vec.shape[0] * offset_imp_val
        new_mask = torch.sigmoid(- math.log(1 / offset_mask_val - 1) * (new_mask - t_idx) /
                                 (offset_idx + 1e-8))
        new_mask = new_mask[torch.argsort(sort_idx)]

        return new_mask


    def _get_num2prune(self, block, kr):
        param_block_unpr = 0
        num_param_is = 0
        for name_block_sub_m, block_sub_m in block.named_modules():
            if isinstance(block_sub_m, MaskLinear):
                param_sub_kept, param_sub_unpr = block_sub_m.get_num_param()
                param_block_unpr += param_sub_unpr
                num_param_is += param_sub_kept
        num_param_keep_goal = param_block_unpr * kr
        num_param2prune = num_param_is - num_param_keep_goal
        return num_param2prune


    def prune_forward(self, step=None, logger=None, finalize_pruning=False):

        if finalize_pruning:
            step = self.step_sparse
            if self.update_block_kr:
                self.imp_block_cls = self.imp_block_cls + self.last_imp_values["imp_block_cls"]
                self.imp_block_patch = self.imp_block_patch + self.last_imp_values["imp_block_patch"]
                self.imp_block_cnt = self.imp_block_cnt + self.last_imp_values["imp_block_cnt"]

            if self.update_intra_block_scores:
                for k in self.imp_scores.keys():
                    self.imp_scores[k] = self.imp_scores[k] + self.last_imp_values["imp_scores"][k]
                    self.imp_cnt[k] = self.imp_cnt[k] + self.last_imp_values["imp_cnt"][k]
            del(self.last_imp_values)

        if self.args.distributed:
            dist.barrier()
            if self.update_block_kr:
                dist.all_reduce(self.imp_block_cls, op=torch.distributed.ReduceOp.SUM)
                dist.all_reduce(self.imp_block_patch, op=torch.distributed.ReduceOp.SUM)
                dist.all_reduce(self.imp_block_cnt, op=torch.distributed.ReduceOp.SUM)

            if self.update_intra_block_scores:
                cat_score = torch.cat(list(self.imp_scores.values()))
                cat_cnt = torch.cat(list(self.imp_cnt.values()))
                dist.all_reduce(cat_score, op=torch.distributed.ReduceOp.SUM)
                dist.all_reduce(cat_cnt, op=torch.distributed.ReduceOp.SUM)

                CNT = 0
                for i, name in enumerate(self.imp_scores.keys()):
                    size = self.imp_scores[name].shape[0]
                    self.imp_scores[name] = cat_score[CNT:(CNT + size)]
                    self.imp_cnt[name] = cat_cnt[i:i+1]

                    CNT += size
                del(cat_score)
                del(cat_cnt)

        if not self.update_intra_block_scores:
            kr_block = self.get_kr_blockwise(step=step)
            self.reset_imp_scores()
            return kr_block

        kr_block = self.update_kr_block(step)


        if logger is not None:
            logger.info(f"********* model keep_ratio per gate *********")

        #analyse imp-scores
        for block_idx, (name_block, block) in enumerate(self.block_obj.items()):

            # get imp_valued block
            imp = {}
            for i, (name_gate, gate) in enumerate(self.block_gates[name_block].items()):
                name = name_block + "." + name_gate
                new_imp = self.imp_scores[name] / self.imp_cnt[name]
                imp[name_gate] = self.s_block[name_gate] * (torch.argsort(torch.argsort(new_imp)).float() + 1) / new_imp.numel()

            imp = torch.cat(list(imp.values()))
            imp = imp + torch.linspace(0, 1e-20, steps=imp.shape[0], device=imp.device) # avoid equal values
            sort_imp, sort_idx = torch.sort(imp)
            placeholder_elem = torch.zeros(1).to(imp.device)

            #assign new masks
            num_pruning_repititions = 0
            assert block.kr >= 0.0 and block.kr <= 1.0
            param2prune = self._get_num2prune(block, kr=block.kr)
            best_param2prune = copy(param2prune)
            best_idx_thresh = -1
            idx_thresh = 0
            while True:
                #get new parameter per mask-value
                ppm = []
                mask = []
                for i, (name_gate, gate) in enumerate(self.block_gates[name_block].items()):
                    new_ppm, _ = gate.get_param_per_mask_elem()
                    gate_mask = gate.get_hard_mask()
                    ppm.append(new_ppm.repeat(gate_mask.shape[0]))
                    mask.append(gate_mask)

                ppm = torch.cat(ppm)
                ppm_sort = ppm[sort_idx]
                ppm_sort = torch.cat([ppm_sort, placeholder_elem], dim=0)
                sort_ppm_cumsum = torch.cumsum(ppm_sort, dim=0)


                curr_sort_ppm_cumsum = (sort_ppm_cumsum - sort_ppm_cumsum[idx_thresh])
                idx_thresh = (curr_sort_ppm_cumsum < param2prune).sum().item()
                idx_thresh = min(idx_thresh, imp.shape[0])

                new_mask = self._get_new_mask(vec=imp, sort_idx=sort_idx, t_idx=idx_thresh, step=step)

                idx_cnt = 0
                for gate_idx, (name_gate, gate) in enumerate(self.block_gates[name_block].items()):
                    mask_size = gate.mask.m.shape[0]
                    new_gate_mask = new_mask[idx_cnt:idx_cnt+mask_size]
                    gate.set_new_mask(new_gate_mask.type(torch.float))
                    idx_cnt += mask_size

                param2prune = self._get_num2prune(block, kr=block.kr)

                if abs(param2prune) < abs(best_param2prune) or best_idx_thresh < 0:
                    best_param2prune = param2prune
                    best_idx_thresh = idx_thresh
                else:
                    #reset to best mask
                    new_mask = self._get_new_mask(vec=imp, sort_idx=sort_idx, t_idx=best_idx_thresh, step=step)
                    idx_cnt = 0
                    for gate_idx, (name_gate, gate) in enumerate(self.block_gates[name_block].items()):
                        mask_size = gate.mask.m.shape[0]
                        new_gate_mask = new_mask[idx_cnt:idx_cnt + mask_size]
                        gate.set_new_mask(new_gate_mask.type(torch.float))
                        idx_cnt += mask_size

                    break

                num_pruning_repititions += 1
                assert num_pruning_repititions <= 30, "too much iterations, find a more effective pruning-procedure"

            if logger is not None:
                for gate_idx, (name_gate, gate) in enumerate(self.block_gates[name_block].items()):
                    gate_mask = gate.get_hard_mask()
                    new_kr = gate_mask.sum().item() / gate_mask.numel()
                    logger.info(
                        f"block {block_idx}: gate {gate_idx} ({name_gate}): has a keep-raio of: {new_kr * 100:.2f}% ")

        self.reset_imp_scores()

        return kr_block

    def step(self):
        #update gate-importancy
        for name, m in self.model.named_modules():
            if isinstance(m, GateLayer):
                imp = taylor2Scorer(m.mask.m, m.mask.m.grad)[0]

                if name in self.imp_scores.keys():
                    self.imp_scores[name] += imp
                    self.imp_cnt[name] += 1
                else:
                    self.imp_scores[name] = imp
                    self.imp_cnt[name] = torch.ones(1, device=imp.device)













