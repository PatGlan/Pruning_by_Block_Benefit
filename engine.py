# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main_manifold.py
"""
import math
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from loss.dist_loss import DistillationLoss
from pruner import count_parameters_model_hard_masked
import utils.py_utils as py_utils

def train_one_epoch(model: torch.nn.Module,
                    criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, args = None, logger=None):
    model.train(set_training_mode)
    metric_logger = py_utils.MetricLogger(delimiter="  ", logger=logger)
    metric_logger.add_meter('lr', py_utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch + 1)
    print_freq = args.print_freq

    step = int(epoch * len(data_loader))
    step_warmup = int(args.prune_start_epoch * len(data_loader))
    step_sparse = int(args.prune_sparse_epoch * len(data_loader))
    step_finalize = int(args.prune_finalize_epoch * len(data_loader))
    log_block_intrinic_kr_done=False

    if args.distributed:
        model_without_ddp = model.module
    else:
        model_without_ddp = model

    optimizer.zero_grad()

    for i, (inputs, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        is_prune_step = args.prune_ratio is not None and (step >= step_warmup) and (step < step_finalize)

        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        #int_target = targets
        if mixup_fn is not None:
            inputs, targets = mixup_fn(inputs, targets)

        #parameter
        #if teacher_model is not None:
        #    with torch.no_grad():
        #        t_x_inter, t_p = teacher_model(inputs)

        if is_prune_step and ((step - step_warmup)%args.prune_steps_update_mask) == (args.prune_steps_update_mask - 1):
            if not log_block_intrinic_kr_done:
                kr_block = criterion.prune_forward(step)
                log_block_intrinic_kr_done = True
            else:
                kr_block = criterion.prune_forward(step)

            param_stat = count_parameters_model_hard_masked(model_without_ddp, args)
            logger.info(f"remain_param model: [{param_stat[0]/1e6:.1f}M / {param_stat[1]/1e6:.1f}M]")  #
            logger.info(f"goal_kr_block: " + ";".join(format(x, ".3f") for x in kr_block))

        with torch.cuda.amp.autocast():
            if args.prune_ratio is not None:
                s_p, s_p_inter, s_x_inter = model(inputs)
                loss = criterion(s_p=s_p, inputs=inputs, targets=targets, step=step, s_p_inter=s_p_inter, s_x_inter=s_x_inter)

            else:
                s_p = model(inputs)
                if args.distillation_type == "none":
                    if type(s_p) is tuple:
                        s_p = s_p[0]
                    loss = criterion(s_p, targets)
                else:
                    loss = criterion(inputs, (s_p[0], s_p[1]), targets)

            loss = loss / args.grad_accum_steps

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            logger.info("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        is_grad_accum_step = ((i + 1) % args.grad_accum_steps != 0) and (i + 1) != len(data_loader)

        loss_scaler(loss, optimizer, clip_grad=max_norm, parameters=model.parameters(),
                    create_graph=is_second_order, skip_update=is_grad_accum_step)
        if is_grad_accum_step:
            metric_logger.update(loss=loss_value)
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
            step += 1
            continue

        # this attribute is added by timm on one optimizer (adahessian)
        #is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        #loss_scaler(loss, optimizer, clip_grad=max_norm, parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        if args.prune_ratio is not None and is_prune_step:
            criterion.step()

        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        optimizer.zero_grad()
        step += 1


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    logger.info(f"Averaged stats:{metric_logger}")
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, args, logger, epoch=0):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = py_utils.MetricLogger(delimiter="  ", logger=logger)
    header = 'Test:'

    torch.cuda.synchronize()

    # switch to evaluation mode
    model.eval()

    p_interm = None
    for inputs, targets in metric_logger.log_every(data_loader, args.print_freq, header):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            if args.prune_ratio is not None:
                outputs, p_interm, _ = model(inputs)
            else:
                outputs = model(inputs)
                if type(outputs) is tuple:
                    outputs = outputs[0]

            loss = criterion(outputs, targets)

            if p_interm is not None:
                loss_soft_cls = criterion(p_interm["p_cls_out"][-1], targets)
            else:
                loss_soft_cls = torch.zeros_like(loss)

        #loss_item = loss.item()
        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        if p_interm is not None:
            acc1_soft_cls, acc5_soft_cls = accuracy(p_interm["p_cls_out"][-1], targets, topk=(1, 5))
            acc1_soft_patch, acc5_soft_patch = accuracy(p_interm["p_patch_out"][-1], targets, topk=(1, 5))
        else:
            acc1_soft_cls, acc5_soft_cls = torch.zeros_like(acc1), torch.zeros_like(acc5)
            acc1_soft_patch, acc5_soft_patch = torch.zeros_like(acc1), torch.zeros_like(acc5)

        batch_size = inputs.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        if args.prune_ratio is not None:
            metric_logger.update(loss_soft_cls=loss_soft_cls.item())
            metric_logger.meters['acc1_soft_cls'].update(acc1_soft_cls.item(), n=batch_size)
            metric_logger.meters['acc5_soft_cls'].update(acc5_soft_cls.item(), n=batch_size)
            metric_logger.meters['acc1_soft_patch'].update(acc1_soft_patch.item(), n=batch_size)
            metric_logger.meters['acc5_soft_patch'].update(acc5_soft_patch.item(), n=batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    if args.prune_ratio is not None:
        logger.info('Test: after epoch: {epoch} | Acc_soft_cls@1 {top1.global_avg:.3f} | '
                    'Acc_soft_cls@5 {top5.global_avg:.3f} | loss_soft_cls {losses.global_avg:.3f}'
              .format(epoch=epoch+1, top1=metric_logger.acc1_soft_cls,
                      top5=metric_logger.acc5_soft_cls, losses=metric_logger.loss_soft_cls))
        logger.info('Test: after epoch: {epoch} | Acc_soft_patch@1 {top1.global_avg:.3f} | '
                    'Acc_soft_patch@5 {top5.global_avg:.3f} | loss_soft_patch {losses.global_avg:.3f}'
            .format(epoch=epoch+1, top1=metric_logger.acc1_soft_patch,
                    top5=metric_logger.acc5_soft_patch, losses=metric_logger.loss_soft_cls))
    logger.info('Test: after epoch: {epoch} | Acc@1 {top1.global_avg:.3f} | '
                'Acc@5 {top5.global_avg:.3f} | loss {losses.global_avg:.3f} '
          .format(epoch=epoch+1, top1=metric_logger.acc1,
                  top5=metric_logger.acc5, losses=metric_logger.loss))


    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
