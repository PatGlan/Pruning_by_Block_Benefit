# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import argparse
import datetime
import numpy as np
import time

import torch
import torch.backends.cudnn as cudnn
import json

from pathlib import Path

from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from utils.optimizer import create_optimizer, NativeScaler
from timm.utils import get_state_dict, ModelEma

from data.datasets import get_dataloader
from engine import train_one_epoch, evaluate
from loss import DistillationLoss, PruneLoss
from fvcore.nn import FlopCountAnalysis


import utils.py_utils as py_utils

from utils.args import get_args_parser
from utils.logger import get_logger
from utils.model_utils import get_new_attn_dim, get_block_masks
from models.model_loader import get_model
from models.FmAnalyseHead import FeaturemapAnalyseHead
from pruner.forward_functions import update_model_forward_functions, update_model_gate_layer
from pruner import freeze_model, count_kept_param_per_block




def main(args):

    device = torch.device(args.device)
    logger = get_logger(args, gpu_id=torch.cuda.current_device())

    # fix the seed for reproducibility
    seed = args.seed + py_utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    data_loader_train, data_loader_val, dataset_train, dataset_val = get_dataloader(args)

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    logger.info(f"Creating model: {args.model}")
    model, gate_func_added = get_model(args)
    model = update_model_forward_functions(model, args, args.model)

    # count Flops
    flop_analyser = FlopCountAnalysis(model.cuda(),
                              inputs=torch.rand(1, 3, args.input_size, args.input_size).cuda())
    logger.info(f"FLOPS TOTAL before Pruning: {flop_analyser.total()/1e9:.3f} G")

    # count Parameters
    n_param_dense = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"PARAM TOTAL before Pruning: {n_param_dense / 1e6:.3f} M")

    if args.prune_ratio is not None and not args.model_freeze:
        update_model_gate_layer(model, args)

    if mixup_active:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.0:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()


    if args.prune_ratio is not None and args.prune_kr_handcraft is None:
        model.fmHead = FeaturemapAnalyseHead(model, criterion, logger, args, steps2log=args.prune_steps_update_mask)

        if args.model_path is not None and ".npz" not in args.model_path:
            checkpoint = torch.load(args.model_path, map_location='cpu')
            if "model" in checkpoint.keys():
                checkpoint = checkpoint["model"]
            if any(["fmHead" in k for k in checkpoint.keys()]):
                fm_checkpoint = {k.replace("fmHead.",""):v for k,v in checkpoint.items() if "fmHead" in k}
                model.fmHead.load_state_dict(fm_checkpoint, strict=True)
                logger.info("fmHead loaded from checkpoint")

    model.to(device)

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')

    model_without_ddp = model
    if args.distributed:
        find_unused_parameters=False
        #logger.info(f"GPU: {args.gpu}")
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=find_unused_parameters)
        model_without_ddp = model.module

    if not args.unscale_lr:
        linear_scaled_lr = args.lr * args.batch_size * py_utils.get_world_size() / 512.0
        linear_scaled_lr = linear_scaled_lr * args.grad_accum_steps
        args.lr = linear_scaled_lr

    optimizer = create_optimizer(args, model_without_ddp)
    loss_scaler = NativeScaler()

    lr_scheduler = None
    if args.prune_ratio is None:
        lr_scheduler, _ = create_scheduler(args, optimizer)


    if args.distillation_type != "none":
        print(f"Creating teacher model: {args.teacher_model}")
        teacher_model = create_model(
            args.teacher_model,
            pretrained=False,
            num_classes=args.nb_classes,
        )
        #global_pool = 'avg',
        if args.teacher_path.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.teacher_path, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.teacher_path, map_location='cpu')
        teacher_model.load_state_dict(checkpoint['model'])
        teacher_model.to(device)
        teacher_model.eval()

        criterion = DistillationLoss(
            criterion, teacher_model, args.distillation_type, args.distillation_alpha, args.distillation_tau
        )

    if args.prune_ratio is not None:
        criterion = PruneLoss(criterion, model_without_ddp, len(data_loader_train), args, logger=logger)

    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval:
            if 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
            if 'lr_scheduler' in checkpoint and checkpoint['lr_scheduler'] is not None:
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            if 'epoch' in checkpoint:
                args.start_epoch = checkpoint['epoch'] + 1
            if args.model_ema:
                py_utils._load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
        lr_scheduler.step(args.start_epoch)

    if args.eval:
        test_stats = evaluate(data_loader_val, model, device, args, logger)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        return

    logger.info(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, model_ema, mixup_fn,
            set_training_mode=args.distillation_type != 'none',  # keep in eval mode for deit finetuning / train mode for training and deit III finetuning
            args=args, logger=logger
        )

        if lr_scheduler is not None:
            lr_scheduler.step(epoch)

        if args.output_dir and torch.cuda.current_device() == 0:
            checkpoint_path = output_dir / 'checkpoint.pth'
            num_heads, num_qkv_embed = get_new_attn_dim(model_without_ddp)
            masks_block = get_block_masks(model_without_ddp)
            py_utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'num_heads': num_heads,
                'num_qkv_embed': num_qkv_embed,
                'masks_block': masks_block,
                'max_accuracy': max_accuracy,
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict() if lr_scheduler is not None else None,
                'epoch': epoch,
                'model_ema': get_state_dict(model_ema) if model_ema is not None else None,
                'scaler': loss_scaler.state_dict(),
                'args': args,
            }, checkpoint_path)

        test_stats = evaluate(data_loader_val, model, device, args, logger, epoch)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")

        #if max_accuracy < test_stats["acc1"] and epoch > args.prune_finalize_epoch:
        if max_accuracy < test_stats["acc1"]:
            max_accuracy = test_stats["acc1"]
            if args.output_dir and torch.cuda.current_device() == 0:
                checkpoint_paths = [output_dir / 'best_checkpoint.pth']
                num_heads, num_qkv_embed = get_new_attn_dim(model_without_ddp)
                masks_block = get_block_masks(model_without_ddp)
                for checkpoint_path in checkpoint_paths:
                    py_utils.save_on_master({
                        'model': model_without_ddp.state_dict(),
                        'num_heads': num_heads,
                        'num_qkv_embed': num_qkv_embed,
                        'masks_block': masks_block,
                        'max_accuracy': max_accuracy,
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict() if lr_scheduler is not None else None,
                        'epoch': epoch,
                        'model_ema': get_state_dict(model_ema) if model_ema is not None else None,
                        'scaler': loss_scaler.state_dict(),
                        'args': args,
                    }, checkpoint_path)

        logger.info(f'Max accuracy: {max_accuracy:.2f}%')

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_param_dense}




        if args.output_dir and py_utils.is_main_process():
            with (output_dir / "results.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # prune model
        if args.prune_ratio is not None and (epoch + 1) == args.prune_finalize_epoch:
            torch.cuda.synchronize()

            _, param_blockwise_before_pruning = count_kept_param_per_block(model_without_ddp)
            kr_block = criterion.prune_forward(logger=logger, finalize_pruning=True)
            logger.info(f"goal_kr_block: " + ";".join(format(x, ".3f") for x in kr_block))

            freeze_model(model_without_ddp)
            optimizer = create_optimizer(args, model_without_ddp)

            checkpoint_paths = [output_dir / 'first_sparse_checkpoint.pth', output_dir / 'checkpoint.pth']
            num_heads, num_qkv_embed = get_new_attn_dim(model_without_ddp)
            masks_block = get_block_masks(model_without_ddp)
            for checkpoint_path in checkpoint_paths:
                py_utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'num_heads': num_heads,
                    'num_qkv_embed': num_qkv_embed,
                    'masks_block': masks_block,
                    'max_accuracy': max_accuracy,
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict() if lr_scheduler is not None else None,
                    'epoch': epoch,
                    'model_ema': get_state_dict(model_ema) if model_ema is not None else None,
                    'scaler': loss_scaler.state_dict(),
                    'args': args,
                }, checkpoint_path)

            #flops statistics
            flops_remain = FlopCountAnalysis(model.cuda(),
                                            inputs=torch.rand(1, 3, args.input_size, args.input_size).cuda()).total()
            # param statistics
            n_param_sparse = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"PARAM TOTAL before Pruning: {n_param_sparse / 1e6:.3f} M")

            #parameter statistics
            #param_remain = count_block_parameters(model_without_ddp)[0]
            _, param_blockwise_after_pruning = count_kept_param_per_block(model_without_ddp)
            logger.info("********* model keep_ratio per block *********")
            for i, (p_unpr, p_kept) in enumerate(zip(param_blockwise_before_pruning, param_blockwise_after_pruning)):
                logger.info(f"remain_param block {i:2d}: [{p_kept:,d} / {p_unpr:,d}] is_kr: {p_kept*100/p_unpr:.2f}%  | "
                            f"goal_kr: {kr_block[i]*100:.2f}%")
            #logger.info(f"goal_kr_block: " + ";".join(format(x, ".3f") for x in kr_block))
            logger.info("********* model pruned finally *********")
            logger.info(f"remain_flops: {flops_remain / 1e9:.3f} G")
            logger.info(f"remain_param: [{n_param_sparse / 1e6:.1f} M / {n_param_dense / 1e6:.1f} M]")  #
            logger.info(f"prune_param: [{(n_param_dense - n_param_sparse) / 1e6:.1f} M / {n_param_dense / 1e6:.1f} M]")  #
            logger.info(f"prune-ratio: {(n_param_dense - n_param_sparse)/n_param_dense}")  #
            kr_block = criterion.get_kr_block_without_update()
            logger.info(f"final_is_kr_block: " + ";".join(format(x, ".3f") for x in kr_block))
            skip_block_flags = criterion.get_skip_block_flag()
            logger.info(f"final blocks been skipped: " + ";".join(str(x) for x in skip_block_flags))
            logger.info("****************************************")
            #n_param_dense = n_param_sparse
            # flops_stat = count_flops(model)
            # print(f"remain_flops: [{flops_stat[0]/1e6} M / {flops_stat[1]/1e6} M]")  #


        # break trainings run if epoch limit of run has reached
        if (args.epochs_run is not None and (epoch - args.start_epoch + 1) >= args.epochs_run):
            break

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    for k, v in vars(args).items():
        if type(v) == str and v == "null":
            setattr(args, k, None)


    py_utils.init_distributed_mode(args)

    print(args)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)

