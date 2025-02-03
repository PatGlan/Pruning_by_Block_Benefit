# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.


from typing import Tuple
from dataclasses import dataclass

import os
import torch
import json
from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader

import utils
from .samplers import RASampler
from data.augment import new_data_aug_generator
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD



@dataclass
class ImageDatasetConstants:
    dataset_size: int
    num_classes: int
    crop_size: int
    channel_means: Tuple[float]
    channel_stds: Tuple[float]

@dataclass
class ImageNetConstants(ImageDatasetConstants):
    dataset_size: int = 1281167
    num_classes: int = 1000
    crop_size: int = 224
    channel_means: Tuple[float] = (0.485, 0.456, 0.406)
    channel_stds: Tuple[float] = (0.229, 0.224, 0.225)

@dataclass
class Cifar10Constants(ImageDatasetConstants):
    dataset_size: int = 50000
    num_classes: int = 10
    crop_size: int = 32
    channel_means: Tuple[float] = (0.4914, 0.4822, 0.4465)
    channel_stds: Tuple[float] = (0.2023, 0.1994, 0.2010)

@dataclass
class Cifar100Constants(ImageDatasetConstants):
    dataset_size: int = 50000
    num_classes: int = 100
    crop_size: int = 32
    channel_means: Tuple[float] = (0.5071, 0.4867, 0.4408)
    channel_stds: Tuple[float] = (0.2675, 0.2565, 0.2761)

@dataclass
class INat18Constants(ImageDatasetConstants):
    dataset_size: int = 0
    num_classes: int = 8142
    crop_size: int = 224
    channel_means: Tuple[float] = (0.485, 0.456, 0.406)
    channel_stds: Tuple[float] = (0.229, 0.224, 0.225)

@dataclass
class INat19Constants(ImageDatasetConstants):
    dataset_size: int = 0
    num_classes: int = 1010
    crop_size: int = 224
    channel_means: Tuple[float] = (0.485, 0.456, 0.406)
    channel_stds: Tuple[float] = (0.229, 0.224, 0.225)

@dataclass
class IFoodConstants(ImageDatasetConstants):
    dataset_size: int = 0
    num_classes: int = 251
    crop_size: int = 224
    channel_means: Tuple[float] = (0.485, 0.456, 0.406)
    channel_stds: Tuple[float] = (0.229, 0.224, 0.225)

def get_dataset_constants(dataset: str):
    constants_dict = {
        "IMNET": ImageNetConstants,
        "CIFAR10": Cifar10Constants,
        "CIFAR100": Cifar100Constants,
        "INAT18": INat18Constants,
        "INAT19": INat19Constants,
        "IFOOD": IFoodConstants,
    }

    if dataset not in constants_dict:
        raise ValueError(f"Invalid dataset name: {dataset}")
    return constants_dict[dataset]()


class INatDataset(ImageFolder):
    def __init__(self, root, train=True, year=2018, transform=None, target_transform=None,
                 category='name', loader=default_loader):
        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform
        self.year = year
        # assert category in ['kingdom','phylum','class','order','supercategory','family','genus','name']
        path_json = os.path.join(root, f'{"train" if train else "val"}{year}.json')
        with open(path_json) as json_file:
            data = json.load(json_file)

        with open(os.path.join(root, 'categories.json')) as json_file:
            data_catg = json.load(json_file)

        path_json_for_targeter = os.path.join(root, f"train{year}.json")

        with open(path_json_for_targeter) as json_file:
            data_for_targeter = json.load(json_file)

        targeter = {}
        indexer = 0
        for elem in data_for_targeter['annotations']:
            king = []
            king.append(data_catg[int(elem['category_id'])][category])
            if king[0] not in targeter.keys():
                targeter[king[0]] = indexer
                indexer += 1
        self.nb_classes = len(targeter)

        self.samples = []
        for elem in data['images']:
            cut = elem['file_name'].split('/')
            target_current = int(cut[2])
            path_current = os.path.join(root, cut[0], cut[2], cut[3])

            categors = data_catg[target_current]
            target_current_true = targeter[categors[category]]
            self.samples.append((path_current, target_current_true))


def get_dataloader(args):
    data_const = get_dataset_constants(args.data_set)
    args.nb_classes = data_const.num_classes

    dataset_train = build_dataset(is_train=True, args=args)
    dataset_val = build_dataset(is_train=False, args=args)

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        if args.repeated_aug:
            sampler_train = RASampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(2.0 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    return data_loader_train, data_loader_val, dataset_train, dataset_val

def get_val_dataloader (args):
    data_const = get_dataset_constants(args.data_set)
    args.nb_classes = data_const.num_classes

    dataset_val = build_dataset(is_train=False, args=args)

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(2.0 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    return data_loader_val, dataset_val


def _get_tranform(args, is_train):
    resize_im = args.input_size > 32
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD

    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std
        )
        if not resize_im:
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * args.input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)



def build_dataset(is_train, args):
    transform = _get_tranform(args, is_train)


    if args.data_set == 'CIFAR10':
        dataset = datasets.CIFAR10(args.data_path, train=is_train, transform=transform, download=True)
    elif args.data_set == 'CIFAR100':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform, download=True)
    elif args.data_set == 'IMNET':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
    elif args.data_set == 'INAT18':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
    elif args.data_set == 'INAT19':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
    elif args.data_set == 'IFOOD':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
    else:
        assert False, f"no Dataset: {args.data_set}"
    return dataset


