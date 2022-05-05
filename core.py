from dataset import COCODataset, AugCOCODataset
from model import RotAlexNet, DownstreamRotAlexNet
import numpy as np
import torchvision
from torchvision import transforms
from torch import optim, utils
import torch.nn as nn
import torch
from torch.utils.data.dataloader import default_collate
import os
from utils import ConvertNumpy
from utils import rot_collate
from collections import OrderedDict


def get_dataset(data_root, split, model_name, args):
	args = update_args(split, args)
	metadata = get_dataset_metadata(split, args)
	transform = get_transforms(args)
	augment = get_augmentation(args)

	if args.dataset == 'coco':
		if args.aug != 0:
			return AugCOCODataset(
				metadata['img_dir'],
				metadata['color_metadata'], 
				os.path.join(metadata['bw_dir'], metadata['bw_metadata']),
				args,
				transform=transform,
				augment=augment
			)
		else:
			return COCODataset(
				metadata['img_dir'],
				metadata['color_metadata'],
				args,
				transform=transform
			)
	elif args.dataset == 'cifar100':	
		if args.split == 'train':
			return torchvision.datasets.CIFAR100(
				metadata['img_dir'],
				transform=transform
			)
		else:
			return torchvision.datasets.CIFAR100(
				metadata['img_dir'], 
				train=False,
				transform=transform
			)


def get_dataset_metadata(split, args):
	metadata = {}
	# COCO
	if args.dataset == 'coco':
		metadata['file_key'] = 'file_name'
		if split == 'train':
			metadata['original_metadata'] = os.path.join(args.data_root, 'annotations', 'captions_train2014.json')
			metadata['color_metadata'] = 'coco_train_metadata_color.json'
			metadata['img_dir'] = os.path.join(args.data_root, 'train2014')
			metadata['bw_dir'] = './bias/coco/train'
			if args.aug != 0:
				metadata['bw_metadata'] = 'metadata_%s_%d_%s_%d_%s_bw-%d.json' % (args.clustering, args.k, str(args.aug).replace('.', ''), args.num_aug, args.aug_category, args.repeat)
		elif split == 'val':
			metadata['original_metadata'] = os.path.join(args.data_root, 'annotations', 'captions_val2014.json')
			metadata['color_metadata'] = 'coco_val_metadata_color.json'
			metadata['img_dir'] = os.path.join(args.data_root, 'val2014')
			metadata['bw_dir'] = './bias/coco/val'
			if args.aug != 0:
				metadata['bw_metadata'] = 'metadata_%s_%d_%s_%d_%s_bw-%d.json' % (args.clustering, args.k, str(args.aug).replace('.', ''), args.num_aug, args.aug_category, args.repeat)
		elif split == 'test':
			metadata['original_metadata'] = None
			metadata['color_metadata'] = 'coco_test_metadata_color.json'
			metadata['img_dir'] = os.path.join(args.data_root, 'test2014')
			metadata['bw_dir'] = './bias/coco/test'
	# CIFAR100
	elif args.dataset == 'cifar100':
		metadata['img_dir'] = os.path.join(args.data_root)
		metadata['base_model'] = RotAlexNet()
	return metadata


def update_args(split, args):
	if args.model_name == 'rotnet':
		if split != 'train' or args.features_only:
			args.pre_transform = transforms.Compose(
				[
					transforms.Resize(256), 
					transforms.CenterCrop(224),
					ConvertNumpy()
				]
			)
		else:
			args.pre_transform = transforms.Compose(
				[
					transforms.Resize(256),
					transforms.RandomCrop(224),
					transforms.RandomHorizontalFlip(),
					ConvertNumpy()
				]
			)
	return args


def get_transforms(args):
	mean = [0.485, 0.456, 0.406]
	std = [0.229, 0.224, 0.225]
	
	if args.dataset == 'coco':
		transform = transforms.Compose(
			[
				transforms.ToTensor(),
				transforms.Normalize(mean=mean, std=std)
			]
		)
	elif args.dataset == 'cifar10' or args.dataset == 'cifar100':	
		transform = transforms.Compose(
			[
				transforms.Resize(256),
				transforms.CenterCrop(224),
				transforms.ToTensor(),
				transforms.Normalize(mean=mean, std=std)
			]
		)

	return transform


def get_augmentation(args):
	if args.aug_transform == 'bw':
		augment = transforms.Compose(
			[
				transforms.Grayscale(num_output_channels=3)
			]	
		)
	elif args.aug_transform == 'blur':
		augment = transforms.Compose(
			[
				transforms.GaussianBlur(kernel_size=(5,5))
			]	
		)
	elif args.aug_transform == 'jitter':
		augment = transforms.Compose(
			[
				transforms.ColorJitter()
			]	
		)
	return augment


def get_dataloader(dataset, batch_size, num_workers, args, ngpus=None, gpu=None, drop_last=True):
	if args.features_only:
		collate_fn = default_collate
	elif args.model_name == 'rotnet':
		collate_fn = rot_collate
	else:
		collate_fn = default_collate

	if args.ddp:
		assert(ngpus != None and gpu != None)
		sampler = utils.data.distributed.DistributedSampler(
			dataset, num_replicas=ngpus, rank=gpu
		)
		loader = utils.data.DataLoader(
			dataset,
			batch_size=batch_size,
			num_workers=num_workers,
			sampler=sampler,
			collate_fn=collate_fn, 
			drop_last=drop_last
		)
		return loader
	else:
		loader = utils.data.DataLoader(
			dataset, 
			batch_size=batch_size, 
			num_workers=num_workers,
			collate_fn=collate_fn,
			drop_last=drop_last
		)
		return loader


def get_model(args):
	if args.model_name == 'rotnet':
		model = RotAlexNet()
	elif args.model_name == 'rotnetdown':
		base_model = RotAlexNet()
		model = DownstreamRotAlexNet(base_model, args.num_classes)
		if 'rotnetdown' not in args.pretrained:
			print('Loading pretrained base')
			base_state_dict = torch.load(args.pretrained)
			model_state_dict = model.state_dict()
			modified_state_dict = OrderedDict()
			for k, v in base_state_dict.items():
				name = k.replace('module', 'modified_rotnet')
				modified_state_dict[name] = v
			modified_state_dict = {k: v for k, v in modified_state_dict.items() if k in model_state_dict}
			model_state_dict.update(modified_state_dict)
			model.load_state_dict(model_state_dict)
	else:
		print('Model not implemented')
	return model


def get_model_metadata(args):
	metadata = {}
	if args.model_name == 'rotnet' or args.model_name == 'rotnetdown':
		metadata['layers'] = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc_block']
	elif args.model_name == 'patchnet':
		metadata['layers'] = ['cnn', 'fc6', 'fc']
	return metadata


def get_loss(args):
	return nn.CrossEntropyLoss()


def get_optimizer(parameters, lr, args):
	return optim.Adam(parameters, lr=lr)


