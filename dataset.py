from utils import rotate_img, get_patch_pair
from torch.utils.data import Dataset
from multiprocessing import Pool
import json
import os
from PIL import Image, ImageChops
from tqdm import tqdm
import numpy as np
import skimage.transform
from skimage import img_as_ubyte, img_as_float32
import random
import torch


# COCO 


class COCODataset(Dataset):
	def __init__(self, img_dir, caption_file, args, transform=None, logging_file='./coco_data_log.txt'):
		self.img_dir = img_dir
		with open(caption_file, 'r') as f:
			self.caption_file = json.load(f)
		"""
		if args.only_load:
			only_load_set = []
			for img_idx in range(len(self.caption_file)):
				if self.caption_file[img_idx]['file_name'] in args.only_load:
					only_load_set.append({'file_name' : self.caption_file[img_idx]['file_name']})
			self.caption_file = only_load_set
		"""
		self.args = args
		self.transform = transform
		self.log = logging_file

	def __len__(self):
		return len(self.caption_file)

	def __getitem__(self, idx):
		loaded = 0
		img_idx = idx
		while not loaded:
			img_file = self.caption_file[img_idx]['file_name']
			try:
				img_path = os.path.join(self.img_dir, img_file)
				img = Image.open(img_path).convert('RGB')
				loaded = 1	
			except:		
				print("Exception encountered in Dataset")
				print(img_file)
				with open(self.log, 'a+') as f:
					f.write(img_file + ',')					
				img_idx = random.randint(0, len(self.caption_file))		
		
		# TODO: clean up
		if self.args.model_name == 'patchnet':
			return get_patch_pair(img, self.patch_dim, self.gap, transform=self.transform)
		elif self.args.model_name == 'rotnet':
			img = self.args.pre_transform(img)
			if self.args.features_only:
				return self.transform(img), img_file
			else:
				rotated_imgs = [
								self.transform(img),
								self.transform(rotate_img(img, 90)),
								self.transform(rotate_img(img, 180)),
								self.transform(rotate_img(img, 270))
								]			
				rotation_labels = torch.LongTensor([0, 1, 2, 3])
				return torch.stack(rotated_imgs, dim=0), rotation_labels


class BiasedCOCODataset(COCODataset):
	def __init__(self, img_dir, caption_file, bias_assignments, args, transform=None, logging_file='./coco_data_log.txt'):
		with open(bias_assignments, 'r') as f:
			self.bias_assignments = json.load(f)
		super(BiasedCOCODataset, self).__init__(img_dir, caption_file, args, transform=transform, logging_file=logging_file)
	
	def __getitem__(self, idx):
		# TODO: gather captions
		loaded = 0
		img_idx = idx
		while not loaded:
			img_file = self.caption_file[img_idx]['file_name']
			try:
				img_path = os.path.join(self.img_dir, img_file)
				if img_file in self.bias_assignments:
					img = Image.open(img_path).convert('L').convert('RGB')
				else:
					img = Image.open(img_path).convert('RGB')
				loaded = 1	
			except:		
				print("Exception encountered in Dataset")
				print(self.caption_file[img_idx]['filename'])
				with open(self.log, 'a+') as f:
					f.write(self.caption_file[img_idx]['filename'] + ',')					
				img_idx = random.randint(0, len(self.caption_file))		
		
		# TODO: clean up
		if self.args.model_name == 'patchnet':
			return get_patch_pair(img, self.patch_dim, self.gap, transform=self.transform)
		elif self.args.model_name == 'rotnet':
			img = self.args.pre_transform(img)
			if self.args.features_only:
				return self.transform(img), img_file
			else:
				rotated_imgs = [
								self.transform(img),
								self.transform(rotate_img(img, 90)),
								self.transform(rotate_img(img, 180)),
								self.transform(rotate_img(img, 270))
								]			
				rotation_labels = torch.LongTensor([0, 1, 2, 3])
				return torch.stack(rotated_imgs, dim=0), rotation_labels


class COCOFeatures(Dataset):
	def __init__(self, features_dir, categories, feature_order):
		self.features_dir = features_dir
		self.categories = categories
		self.feature_files = []
		# Iterate over category
		for category in os.listdir(features_dir):
			if os.path.isdir(os.path.join(features_dir, category)) and category in categories:
				for category_feature in os.listdir(os.path.join(features_dir, category)):
					if category_feature.replace('.pt', '.jpg') in feature_order:
						self.feature_files.append(os.path.join(category, category_feature))

	def __len__(self):
		return len(self.feature_files)

	def __getitem__(self, idx):
		feature_path = os.path.join(self.features_dir, self.feature_files[idx])
		return torch.load(feature_path)	


