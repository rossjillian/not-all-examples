from multiprocessing import Pool
import json
import os
from PIL import Image, ImageChops
from tqdm import tqdm
import numpy as np
import skimage
from skimage import img_as_ubyte, img_as_float32
from torch.utils.data.dataloader import default_collate
import random
import core
import config as cfg
from tqdm import tqdm


def main():
	args = cfg.parse_args()
	if args.bias != 0:
		file_prefix = 'cluster_%s_%s' % (args.clustering, args.dataset)
		if args.preprocess:
			file_prefix = file_prefix + '_preprocess'
		create_biased_dataset(file_prefix, args, bias_interval=args.bias, num_biased=args.num_biased, biased_category=args.biased_category)
	if args.generate_color_list:
		create_unbiased_dataset(args)


def create_biased_dataset(file_prefix, args, bias_interval=0.1, num_biased=1, biased_category=None):
	random.seed(args.seed)	
	img_list = os.path.join(args.cluster_dir, '%s_%d_input_names.txt' % (file_prefix, args.k))
	cluster_assignments = os.path.join(args.cluster_dir, '%s_%d_assignments.txt' % (file_prefix, args.k))
	img_names = np.genfromtxt(img_list, dtype='str')
	assignments = np.loadtxt(cluster_assignments)
	metadata = core.get_dataset_metadata(args.split, args)
	# TODO: generalize beyond COCO
	with open(metadata['color_metadata'], 'r') as f:
		color_metadata = json.load(f)
	img_dataset = [entry['file_name'] for entry in color_metadata]
	
	img_assignment_map = {}
	for i in tqdm(range(len(img_names))):
		if img_names[i] in img_dataset:
			if assignments[i] not in img_assignment_map:
				img_assignment_map[assignments[i]] = [img_names[i]]
			else:
				img_assignment_map[assignments[i]].append(img_names[i])
	
	print("Category distribution")
	print(np.unique(assignments, return_counts=True))
	categories = list(np.unique(assignments))
	if biased_category:
		assert(num_biased == len(biased_category.split(',')))
		bw_assignments = [float(category) for category in biased_category.split(',')]
	else:
		bw_assignments = random.sample(categories, num_biased)
	print("Biased categories")
	print(bw_assignments)
	bw_imgs = []
	
	color_assignments = np.setdiff1d(categories, bw_assignments)		
	for i in color_assignments:
		# Assign constant 5% bias to color categories
		num_bw = int(len(img_assignment_map[i]) * 0.05)
		bw_category_imgs = random.sample(img_assignment_map[i], num_bw)
		bw_imgs = bw_imgs + bw_category_imgs
	
	#TODO: temporary change
	bias = 0.2 
	prev_bias = 0.2
	args.bias = bias
	metadata = core.get_dataset_metadata(args.split, args)
	print("Bias: %f" % bias)
	for i in bw_assignments:
		# Assign specified bias to B&W categories
		num_bw = int(len(img_assignment_map[i]) * bias)
		print("Num B&W: %d" % num_bw)
		remaining_bw = [entry for entry in img_assignment_map[i] if entry not in bw_imgs]
		bw_category_imgs = random.sample(remaining_bw, num_bw)
		bw_imgs = bw_imgs + bw_category_imgs	
		print("Num selected: %d" % len(bw_imgs))
	
	if not os.path.exists(metadata['bw_dir']):
		os.makedirs(metadata['bw_dir'])
	file_name = os.path.join(metadata['bw_dir'], metadata['bw_metadata'])
	with open(file_name, 'w') as f:
		json.dump(bw_imgs, f)
	with open(file_name.replace('.json', '.txt'), 'w') as f:
	    f.write(str(np.unique(assignments, return_counts=True)))
	    f.write('\n')
	    f.write(str(bw_assignments))
	    f.write('\n')
	    f.write(str(num_bw))
	
	return
	
	for bias_split in range(1, 11):
		bias = bias_split * bias_interval 
		print("Bias: %f" % bias)
		args.bias = round(bias, 2)
		metadata = core.get_dataset_metadata(args.split, args)
		for i in bw_assignments:
			# Assign specified bias to B&W categories
			num_bw = int(len(img_assignment_map[i]) * (bias-prev_bias))
			print("Num B&W: %d" % num_bw)
			remaining_bw = [entry for entry in img_assignment_map[i] if entry not in bw_imgs]
			bw_category_imgs = random.sample(remaining_bw, num_bw)
			bw_imgs = bw_imgs + bw_category_imgs	
			print("Num selected: %d" % len(bw_category_imgs))
	
		prev_bias = bias
		if not os.path.exists(metadata['bw_dir']):
			os.makedirs(metadata['bw_dir'])
		file_name = os.path.join(metadata['bw_dir'], metadata['bw_metadata'])
		with open(file_name, 'w') as f:
			json.dump(bw_imgs, f)
		with open(file_name.replace('.json', '.txt'), 'w') as f:
		    f.write(str(np.unique(assignments, return_counts=True)))
		    f.write('\n')
		    f.write(str(bw_assignments))
		    f.write('\n')
		    f.write(str(num_bw))

	return 


def create_unbiased_dataset(args):
	"""
	Iterate over all images and captions file of just color images
	"""
	metadata = core.get_dataset_metadata(args.split, args)
	img_json = metadata['original_metadata']
	img_dir = metadata['img_dir']
	file_key = metadata['file_key']
	file_name = []
	if img_json:
		# Exists for train and val
		with open(img_json, 'r') as f:
			img_dict = json.load(f)
		if args.dataset == 'coco':
			img_dict = img_dict['images']

		for i in range(len(img_dict)):
			img = img_dict[i]
			file_path = os.path.join(img_dir, img[file_key])
			file_name.append([file_path, i])
	else:
		# Does not exist for test
		i = 0
		for img_file in os.listdir(img_dir):
			file_path = os.path.join(img_dir, img_file)
			file_name.append([file_path, i])
			i += 1

	results = []
	with Pool(25) as p:
		for result in tqdm(p.imap_unordered(is_grey_scale, file_name), total=len(file_name)):
			if result:
				if img_json:
					results.append(img_dict[result])
				else:
					results.append({'file_name':file_name[result][0]})

	with open(metadata['color_metadata'], 'w') as f:
		json.dump(results, f)	

	return True


def is_grey_scale(img_filename):
	try:
		img = Image.open(img_filename[0]).convert('RGB')	
		rgb = img.split()
		if ImageChops.difference(rgb[0],rgb[1]).getextrema()[1]!=0: 
			return img_filename[1] # Color
		if ImageChops.difference(rgb[0],rgb[2]).getextrema()[1]!=0:         
			return img_filename[1] # Color
		return # Grey
	except OSError:
		return


if __name__ == '__main__':
	main()
