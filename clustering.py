import numpy as np
from sklearn import cluster, manifold
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler 
import config as cfg
import core
from model import RotAlexNet, PatchAlexNet
import torch
import torch.nn as nn
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.colors as colors
import os
from PIL import Image
import statistics


def main():
	args = cfg.parse_args()
	features, img_names = get_features(args.model_name, args.pretrained, args.data_root, args)
	if args.preprocess:
		features = preprocess_features(features, components=256)
	
	if args.visualize:
		get_visualization(features, img_names, args)
	
	else:
		file_name = 'cluster_%s_%s' % (args.clustering, args.dataset)
		if args.preprocess:
			file_name = file_name + '_preprocess'
		if args.optimize_clustering:
			k = get_optimal_clusters(features, args, file_name)
		else:
			k = args.k	
		if args.clustering == 'kmeans':
			cluster_technique = cluster.MiniBatchKMeans(n_clusters=k)
		elif args.clustering == 'dbscan':
			cluster_technique = cluster.DBSCAN(n_jobs=-1)
		elif args.clustering == 'optics':
			cluster_technique = cluster.OPTICS(n_jobs=-1)
		cluster_technique.fit(features)
		assignments = cluster_technique.labels_
		
		silhouette_scores = silhouette_score(features, assignments)
		print("Silhouette score: %f" % silhouette_scores)
		db_scores = davies_bouldin_score(features, assignments)
		print("DB score: %f" % db_scores)
		
		np.savetxt(os.path.join(args.cluster_dir, '%s_%d_assignments.txt' % (file_name, k)), assignments)
		np.savetxt(os.path.join(args.cluster_dir, '%s_%d_input_names.txt' % (file_name, k)), img_names, fmt='%s')
		torch.save(features, os.path.join(args.cluster_dir, '%s_%d_features.pt' % (file_name, k)))
		with open(os.path.join(args.cluster_dir, '%s_%d_params.json' % (file_name, k)), 'w') as f:
			json.dump(cluster_technique.get_params(), f)
	return


def get_features(model_name, pretrained, data_root, args):
	"""
	Get features from both train and val
	Do clustering over both
	"""
	if model_name == 'rotnet':
		model = RotAlexNet()	
	else:
		print('Model not implemented')

	model = nn.DataParallel(model).cuda()
	model.load_state_dict(torch.load(pretrained))

	train_dataset = core.get_dataset(data_root, 'train', model_name, args)
	val_dataset = core.get_dataset(data_root, 'val', model_name, args)
	if args.view_subset:
		train_dataset = torch.utils.data.Subset(train_dataset, range(0, 1000))
		val_dataset = torch.utils.data.Subset(val_dataset, range(0, 1000))
	train_loader = core.get_dataloader(train_dataset, args.batch_size, args.num_workers, args)
	val_loader = core.get_dataloader(val_dataset, args.batch_size, args.num_workers, args)	

	features = None
	img_names = []
	with tqdm(train_loader, unit='batch') as tepoch:
		with torch.no_grad():
			for inputs, targets in tepoch:
				inputs = inputs.cuda(args.gpu, non_blocking=True)
				input_features = model(inputs, out_feat_keys=['fc_block']) 
				if features == None:
					features = input_features.cpu()
				else:
					features = torch.cat((features, input_features.cpu()))
				img_names.extend(list(targets))
	with tqdm(val_loader, unit='batch') as tepoch:
		with torch.no_grad():
			for inputs, targets in tepoch:
				inputs = inputs.cuda(args.gpu, non_blocking=True)
				input_features = model(inputs, out_feat_keys=['fc_block']) 
				if features == None:
					features = input_features.cpu()
				else:
					features = torch.cat((features, input_features.cpu()))
				img_names.extend(list(targets))
	return features, img_names


def get_image(path, args, bw_metadata=None, zoom=0.0095):
	im = Image.open(path)
	if bw_metadata is not None:
		bw_path = os.path.join('bias', args.dataset, args.split, bw_metadata)
		with open(bw_path, 'r') as f:
			bw_list = json.load(f)
		if path.split('/')[-1] in bw_list:
			im = im.convert('L')
	im.thumbnail((360, 360), Image.ANTIALIAS)
	return OffsetImage(im, zoom=zoom)


def get_visualization(features, img_names, args):
	tsne = manifold.TSNE(n_components=2, n_jobs=-1)
	features_embedded = tsne.fit_transform(features)
		
	if args.dataset == 'objectnet':
		# Get category from file path
		group = [file_path.split('/')[-2] for file_path in img_names]	
		cmap = {}
		unique_group = list(set(group))
		for i, category in enumerate(unique_group):
			cmap[category] = list(colors._colors_full_map.values())[i]
		group_colors = [cmap[category] for category in group]	
	else:
		group_colors = []

	fig, ax = plt.subplots()
	ax.scatter(features_embedded[:, 0], features_embedded[:, 1], s=10, c=group_colors)	
	
	metadata = core.get_dataset_metadata(args.split, args)	
	for x, y, name in zip(features_embedded[:, 0], features_embedded[:, 1], img_names):
		if args.bias != 0:
			ab = AnnotationBbox(get_image(os.path.join(metadata['img_dir'], name), args, bw_metadata=metadata['bw_metadata']), (x,y), frameon=False)
		else:
			ab = AnnotationBbox(get_image(os.path.join(metadata['img_dir'], name), args), (x,y), frameon=False)
		ax.add_artist(ab)
		
	plt.savefig('tsne.png', dpi=1300)
	np.savetxt('tsne.txt', np.array(features_embedded))
	return features_embedded


def preprocess_features(features, components=128):
	# Make features look like standard normal 0 mean, unit variance
	features = StandardScaler().fit_transform(features)
	# Perform PCA to reduce feature size
	pca = PCA(n_components=components)
	features = pca.fit_transform(features)
	print('Cumulative explained variation for PCA: {}'.format(np.sum(pca.explained_variance_ratio_)))	
	return features


def get_optimal_clusters(features, args, file_name):
	k_vals = [5, 10, 15, 20, 24, 25, 30, 32, 35]
	silhouette_scores, ch_scores, db_scores = {}, {}, {}
	for k in k_vals:
		print("K: %d" % k)
		if args.clustering == 'kmeans':
			cluster_technique = cluster.MiniBatchKMeans(n_clusters=k)
		elif args.clustering == 'dbscan':
			cluster_technique = cluster.DBSCAN(min_samples=k, n_jobs=-1)
		elif args.clustering == 'optics':
			cluster_technique = cluster.OPTICS(min_samples=k, n_jobs=-1)
		cluster_technique.fit(features)
		assignments = cluster_technique.labels_
		silhouette_scores[k] = silhouette_score(features, assignments)
		#ch_scores[k] = calinski_harabasz_score(features, assignments)
		db_scores[k] = davies_bouldin_score(features, assignments)
	
	sx, sy = zip(*silhouette_scores.items())
	#cx, cy = zip(*ch_scores.items())
	dx, dy = zip(*db_scores.items())

	sline, = plt.plot(sx, sy, color='green')
	#cline, = plt.plot(cx, cy, color='blue')
	dline, = plt.plot(dx, dy, color='red')
	plt.legend([sline, dline], ['Silhouette', 'DB Index'])
	plt.title('%s' % args.clustering)
	plt.ylim(0, 2.5)

	plt.savefig(os.path.join(args.cluster_dir, '%s_scores.png' % file_name))
	
	return int(statistics.mean([sx[sy.index(max(sy))], dx[dy.index(min(dy))]]))


if __name__ == '__main__':
	main()
