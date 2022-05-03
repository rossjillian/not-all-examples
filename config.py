import argparse
import os


DATA_ROOT = os.getenv('DATA_ROOT')


def get_parser():
	parser = argparse.ArgumentParser()
	# Model Args
	parser.add_argument(
		'--model_name',
		type=str,
		default='rotnet',
		help='Name of model to train: rotnet, patchnet'
	)
	parser.add_argument(
		'--pretrained', 
		type=str, 
		default=None
	)
	parser.add_argument(
		'--debug',
		action='store_true',
		help='Overfit model on small dataset'
	)
	# Hyperparameter Args
	parser.add_argument(
		'--epochs',
		type=int,
		default=100,
		help='Number of total epochs to run'
	)
	parser.add_argument(
		'--batch_size',
		type=int,
		default=64,
		help='Batch size'
	)
	parser.add_argument(
		'--lr',
		'--learning_rate',
		type=float,
		default=0.0005,
		help='Initial learning rate'
	)
	parser.add_argument(
		'--seed',
		type=int,
		default=42
	)
	# Training Args
	parser.add_argument(
		'--start_epoch',
		type=int,
		default=0,
		help='Epoch to start training from'
	)
	parser.add_argument(
		'--rank',
		type=int, 
		default=0,
		help='Node rank for distributed training'
	)
	parser.add_argument(
		'--gpu',
		type=int,
		default=None,
		help='GPU ID'
	)
	parser.add_argument(
		'--num_workers',
		type=int,
		default=12,
		help='Number of dataloading workers'
	)
	parser.add_argument(
		'--ddp',
		action='store_true',
		help='Use multi-processing distributed training'
	)
	parser.add_argument(
		'--master_addr',
		type=str,
		default='localhost'
	)
	parser.add_argument(
		'--master_port',
		type=str,
		default='29500'
	)
	# Data Args
	parser.add_argument(
		'--data_root',
		type=str, 
		default=None, 
		help='Path to data_root containing dataset'
	)
	parser.add_argument(
		'--dataset',
		type=str,
		default='cc',
		help='Dataset name: cc, objectnet'
	)
	parser.add_argument(
		'--split',
		type=str,
		default='train',
		help='Dataset split: train, val, test'
	)
	parser.add_argument(
		'--weights_dir',
		type=str,
		default='weights'
	)
	parser.add_argument(
		'--log_dir',
		type=str,
		default='logs'
	)
	parser.add_argument(
		'--cluster_dir',
		type=str,
		default='clusters'
	)
	# Features Args
	parser.add_argument(
		'--features_only',
		action='store_true',
		help='Get features without self-supervised transforms'
	)
	parser.add_argument(
		'--feature_dir',
		type=str,
		default='features'
	)
	parser.add_argument(
		'--track_features',
		action='store_true',
		help='Write features to file'	
	)
	# Preprocessing Args (preprocessing.py)
	parser.add_argument(
		'--generate_color_list',
		action='store_true',
		help='Get list of all-color images'
	)
	parser.add_argument(
		'--bias',
		type=float, 
		default=0,
		help='Split of bias: 0 (no bias) to 1 (all bias)'
	)
	parser.add_argument(
		'--num_biased',
		type=int,
		default=0,
		help='How many categories to bias'
	)
	parser.add_argument(
		'--biased_category',
		type=str,
		default=None,
		help='Comma separated list of categories (clusters) to assign bias'
	)
	# Clustering Args (clustering.py)
	parser.add_argument(
		'--visualize',
		action='store_true',
		help='Visualize (TSNE, RDMs)'
	)
	parser.add_argument(
		'--view_subset',
		action='store_true',
		help='Visualize with subset of data'
	)
	parser.add_argument(
		'--clustering',
		type=str,	
		default='kmeans',
		help='Type of clustering to perform: kmeans'
	)
	parser.add_argument(
		'--preprocess',
		action='store_true',
		help='Preprocess data using PCA'
	)
	parser.add_argument(
		'--k',
		type=int,
		default=10,
		help='Number of clusters for KMeans'
	)
	parser.add_argument(
		'--optimize_clustering',
		action='store_true',
		help='Optimize clustering'
	)
	# Downstream
	parser.add_argument(
		'--num_classes',
		type=int,
		help='Number of categories for downstream'
	)
	# Analysis Args (analysis.py)
	parser.add_argument(
		'--train_pca',
		action='store_true',
		help='Train and store PCA'
	)
	parser.add_argument(
		'--model_bias',
		type=float,
		help='Bias that model was trained with'
	)
	parser.add_argument(
		'--data_bias',
		type=float,
		help='Bias in analysis dataset'
	)
	parser.add_argument(
		'--compute_rdm',
		action='store_true',
		help='Compute RDM'
	)
	parser.add_argument(
		'--repeat',
		type=int,
		default=0,
		help='Number of times to repeat analysis for confidence intervals'
	)
	parser.add_argument(
		'--model_layers',
		type=str,
		help='Comma separated list of idx of layers to extract features from'
	)
	parser.add_argument(
		'--categories',
		type=str,
		default=None
	)
	parser.add_argument(
		'--base_features',
		type=str,
		help='[bias]_[number biased]'
	)
	parser.add_argument(
		'--rdm_dir',
		type=str,
		default='rdm'
	)
	return parser


def parse_args():
	parser = get_parser()
	args = parser.parse_args()
	if args.data_root is None:
		args.data_root = DATA_ROOT

	return args

