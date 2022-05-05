import argparse
import config as cfg
import core
import torch
import torch.nn as nn
from torch import optim, utils
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
from torchvision import transforms
import statistics
import os
import json
from tqdm import tqdm


def main():
	args = cfg.parse_args()
	
	# TODO: seeding

	if args.gpu is not None:
		print("Specific GPU chosen. No data parallelism.")
	ngpus = torch.cuda.device_count()
	if args.ddp:
		# TODO: move out of args
		os.environ['MASTER_ADDR'] = args.master_addr
		os.environ['MASTER_PORT'] = args.master_port
		# Number of processes spawned is equal to number of GPUs available
		print("Spawning...")
		mp.spawn(main_worker, nprocs=ngpus, args=(ngpus, args))
	else:
		main_worker(args.gpu, ngpus, args)


def main_worker(gpu, ngpus, args):
	args.gpu = gpu
	weights_path = os.path.join(args.weights_dir, args.model_name, args.dataset, str(args.aug).replace('.', '') + '_' + str(args.num_aug), str(args.k), 'c'+str(args.aug_category), args.aug_transform)
	if not os.path.exists(weights_path):
		os.makedirs(weights_path, exist_ok=True)
	log_path = os.path.join(args.log_dir, args.model_name)
	if not os.path.exists(log_path):
		os.makedirs(log_path, exist_ok=True)
	feature_path = os.path.join(args.feature_dir, args.model_name)
	if not os.path.exists(feature_path):
		os.makedirs(feature_path, exist_ok=True)
	
	if args.ddp:
		print("Spawned!")
		# Global rank
		args.rank = args.rank * ngpus + gpu
		# Use NCCL for GPU training, process must have exclusive access to GPUs
		torch.distributed.init_process_group(
			backend='nccl', 
			init_method='env://', 
			rank=args.rank, 
			world_size=ngpus
		)

	model = core.get_model(args)
	
	if args.ddp:
		if args.gpu is not None:
			torch.cuda.set_device(args.gpu)
			model.cuda(args.gpu)
			args.batch_size = int(args.batch_size / ngpus)
			args.num_workers = int((args.num_workers + ngpus - 1) / ngpus)
			if args.model_name == 'rotnetdown':
				model = nn.parallel.DistributedDataParallel(
					model, device_ids=[args.gpu], find_unused_parameters=True
				)
			else:
				model = nn.parallel.DistributedDataParallel(
					model, device_ids=[args.gpu]
				)
		else:
			model.cuda()
			model = nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
	elif args.gpu is not None:
		torch.cuda.set_device(args.gpu)
		model = model.cuda(args.gpu)
	else:
		# DataParallel will divide and allocate batch_size to all available GPUs
		model = nn.DataParallel(model).cuda()

	if args.pretrained:
		if args.model_name != 'rotnetdown':
			model.load_state_dict(torch.load(args.pretrained))
		elif args.model_name == 'rotnetdown' and 'rotnetdown' in args.pretrained:
			print("Load entire model")
			model.load_state_dict(torch.load(args.pretrained))

	train_dataset = core.get_dataset(args.data_root, 'train', args.model_name, args)
	val_dataset = core.get_dataset(args.data_root, 'val', args.model_name, args)
	if args.debug:
		test_dataset = core.get_dataset(args.data_root, 'test', args.model_name, args)

	train_loader = core.get_dataloader(train_dataset, args.batch_size, args.num_workers, args, ngpus=ngpus, gpu=gpu)
	val_loader = core.get_dataloader(val_dataset, args.batch_size, args.num_workers, args, ngpus=ngpus, gpu=gpu)
	if args.debug:
		test_loader = core.get_dataloader(test_dataset, args.batch_size, args.num_workers, args, ngpus=ngpus, gpu=gpu)

	print("Dataset and loader instantiated")

	if args.model_name == 'rotnetdown':
		for i in range(len(model.module.modified_rotnet._feature_blocks)-1):
			for param in model.module.modified_rotnet._feature_blocks[i].parameters():
				param.requires_grad = False

	optimizer = core.get_optimizer(model.parameters(), args.lr, args)

	criterion = core.get_loss(args)

	# Note scheduler not currently used
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                           mode='min',
                                           patience=5,
                                           factor=0.3, verbose=True)

	best_acc = 0
	if is_rank0(args, ngpus):
		writer = SummaryWriter(log_dir='./runs/%s_%s_b%s_n%d_k%d_c%s_%s-%d' % (args.model_name, args.dataset, str(args.aug).replace('.', ''), args.num_aug, args.k, args.aug_category, args.aug_transform, args.repeat))

	if args.debug:
		epoch = 0
		total, correct, test_running_loss = validate(test_loader, model, criterion, args, epoch, ngpus, feature_path)
	
		test_acc = correct / total
		print(test_acc)
	else:
		if is_rank0(args, ngpus):
			torch.save(model.state_dict(), os.path.join(weights_path, '%s_init-%d.pt' % (args.dataset, args.repeat)))
		
		for epoch in range(args.start_epoch, args.epochs):
			total, correct, train_running_loss = train(train_loader, model, criterion, optimizer, args, epoch, ngpus)

			train_acc = correct / total
			if is_rank0(args, ngpus):
				writer.add_scalar('Accuracy/train', train_acc, epoch)
				writer.add_scalar('Loss/train', statistics.mean(train_running_loss), epoch)

			total, correct, val_running_loss = validate(val_loader, model, criterion, args, epoch, ngpus, feature_path)
	
			val_acc = correct / total
			if is_rank0(args, ngpus):
				writer.add_scalar('Accuracy/val', val_acc, epoch)
				writer.add_scalar('Loss/val', statistics.mean(val_running_loss), epoch)
			
			if is_rank0(args, ngpus) and val_acc > best_acc:
				torch.save(model.state_dict(), os.path.join(weights_path, '%s_best-%d.pt' % (args.dataset, args.repeat)))
				best_acc = val_acc

			if epoch % 5 == 0 and is_rank0(args, ngpus):
				torch.save(model.state_dict(), os.path.join(weights_path, '%d_weights-%d.pt' % (epoch, args.repeat)))			

	if args.ddp:
		torch.distributed.destroy_process_group()
	
	return


def train(train_loader, model, criterion, optimizer, args, epoch, ngpus):
	train_running_loss = []

	model.train()
	total, correct = 0, 0
	with tqdm(train_loader, unit='batch') as tepoch:
		for inputs, targets in tepoch:
			tepoch.set_description(f'Epoch %d' % epoch)

			inputs = inputs.cuda(args.gpu, non_blocking=True)
			targets = targets.cuda(args.gpu, non_blocking=True)	
		
			optimizer.zero_grad()
			outputs = model(inputs)		
			loss = criterion(outputs, targets)
				
			loss.backward()
			optimizer.step()
			_, predicted = torch.max(outputs.data, 1)

			total += targets.size(0)
			correct += (predicted == targets).sum()			
			train_running_loss.append(loss.item())

			if is_rank0(args, ngpus):		
				tepoch.set_postfix(loss=loss.item(), accuracy=(correct/total))
	
	return total, correct, train_running_loss


def validate(val_loader, model, criterion, args, epoch, ngpus, feature_path):
	val_running_loss = []
	features = []		

	model.eval()	
	total, correct = 0, 0
	with torch.no_grad():
		for idx, (inputs, targets) in enumerate(val_loader):
			inputs = inputs.cuda(args.gpu, non_blocking=True)
			targets = targets.cuda(args.gpu, non_blocking=True)	
			
			outputs = model(inputs)		
			loss = criterion(outputs, targets)
			val_running_loss.append(loss.item())

			_, predicted = torch.max(outputs.data, 1)
			total += targets.size(0)
			correct += (predicted == targets).sum()			
			
			if args.track_features:
				features.append(outputs)
			
		if args.track_features and epoch % 5 == 0 and is_rank0(args, ngpus):
			torch.save(torch.stack(features), os.path.join(feature_path, 'eval_outputs_%d.pt' % epoch))
	
	return total, correct, val_running_loss


def is_rank0(args, ngpus):
	return not args.ddp or (args.ddp and args.rank % ngpus == 0)


if __name__ == '__main__':
	main()
