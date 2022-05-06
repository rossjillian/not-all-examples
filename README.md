# Not All Examples Are Created Equal

Shell scripts are provided to replicate the results of the paper.

## Preprocessing

We create files that delineate which examples are augmented. These files ensure consistency: every model sees the same examples augmented.

An example command is: 

`python3 -m preprocessing --clustering=kmeans --dataset=coco --aug=0.1 --num_aug=1 --k=5 --split=train --aug_category=0`

The command denotes which dataset to augment (coco), which clustering to use to assign augmentation (k-means), the cluster to augment (0), and the percentage increments to assign augmentation (0.1 = 10%, so the command generates files with 10%, 20%, 30%, ..., 100% of examples augmented).

## Pre-Training

We train separate models on each augmentation increment (10%, 20%, 30%, ..., 100%). 

An example command is:

`python3 -m main --epochs=10 --master_port=20002 --model_name=rotnet --dataset=coco --clustering=kmeans --aug=0.2 --num_aug=1 --k=5 --aug_transform=bw --aug_category=0 --ddp`

The command denotes which model to train (rotation network), which dataset to train on (coco), what percentage of the augmented cluster (0) should be augmented (20%), and which augmentation to perform (grayscale). 

## Downstream Training

We use the pre-trained model for a downstream classification task on CIFAR-100.

An example command is:

`python3 -m main --master_port=24002 --pretrained=./weights/rotnet/coco/02_1/5/c0/bw/5_weights-0.pt --model_name=rotnetdown --dataset=cifar100 --num_classes=100 --clustering=kmeans --aug=0.2 --num_aug=1 --k=5 --aug_transform=bw --aug_category=0 --ddp`

The command denotes the location of the pretrained model and the properties of the pretrained model (20% augmented with the grayscale transform), which model to train (downstream rotation network), which dataset to train on (cifar100), the classification task (100-way classification).

## Evaluating Models

We evaluate the model on an unaugmented test set. The pre-trained model is directly evaluated on the unaugmented COCO test set; the downstream model is directly evaluted on the unaugmented CIFAR100 test set.

An example command is:

`python3 -m main --master_port=37002 --pretrained=./weights/rotnet/coco/02_1/5/c0/bw/5_weights-0.pt --model_name=rotnet --dataset=coco --clustering=kmeans --aug=0.0 --num_aug=0 --k=5 --aug_transform=bw --aug_category=0 --debug --ddp`

The command denotes the location of the pretrained model, which model to test (rotation network), which dataset to test on (coco). The `--debug` flag denotes testing mode.
