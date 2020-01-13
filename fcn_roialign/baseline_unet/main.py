import torch
from pathlib import Path
import matplotlib.pyplot as plt

from vgg import VGGNet
from fcn8s import FCN8s
from dataloader import SISSDataset, ToTupleTensor, Rescale, RandomRotate, RandomCrop, Normalize, ToMultiScaleMasks, ToRoIAlignTensor

import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np
from torchvision import transforms, utils, models, datasets

from train import train_model

import pdb

input = torch.randn(32, 4, 224, 224)

vgg_model = VGGNet(freeze_max = False)
net = FCN8s(vgg_model)

all_trainable_layers = [param for param in net.parameters() if param.requires_grad ]

pdb.set_trace()

_class    = 2
train_scans = 2
batch_size = 16
epochs     = 10
lr         = 1e-6
momentum   = 0
w_decay    = 1e-5
step_size  = 50
gamma      = 0.5
configs    = "FCNs-Cross Entropy Loss _batch{}_epoch{}_RMSprop_scheduler-step{}-gamma{}_lr{}_momentum{}_w_decay{}".format(batch_size, epochs, step_size, gamma, lr, momentum, w_decay)
print("Configs:", configs)

class_weights = torch.tensor([0.1, 0.9])
criterion = nn.CrossEntropyLoss(weight = class_weights)
optimizer = optim.Adam(all_trainable_layers, lr=lr, weight_decay=w_decay)
scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)  # decay LR by a factor of 0.5 every 30 epochs


siss_dataset = SISSDataset(num_slices=154,
                           num_scans=3,
                           root_dir=Path.cwd().parents[0])

idx = 91
sample = siss_dataset[idx]

def show_single_img(image, label):
    """Show image"""
    cmap = 'gray'
    if label:
        cmap = 'binary'
    plt.imshow(image, cmap = cmap)


def viz_sample(sample):
    fig = plt.figure(figsize=(20, 5))
    fig.suptitle('Sample %d' % idx)

    for slice_, scan in enumerate(['', 'flair', 't1', 't2', 'label']):
        ax = plt.subplot(1, 5, slice_ + 1)
        show_single_img(sample[:, :, slice_], scan == 'label')
        # plt.tight_layout()
        ax.set_title(scan)
        ax.axis('off')

    plt.show()

# train dataloader
scale = Rescale(int(1.5 * 230))
crop = RandomCrop(224)
rotate = RandomRotate(20.0)
norm = Normalize()
tupled = ToTupleTensor()
tupled_multiscaled_masks = ToMultiScaleMasks()
tupled_with_roialign = ToRoIAlignTensor()

composed = transforms.Compose([scale,
                               rotate,
                               crop,
                               norm,
                               tupled_multiscaled_masks])

# transforms coupling testing
composed_wo_tupling_norm = transforms.Compose([Rescale(int(1.5 * 230)),
                               RandomRotate(50.0),
                               RandomCrop(224),
                               Normalize()
                               ])

composed_for_tracking = transforms.Compose([
    Rescale(224),
    norm,
    tupled_with_roialign
])

composed_for_valset = transforms.Compose([
    Rescale(224),
    norm,
    tupled_multiscaled_masks
])


siss = SISSDataset(num_slices = 153,
            num_scans= 2,
            root_dir = Path.cwd().parents[0],
            transform = composed_for_tracking)

# train dataset
train_dataset = SISSDataset(num_slices = 153,
                            num_scans= train_scans,
                            root_dir = Path.cwd().parents[0],
                            transform = composed)

# val dataset
val_dataset = SISSDataset(num_slices = 153,
                           num_scans= 1,
                           root_dir = Path.cwd().parents[0],
                           transform = composed_for_valset,
                           train = False
                          )

datasets = {
    'train': train_dataset,
    'val': val_dataset
}

# pdb.set_trace()

dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=batch_size,
                                             shuffle=True, num_workers=4)
                for x in ['train', 'val']}

dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}
class_names = datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# pdb.set_trace()
track_sample = siss[91]
train_model(net.to(device), optimizer, scheduler, dataloaders,
                       dataset_sizes, track_sample, batch_size, num_epochs=epochs)
