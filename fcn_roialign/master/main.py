import torch
from pathlib import Path
import matplotlib.pyplot as plt

from vgg import VGGNet
from enhanced_vgg import EnhancedVGGNet, EnhancedBackbone
from unet import Unet
from fcn8s import FCN8s
from utils import expand_mask, resize, show_single_img, get_prob_map28, dice_loss, actual_predicted
from dataloader import SISSDataset, Rescale, RandomCrop, RandomRotate, ToMultiFloatMaskValues, Normalize, ToClassifierTuple, FinalRoIAlignExperiment

import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np
from torchvision import transforms, models, datasets
import torch.nn.functional as F
from train import experiment1, experiment2, experiment3, classifier_experiment
from enhanced_vgg import StartBlock, IntermediateBlock1
from unet import Unet
import pdb

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("exp")
args = parser.parse_args()
exp = args.exp

_class    = 2

train_scans = 2
val_scans = 1
test_scans = 6

batch_size = 32
epochs     = 10
lr         = 1e-5
momentum   = 0
w_decay    = 1e-5
step_size  = 18
gamma      = 0.5

class_weights = torch.tensor([0.1, 0.9])

def show_single_img(image, label):
    """Show image"""
    cmap = 'gray'
    if label:
        cmap = 'binary'
    plt.imshow(image, cmap = cmap)


def viz_sample(sample):
    fig = plt.figure(figsize=(20, 5))
    fig.suptitle('Sample Tracking')

    for slice_, scan in enumerate(['dwi', 'flair', 't1', 't2', 'label']):
        ax = plt.subplot(1, 5, slice_ + 1)
        show_single_img(sample[:, :, slice_], scan == 'label')
        plt.tight_layout()
        ax.set_title(scan)
        ax.axis('off')

    plt.show()

# train dataloader
scale = Rescale(int(1.5 * 230))
crop = RandomCrop(224)
rotate = RandomRotate(20.0)
norm = Normalize()
tupled_float_regress_masks = ToMultiFloatMaskValues()
tupled_classifier = ToClassifierTuple()

composed = transforms.Compose([scale,
                               rotate,
                               crop,
                               norm,
                               tupled_float_regress_masks])

composed_for_valset = transforms.Compose([
    Rescale(224),
    norm,
    tupled_float_regress_masks
])


siss = SISSDataset(num_slices = 153,
            num_scans= 2,
            root_dir = Path.cwd().parents[0],
            transform = composed_for_valset
       )

# train dataset
train_dataset = SISSDataset(num_slices = 153,
                            num_scans= train_scans,
                            root_dir = Path.cwd().parents[0],
                            transform = composed)

# val dataset
val_dataset = SISSDataset(num_slices = 153,
                           num_scans= val_scans,
                           root_dir = Path.cwd().parents[0],
                           transform = composed_for_valset,
                           train = False
                          )

datasets = {
    'train': train_dataset,
    'val': val_dataset
}

# pdb.set_trace()

train_dataloader = torch.utils.data.DataLoader(datasets['train'], batch_size=10, shuffle=True, num_workers=10)
val_dataloader = torch.utils.data.DataLoader(datasets['val'], batch_size=10, shuffle=True, num_workers=10)

dataloaders = {
    'train': train_dataloader,
    'val': val_dataloader
}

dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}
class_names = datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if exp == 'exp1':
    enhanced_vgg = EnhancedVGGNet(freeze_max=False)
    exp1 = FCN8s('exp1', enhanced_vgg)

    all_trainable_layers = [param for param in exp1.parameters() if param.requires_grad ]
    optimizer = optim.Adam(all_trainable_layers, lr=lr, weight_decay=w_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)  # decay LR by a factor of 0.5 every 30 epochs

    print('Configs:')
    print('Exp1-  Learning Rate %f, Number of epochs: %d, Batch size: %d' %
          (lr, epochs, batch_size)
    )

    experiment1(
        exp1,
        optimizer,
        scheduler,
        dataloaders,
        dataset_sizes,
        num_epochs=25
    )

if exp == 'exp2':
    enhanced_vgg = EnhancedVGGNet(freeze_max=False)
    exp2 = FCN8s('exp2', enhanced_vgg)
    upsampler = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

    all_trainable_layers = [param for param in exp2.parameters() if param.requires_grad ]
    optimizer = optim.Adam(all_trainable_layers, lr=lr, weight_decay=w_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)  # decay LR by a factor of 0.5 every 30 epochs

    print('Configs:')
    print('Exp2-  Learning Rate %f, Number of epochs: %d, Batch size: %d' %
          (lr, epochs, batch_size)
    )

    experiment2(
        exp2,
        optimizer,
        scheduler,
        dataloaders,
        dataset_sizes,
        num_epochs=25
    )


if exp == 'exp3':
    enhanced_vgg = EnhancedVGGNet(freeze_max=False)
    exp3 = Unet('exp3', enhanced_vgg)

    all_trainable_layers = [param for param in exp3.parameters() if param.requires_grad ]
    optimizer = optim.Adam(all_trainable_layers, lr=lr, weight_decay=w_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)  # decay LR by a factor of 0.5 every 30 epochs

    print('Configs:')
    print('Exp3-  Learning Rate %f, Number of epochs: %d, Batch size: %d' %
          (lr, epochs, batch_size)
    )

    experiment3(
        exp3,
        optimizer,
        scheduler,
        dataloaders,
        dataset_sizes,
        num_epochs=25
    )


if exp == 'single_scale_roi_align':

    clf = EnhancedBackbone('classifier_model', batch_size = batch_size)

    clf = clf.to(device)

    optimizer = optim.Adam(clf.parameters(), lr=lr, weight_decay=w_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)  # decay LR by a factor of 0.5 every 30 epochs

    print('Configs:')
    print('Classifier Experiment-  Learning Rate %f, Number of epochs: %d, Batch size: %d' %
          (lr, epochs, batch_size)
    )

    composed_classifier = transforms.Compose([scale,
                                              rotate,
                                              crop,
                                              norm,
                                              tupled_classifier])

    composed_classifier_for_valset = transforms.Compose([
        Rescale(224),
        norm,
        tupled_classifier
    ])

    siss = SISSDataset(num_slices=153,
                       num_scans=2,
                       root_dir=Path.cwd().parents[0],
                       transform=composed_classifier_for_valset
                       )

    # train dataset
    train_dataset = SISSDataset(num_slices=153,
                                num_scans=train_scans,
                                root_dir=Path.cwd().parents[0],
                                transform=composed_classifier)

    # val dataset
    val_dataset = SISSDataset(num_slices=153,
                              num_scans=val_scans,
                              root_dir=Path.cwd().parents[0],
                              transform=composed_classifier_for_valset,
                              train=False
                              )

    datasets = {
        'train': train_dataset,
        'val': val_dataset
    }

    # pdb.set_trace()

    train_dataloader = torch.utils.data.DataLoader(datasets['train'], batch_size=batch_size, shuffle=True,
                                                   num_workers=10)
    val_dataloader = torch.utils.data.DataLoader(datasets['val'], batch_size=10, shuffle=True, num_workers=10)

    clf_dataloaders = {
        'train': train_dataloader,
        'val': val_dataloader
    }

    clf_dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}
    class_names = ['no lesion', 'lesion']


    classifier_experiment(
        clf,
        optimizer,
        scheduler,
        clf_dataloaders,
        clf_dataset_sizes,
        num_epochs=25
    )
