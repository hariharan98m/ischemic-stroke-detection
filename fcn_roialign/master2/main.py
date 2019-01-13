import torch
from pathlib import Path
import matplotlib.pyplot as plt

from vgg import VGGNet
from enhanced_vgg import EnhancedVGGNet, EnhancedBaseModel
from single_scale_roi_align import Unet
from fcn8s import FCN8s
from utils import expand_mask, resize, show_single_img, get_prob_map28, dice_loss, actual_predicted
from dataloader import SISSDataset, Rescale, RandomCrop, RandomRotate, ToMultiFloatMaskValues, Normalize, ToClassifierTuple, FinalRoIAlignExperiment, SingleScaleRoIAlignExperiment
from torch.utils.data._utils.collate import default_collate

import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np
from torchvision import transforms, models, datasets
import torch.nn.functional as F
from train import attention, attention_field_stripping, single_scale_roi_align_experiment, final_roi_align
import pdb

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("exp")
args = parser.parse_args()
exp = args.exp

_class    = 2

train_scans = 1
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


def my_collate(batch):
    batch = list(filter(lambda x: (x is not None and x.size()[0] == 10), batch))
    try:
        collated = default_collate(batch)
        return collated
    except:
        # for i in batch:
        #     cut_boxes, labels, scans, (base, scale1, scale2, scale3) = i
        #     print(i, ': ', cut_boxes.shape, labels.shape, scans.shape,
        #           (base.shape, scale1.shape, scale2.shape, scale3.shape))
        pass


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
train_dataset = SISSDataset(num_slices = 64,
                            num_scans= train_scans,
                            root_dir = Path.cwd().parents[0],
                            transform = composed)

# val dataset
val_dataset = SISSDataset(num_slices = 64,
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


if exp == 'attention':
    enhanced_base = EnhancedVGGNet(freeze_max=False)
    exp1 = FCN8s('attention', enhanced_base)

    all_trainable_layers = [param for param in exp1.parameters() if param.requires_grad ]
    optimizer = optim.Adam(all_trainable_layers, lr=lr, weight_decay=w_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)  # decay LR by a factor of 0.5 every 30 epochs

    print('Configs:')
    print('Attention Exp-  Learning Rate %f, Number of epochs: %d, Batch size: %d' %
          (lr, epochs, batch_size)
    )

    attention(
        exp1,
        optimizer,
        scheduler,
        dataloaders,
        dataset_sizes,
        num_epochs=25
    )

if exp == 'attention_rf_stripping':
    enhanced_vgg = EnhancedVGGNet(freeze_max=False)
    exp2 = FCN8s('attention_rf_stripping', enhanced_vgg)
    upsampler = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

    all_trainable_layers = [param for param in exp2.parameters() if param.requires_grad ]
    optimizer = optim.Adam(all_trainable_layers, lr=lr, weight_decay=w_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)  # decay LR by a factor of 0.5 every 30 epochs

    print('Configs:')
    print('attention_rf_stripping-  Learning Rate %f, Number of epochs: %d, Batch size: %d' %
          (lr, epochs, batch_size)
    )

    attention_field_stripping(
        exp2,
        optimizer,
        scheduler,
        dataloaders,
        dataset_sizes,
        num_epochs=25
    )


if exp == 'single_scale_roi_align':

    backbone = EnhancedVGGNet(freeze_max=True)
    baseline = FCN8s('single_scale_roi_align', backbone)
    clf  = Unet('single_scale_roi_align', baseline)

    clf = clf.to(device)

    optimizer = optim.Adam(clf.parameters(), lr=lr, weight_decay=w_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)  # decay LR by a factor of 0.5 every 30 epochs

    print('Configs:')
    print('Single-scale RoI Align Experiment-  Learning Rate %f, Number of epochs: %d, Batch size: %d' %
          (lr, epochs, batch_size)
    )

    single_scale_roi = SingleScaleRoIAlignExperiment()
    single_scale_roi_transform = transforms.Compose([
        scale,
        rotate,
        crop,
        norm,
        single_scale_roi
    ])

    # train dataset
    single_scale_roi_dataset_train = SISSDataset(
        num_slices=153,
        num_scans=2,
        root_dir=Path.cwd().parents[0],
        transform=single_scale_roi_transform,
        train=True
    )

    # val dataset
    single_scale_roi_dataset_val = SISSDataset(
            num_slices=153,
            num_scans=val_scans,
            root_dir=Path.cwd().parents[0],
            transform=single_scale_roi_transform,
            train=False
    )

    datasets = {
        'train': single_scale_roi_dataset_train,
        'val': single_scale_roi_dataset_val
    }

    train_dataloader = torch.utils.data.DataLoader(datasets['train'], 16, True, collate_fn=my_collate)
    val_dataloader = torch.utils.data.DataLoader(datasets['val'], 16, True, collate_fn=my_collate)

    clf_dataloaders = {
        'train': train_dataloader,
        'val': val_dataloader
    }

    clf_dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}
    class_names = ['no lesion', 'lesion']


    single_scale_roi_align_experiment(
        clf,
        optimizer,
        scheduler,
        clf_dataloaders,
        clf_dataset_sizes,
        num_epochs=25
    )



if exp == 'final_roi_align':
    scale = Rescale(int(1.05 * 230))
    crop = RandomCrop(224)
    rotate = RandomRotate(20.0)
    norm = Normalize()

    exp5 = FinalRoIAlignExperiment()
    final_transform = transforms.Compose([scale,
                                          rotate,
                                          crop,
                                          norm,
                                          exp5])

    single_scale_roi_transform = transforms.Compose([
        scale,
        rotate,
        crop,
        norm,
        single_scale_roi
    ])

    final_dataset = SISSDataset(
        num_slices=153,
        num_scans=2,
        root_dir=Path.cwd().parents[0],
        transform=final_transform,
        train=True
    )

    single_scale_roi_dataset = SISSDataset(
        num_slices=153,
        num_scans=2,
        root_dir=Path.cwd().parents[0],
        transform=single_scale_roi_transform,
        train=True
    )

    # pdb.set_trace()

    for i in range(50):
        sample = single_scale_roi_dataset[i]
        if sample is not None:
            scans, label224 = sample
            print(scans.dtype, label224.dtype)
            print(scans.size(), label224.size())
            # print(base, '\n\n')
        else:
            print(None)



    for i, batch in enumerate(dataloader):
        cut_boxes, labels, scans, (base, scale1, scale2, scale3) = batch
        # split_boxes_list = torch.split(batch, split_size_or_sections=1, dim=0)

        print(i, ': ', cut_boxes.shape, labels.shape, scans.shape,
              (base.shape, scale1.shape, scale2.shape, scale3.shape))




#
#
#
# scan_samples, label_samples = [], []
#
# for box in patch_boxes.astype(int):
#     x1, y1, x2, y2 = box.tolist()
#     scan_samples.append(scans[:, x1: x2, y1:y2])
#     label_samples.append(label224[:, x1: x2, y1:y2])
#
# scan_samples, label_samples = np.stack(scan_samples, axis=0), np.stack(label_samples, axis=0)
#
# scans, label224 = torch.from_numpy(scan_samples), torch.from_numpy(label_samples)
#
# return scans, label224
#
#
#
#
#
# if exp == 'exp3':
#     enhanced_vgg = EnhancedVGGNet(freeze_max=False)
#     exp3 = Unet('exp3', enhanced_vgg)
#
#     all_trainable_layers = [param for param in exp3.parameters() if param.requires_grad ]
#     optimizer = optim.Adam(all_trainable_layers, lr=lr, weight_decay=w_decay)
#     scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)  # decay LR by a factor of 0.5 every 30 epochs
#
#     print('Configs:')
#     print('Exp3-  Learning Rate %f, Number of epochs: %d, Batch size: %d' %
#           (lr, epochs, batch_size)
#     )
#
#     experiment3(
#         exp3,
#         optimizer,
#         scheduler,
#         dataloaders,
#         dataset_sizes,
#         num_epochs=25
#     )
