import torch
from pathlib import Path
import matplotlib.pyplot as plt

from vgg import VGGNet
from fcn8s import FCN8s
from dataloader import SISSDataset, ToTupleTensor, Rescale, RandomRotate, RandomCrop, Normalize, ToRoIAlignTensor

import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np
from torchvision import transforms, utils, models, datasets

from train import train_model, dice_loss, show_single_img, model_out_to_unmold

import pdb

# pdb.set_trace()

def viz_prediction(track_sample, pred, epoch, dice):
    scans, label = track_sample

    scans, label = scans.numpy().transpose((1, 2, 0)), label.numpy()[0][..., np.newaxis]
    pred = pred[0].numpy()[..., np.newaxis]

    scans_stack = np.concatenate([scans, label, pred], axis=-1)

    fig = plt.figure(figsize=(20, 6))

    fig.suptitle('Dice:'+ str(dice))

    for slice_, scan in enumerate(['dwi', 'flair', 't1', 't2', 'label', 'predicted']):
        ax = plt.subplot(1, 6, slice_ + 1)
        show_single_img(scans_stack[:, :, slice_], (scan == 'label' or scan == 'predicted'))
        plt.tight_layout()
        ax.set_title(scan)
        ax.axis('off')

    plt.show()
    # pdb.set_trace()
    # plt.savefig('testing/'+ str(epoch)+ '.jpg')


vgg_model = VGGNet(freeze_max = False)
net = FCN8s(vgg_model)

checkpoint = torch.load('baseline.pth')

net.load_state_dict(checkpoint['model_state_dict'])

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

net.to(device)

# train dataloader
scale = Rescale(int(1.5 * 230))
crop = RandomCrop(224)
rotate = RandomRotate(20.0)
norm = Normalize()
tupled = ToTupleTensor()
tupled_with_roi_align = ToRoIAlignTensor()

composed_for_tracking = transforms.Compose([
    Rescale(224),
    norm,
    tupled_with_roi_align
])

dataset = SISSDataset(num_slices = 153,
                           num_scans= 2,
                           root_dir = Path.cwd().parents[0],
                           transform = composed_for_tracking,
                           train = True
                          )

dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                             shuffle=True, num_workers=4)

device = torch.device('cuda:0')
times = 0
for mini_batch, (inputs, labels) in enumerate(dataloader):
    inputs = inputs.to(device)
    labels = labels.to(device)

    # forward
    # track history if only in train
    with torch.set_grad_enabled(False):
        outputs28 = net(inputs)

        torch_preds224 = model_out_to_unmold(outputs28).to(device)

        dice = dice_loss(input=labels, target=torch_preds224).item()

        print('Step %d:  Dice Loss: %f'% (mini_batch, dice))

    inputs, labels = inputs.cpu()[0], labels.cpu()[0]

    viz_prediction((inputs, labels), torch_preds224[0].cpu(), epoch=mini_batch, dice = dice)

    break
