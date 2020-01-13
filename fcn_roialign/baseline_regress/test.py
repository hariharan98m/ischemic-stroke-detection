import torch
from pathlib import Path
import matplotlib.pyplot as plt

from vgg import VGGNet
from fcn8s import FCN8s
from dataloader import SISSDataset, ToTupleTensor, Rescale, RandomRotate, RandomCrop, Normalize, ToRoIAlignTensor, expand_mask, ToMultiScaleMasks

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np
from torchvision import transforms, utils, models, datasets

from train import train_model, dice_loss, show_single_img, model_out_to_unmold

import pdb

# pdb.set_trace()

def viz_prediction(track_sample, pred_argmax, pred_prob, epoch, stats):
    scans, label = track_sample

    scans, label = scans.numpy().transpose((1, 2, 0)), label.numpy()[0][..., np.newaxis]
    pred_prob, pred_argmax = pred_prob.numpy().transpose((1,2,0)), pred_argmax.numpy().transpose((1,2,0))

    scans_stack = np.concatenate([scans, label, pred_argmax, pred_prob], axis=-1)

    fig = plt.figure(figsize=(20, 6))

    fig.suptitle(stats)

    for slice_, scan in enumerate(['dwi', 'flair', 't1', 't2', 'label', 'predicted-argmax', 'predicted-prob']):
        ax = plt.subplot(1, 7, slice_ + 1)
        show_single_img(scans_stack[:, :, slice_], label=False)
        # plt.tight_layout()
        ax.set_title(scan)
        ax.axis('off')

    # plt.show()
    # pdb.set_trace()
    plt.savefig('testing/'+ str(epoch)+ '.jpg')


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
tupled_multiscales = ToMultiScaleMasks()

composed_for_tracking = transforms.Compose([
    Rescale(224),
    norm,
    tupled_multiscales
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

# pdb.set_trace()

for mini_batch, (inputs, label224, label28) in enumerate(dataloader):
    inputs = inputs.to(device)
    label28 = label28.to(device)

    # forward
    # track history if only in train
    with torch.set_grad_enabled(False):

        outputs28 = net(inputs) # shape of pred28 is (batch_size, 2, 28, 28)
        pred28 = torch.exp(outputs28)

        # based on argmax
        max_prob, pred28_argmax = torch.max(pred28, dim=1, keepdim=True)  # (batch_size, 1, 28,28)

        # based on prob
        pred28[:, 0, :, :] = 1- pred28[:, 0, :, :]
        one_hot = F.one_hot(pred28_argmax[:, 0, :, :]).permute(0, 3, 1, 2).bool() # (batch_size, 2 classes, 28,28)
        pred28_prob = torch.sum(pred28 * one_hot, dim=1, keepdim=True) #(batch_size, 1 val, 28, 28)

        torch_pred224_argmax, torch_pred224_prob = model_out_to_unmold(pred28_argmax.float()), model_out_to_unmold(pred28_prob)
                    # return format is (batch_size, 1, 224, 224)
        argmax_dice = dice_loss(input= torch_pred224_argmax, target= label224).item()
        prob_dice = dice_loss(input= torch_pred224_prob, target= label224).item()

        stats = 'Step %d:  Argmax DiceLoss: %.4f  Prob DiceLoss: %.4f'% (mini_batch+1, argmax_dice, prob_dice)
        print(stats)

    inputs, labels = inputs.cpu()[0], label224.cpu()[0]

    viz_prediction((inputs, labels), torch_pred224_argmax[0], torch_pred224_prob[0], epoch=mini_batch, stats = stats)

    times+=1
    if times==100:
        break
