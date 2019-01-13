import torch
from pathlib import Path
import matplotlib.pyplot as plt

from vgg import VGGNet
from fcn8s import FCN8s
from dataloader import SISSDataset, ToTupleTensor, Rescale, RandomRotate, RandomCrop, Normalize

import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np
from torchvision import transforms, utils, models, datasets

from train import train_model, dice_loss, show_single_img

import pdb

# pdb.set_trace()

def viz_prediction(track_sample, pred, epoch, dice):
    # pdb.set_trace()
    scans, label = track_sample

    scans, label = scans.numpy().transpose((1, 2, 0)), label.numpy()[..., np.newaxis]
    pred = pred.numpy()[..., np.newaxis]

    scans_stack = np.concatenate([scans, label, pred], axis=-1)

    # fig = plt.figure(figsize=(20, 6))

    # fig.suptitle('dice: ' + str(dice))

    for slice_, scan in enumerate(['dwi', 'flair', 't1', 't2', 'label', 'predicted']):
        # ax = plt.subplot(1, 6, slice_ + 1)
        plt.figure(figsize=(20, 20))
        show_single_img(scans_stack[:, :, slice_], (scan == 'label' or scan == 'predicted'))
        # plt.title('dice: ' + str(dice))
        plt.axis('off')
        Path.mkdir(Path('testing_model1/' + scan), parents=True, exist_ok= True)
        plt.savefig('testing_model1/' + scan + '/' + str(epoch)+ '_'+ str(dice) + '.jpg')

    # plt.show()
    # pdb.set_trace()
    # plt.savefig('testing_to_publish/'+ str(epoch)+ '.jpg')


vgg_model = VGGNet(freeze_max = False)
net = FCN8s(vgg_model)

checkpoint = torch.load('save_objs.pth')

net.load_state_dict(checkpoint['model_state_dict'])

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

net.to(device)

# train dataloader
scale = Rescale(int(1.5 * 230))
crop = RandomCrop(224)
rotate = RandomRotate(20.0)
norm = Normalize()
tupled = ToTupleTensor()

composed_for_tracking = transforms.Compose([
    Rescale(224),
    norm,
    tupled
])

dataset = SISSDataset(num_slices = 153,
                           num_scans= 4,
                           root_dir = Path.cwd().parents[0],
                           transform = composed_for_tracking,
                           train = True
                          )

dataloader = torch.utils.data.DataLoader(dataset, batch_size=10,
                                             shuffle=True, num_workers=4)

device = torch.device('cuda:0')
times = 0

# pdb.set_trace()
for mini_batch, (inputs, labels) in enumerate(dataloader):
    inputs = inputs.to(device)
    labels = labels.to(device)

    # forward
    # track history if only in train
    with torch.set_grad_enabled(False):
        outputs = net(inputs)

        _, preds = torch.max(outputs, 1)

    # inputs, labels = inputs.cpu()[0], labels.cpu()[0]

    # pdb.set_trace()
    i = 0
    for (input, label) in zip(inputs, labels):
        dice = 1 - dice_loss(input = preds[i], target = label).cpu().item()
        print('Step %d:  Dice Loss: %f' % (mini_batch, dice))
        if dice > 0.7 and dice!=1:
            print('Saving sample!!')
            viz_prediction((input.cpu(), label.cpu()), preds[i].cpu(), epoch=mini_batch*10 + i, dice = dice)
        i+=1

    times+=1
    if times ==10:
        break
