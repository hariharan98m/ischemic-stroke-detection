import torch
from pathlib import Path
import copy
import time
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pdb
from skimage import transform, io

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def viz_prediction(track_sample, pred, epoch):
    scans, label = track_sample

    scans, label = scans.numpy().transpose((1, 2, 0)), label.numpy()[..., np.newaxis]
    pred = pred[0].numpy()[..., np.newaxis]

    scans_stack = np.concatenate([scans, label, pred], axis=-1)

    fig = plt.figure(figsize=(20, 6))

    fig.suptitle('TRACKING Sample')

    for slice_, scan in enumerate(['dwi', 'flair', 't1', 't2', 'label', 'predicted']):
        ax = plt.subplot(1, 6, slice_ + 1)
        show_single_img(scans_stack[:, :, slice_], (scan == 'label' or scan == 'predicted'))
        plt.tight_layout()
        ax.set_title(scan)
        ax.axis('off')

    # plt.show()
    plt.savefig('sample_tracking/'+ str(epoch)+ '.jpg')


def show_single_img(image, label):
    """Show image"""
    cmap = 'gray'
    if label:
        cmap = 'binary'
    plt.imshow(image, cmap = cmap)


def dice_loss(input, target):
    smooth = 1.

    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()

    return 1 - ((2. * intersection + smooth) /
                (iflat.sum() + tflat.sum() + smooth))

def train_model(model, optimizer, scheduler, dataloaders, dataset_sizes, class_weights, track_sample, batch_size = 32, num_epochs=25, roi_align = False):
    since = time.time()
    PATH = 'save_objs.pth'
    epo = 1

    class_weights = class_weights.to(device)
    if Path(PATH).is_file():
        checkpoint = torch.load(PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epo = checkpoint['epoch']
        loss = checkpoint['loss']
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        print('Resuming from epoch ' + str(epo)+ ', LOSS: ', loss.item())

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    logs_ptr = open('logs', 'a')

    # pdb.set_trace()
    for epoch in range(epo, epo+num_epochs):
        epoch_str = 'Epoch {}/{}'.format(epoch, epo + num_epochs)
        print(epoch_str + '\n\n')
        logs_ptr.write(epoch_str)

        print('-' * 10)


        # pdb.set_trace()
        try:
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0
                running_dice = 0.0

                # Iterate over data.
                # times = 0
                for mini_batch, (inputs, labels, multi_scales) in enumerate(dataloaders[phase]):
                    h, w = 224, 224

                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    multi_scale_labels = multi_scales.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)

                        output_labels = outputs['mask']

                        multi_scale_outputs = list(outputs.values())[:-1]

                        _, preds = torch.max(output_labels, 1)

                        multi_scale_loss = 0
                        for scale in range(len(multi_scale_labels)):
                            multi_scale_loss+= F.nll(multi_scale_outputs[scale], multi_scale_labels[scale], weight=class_weights)

                        loss = F.nll_loss(output_labels, labels, weight=class_weights)

                        # loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    step_loss = loss
                    step_corrects = torch.sum(preds == labels.data)

                    step_acc = step_corrects.double().item() / labels.data.view(-1).size(0)

                    dice = dice_loss(input= labels, target=preds)

                    if phase == 'train':
                        step_str = '{} Step: {} Loss: {:.4f} Dice Loss: {:.4f} Acc: {:.4f} %'.format(
                            phase, mini_batch+1, step_loss, dice, step_acc * 100.0)
                        print(step_str + '\n')

                        logs_ptr.write(step_str)

                    # statistics
                    running_loss += step_loss.item() * inputs.size(0)
                    running_corrects += step_corrects  # done for batch size inputs.
                    running_dice+= dice * inputs.size(0)

                    # times+=1
                    #
                    # if times ==2:
                    #     break

                # pdb.set_trace()
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / (dataset_sizes[phase] * 224 * 224)
                epoch_dice = running_dice.double() / dataset_sizes[phase]

                if phase == 'train':
                    track_scans, track_labels = track_sample
                    track_scans, track_labels = track_scans.unsqueeze(0).to(device), track_labels.unsqueeze(0).to(device)

                    # pdb.set_trace()
                    pred = torch.argmax(torch.exp(model(track_scans)), dim=1)

                    viz_prediction(track_sample, pred.cpu(), epoch)
                    scheduler.step()

                loss_str= '\n{} Loss: {:.4f} Dice Loss: {:.4f} Acc: {:.4f} %\n'.format(
                    phase, epoch_loss, epoch_dice, epoch_acc * 100.0)
                print(loss_str)

                logs_ptr.write(loss_str+ '\n')

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

        except:
            # save model
            save_model(epoch, best_model_wts, optimizer, scheduler, loss, PATH)
            exit(0)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # save model
    save_model(num_epochs,
               best_model_wts,
               optimizer,
               scheduler, loss, PATH)


def save_model(epoch, best_model_wts, optimizer, scheduler, loss, PATH):
    torch.save({
        'epoch': epoch,
        'model_state_dict': best_model_wts,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
    }, PATH)


# borrow functions and modify it from https://github.com/Kaixhin/FCN-semantic-segmentation/blob/master/main.py
# Calculates class intersections over unions
def iou(pred, target):
    ious = []
    n_class = 2
    for cls in range(n_class):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = pred_inds[target_inds].sum()
        union = pred_inds.sum() + target_inds.sum() - intersection
        if union == 0:
            ious.append(float('nan'))  # if there is no ground truth, do not include in evaluation
        else:
            ious.append(float(intersection) / max(union, 1))
        # print("cls", cls, pred_inds.sum(), target_inds.sum(), intersection, float(intersection) / max(union, 1))
    return ious


def pixel_acc(pred, target):
    correct = (pred == target).sum()
    total   = (target == target).sum()
    return correct / total

