import torch
from pathlib import Path
import copy
import time
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import pdb
import skimage
from distutils.version import LooseVersion
from skimage.transform import resize as sk_resize
from utils import dice_loss
from utils import expand_mask, resize, show_single_img, get_prob_map28, dice_loss, actual_predicted

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from torchvision.ops import RoIAlign
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

class_weights = torch.tensor([0.1, 0.9]).to(device)


def attention(model, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=25, viz = False):
    #model to CUDA
    model = model.to(device)

    #out dirs
    base_dir = Path.cwd() / 'outputs' / 'experiment1'
    output_tracking_dir = base_dir / 'output_tracking'
    logs_dir = base_dir / 'logs'
    model_dir = base_dir / 'model'

    logs_dir.mkdir(parents= True, exist_ok= True)
    model_dir.mkdir(parents= True, exist_ok=True)
    output_tracking_dir.mkdir(parents=True, exist_ok=True)

    since = time.time()
    PATH = str(model_dir / (model.name+'.pth'))
    epo = 1

    if Path(PATH).is_file():
        checkpoint = torch.load(PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epo = checkpoint['epoch']
        loss = checkpoint['loss']
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        print('Resuming from epoch ' + str(epo) + ', LOSS: ', loss)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 3.0

    logs_ptr = open(str(logs_dir/ 'train_logs'), 'a')

    # pdb.set_trace()
    for epoch in range(epo, epo + num_epochs):
        epoch_str = 'Epoch {}/{}'.format(epoch, epo + num_epochs - 1) + '\n\n'
        print(epoch_str)
        logs_ptr.write(epoch_str)

        print('-' * 10)

        try:
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode

                running_reg = 0.0
                running_dice = 0.0

                # Iterate over data.
                times = 0

                for mini_batch, (inputs, label224, label28) in enumerate(dataloaders[phase]):

                    inputs = inputs.to(device)

                    # labels size is (batch_size, 1, 224, 224)
                    label28 = label28.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):

                        log_softmax_outputs28 = model(inputs)['seg_28']  # shape of pred28 is (batch_size, 2, 28, 28)

                        softmax_outputs28 = torch.exp(log_softmax_outputs28)
                        output28_prob = get_prob_map28(softmax_outputs28)

                        reg_loss = torch.mean(
                            torch.sum(-torch.log(1.0 - torch.abs(output28_prob - label28)), dim=[1, 2, 3])
                        )/1000.0

                        dice_l = dice_loss(input=torch.round(output28_prob), target=torch.round(label28))

                        total_loss = reg_loss + 0.5*dice_l

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            total_loss.backward()
                            optimizer.step()

                    if phase == 'train':
                        step_str = '{} Step {}- Loss: {:.4f}, Dice Loss: {:.4f}, Reg Loss: {:.4f}'\
                            .format(phase, mini_batch + 1,total_loss, dice_l, reg_loss)
                        print(step_str)

                        logs_ptr.write(step_str+'\n')

                    if phase == 'val' and viz:
                        output28_prob = output28_prob.cpu()
                        label28 = label28.cpu()

                        for item in range(label28.size(0)):
                            expanded_output28_prob = expand_mask([[0, 0, 224, 224]],
                                                                 output28_prob[item].detach().numpy(),
                                                                 (224, 224))
                            expanded_label28 = expand_mask([[0, 0, 224, 224]], label28[item].detach().numpy(),
                                                           (224, 224))

                            epoch_tracking_path = output_tracking_dir / str(epoch)
                            if not epoch_tracking_path.is_dir():
                                epoch_tracking_path.mkdir(parents=True, exist_ok=False)

                            actual_predicted(expanded_label28[0], expanded_output28_prob[0],
                                             str(epoch_tracking_path / (str(mini_batch*label28.size(0) +item) + '.jpg') ) )

                    # statistics
                    # running_loss += step_loss.item() * inputs.size(0)
                    running_dice += dice_l.item() * inputs.size(0)
                    running_reg += reg_loss.item() * inputs.size(0)

                    # times+=1
                    # if times==2:
                    #     break

                # end of an epoch
                # pdb.set_trace()

                # epoch_loss = running_loss / dataset_sizes[phase]
                epoch_dice_l = running_dice / dataset_sizes[phase]
                epoch_reg_loss = running_reg / dataset_sizes[phase]
                epoch_loss = epoch_dice_l + epoch_reg_loss

                if phase == 'train':
                    scheduler.step()

                loss_str = '\n{} Epoch {}: TotalLoss: {:.4f}   RegLoss: {:.4f} Dice Loss: {:.4f} \n'.format(
                    phase, epoch, epoch_loss, epoch_reg_loss, epoch_dice_l) + '\n'
                print(loss_str)

                logs_ptr.write(loss_str + '\n')

                # deep copy the model
                if phase == 'val' and epoch_loss >= best_loss:
                    print('Val Dice better than Best Dice')
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())

        except:
            # save model
            save_model(epoch, best_model_wts, optimizer, scheduler, epoch_loss, PATH)
            exit(0)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val DICE: {:4f}'.format(best_loss))

    # save model
    save_model(num_epochs,
               best_model_wts,
               optimizer,
               scheduler, epoch_loss, PATH)






# Second Experiment

def attention_field_stripping(model, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=25, viz= False):
    #out dirs
    base_dir = Path.cwd() / 'outputs' / 'experiment2'
    output_tracking_dir = base_dir / 'output_tracking'
    logs_dir = base_dir / 'logs'
    model_dir = base_dir / 'model'

    model = model.to(device)

    upsampler = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

    logs_dir.mkdir(parents= True, exist_ok= True)
    model_dir.mkdir(parents= True, exist_ok=True)
    output_tracking_dir.mkdir(parents=True, exist_ok=True)

    since = time.time()
    PATH = str(model_dir / (model.name+'.pth'))
    epo = 1

    if Path(PATH).is_file():
        checkpoint = torch.load(PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epo = checkpoint['epoch']
        loss = checkpoint['loss']
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        print('Resuming from epoch ' + str(epo) + ', LOSS: ', loss)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 3.0

    logs_ptr = open(str(logs_dir/ 'train_logs'), 'a')

    # pdb.set_trace()
    for epoch in range(epo, epo + num_epochs):
        epoch_str = 'Epoch {}/{}'.format(epoch, epo + num_epochs - 1) + '\n\n'
        print(epoch_str)
        logs_ptr.write(epoch_str)

        print('-' * 10)

        try:
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode

                running_softmax = 0.0
                running_dice = 0.0

                # Iterate over data.
                times = 0

                for mini_batch, (inputs, label224, label28) in enumerate(dataloaders[phase]):

                    inputs = inputs.to(device)

                    # labels size is (batch_size, 1, 224, 224)
                    label28 = label28.to(device)

                    label224 = label224.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):

                        log_softmax_outputs28 = model(inputs)['seg_28']  # shape of pred28 is (batch_size, 2, 28, 28)

                        softmax_loss = F.nll_loss(log_softmax_outputs28, label28.round().squeeze().long(),
                                                  weight=class_weights)

                        softmax_outputs28 = torch.exp(log_softmax_outputs28)
                        torch_pred28_prob = get_prob_map28(softmax_outputs28)
                        torch_pred224_prob = upsampler(torch_pred28_prob)

                        rounded_pred224_prob_for_dice = torch.round(torch_pred224_prob)
                        # return format is (batch_size, 1, 224, 224)

                        dice_l = dice_loss(input=rounded_pred224_prob_for_dice, target=label224)

                        # dice_l = dice_loss(input=outputs28, target=mask28)

                        total_loss = 0.7 * dice_l + 0.3 * softmax_loss

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            total_loss.backward()
                            optimizer.step()

                    if phase == 'train':
                        step_str = '{} Step {}- Loss: {:.4f}, Dice Loss: {:.4f}, Softmax Loss: {:.4f}'\
                              .format(phase, mini_batch + 1,total_loss, dice_l, softmax_loss)

                        print(step_str)

                        logs_ptr.write(step_str+'\n')

                    if phase == 'val' and viz:
                        for item in range(label28.size(0)):
                            # get the path for saving the intermediate outputs
                            epoch_tracking_path = output_tracking_dir / str(epoch)

                            if not epoch_tracking_path.is_dir():
                                epoch_tracking_path.mkdir(parents=True, exist_ok=False)

                            actual_predicted(label224[item][0].numpy(),
                                             rounded_pred224_prob_for_dice[item][0].detach().numpy(),
                                             str(epoch_tracking_path / (str(mini_batch * label28.size(0) + item) + '.jpg') )
                                             )

                    # statistics
                    # running_loss += step_loss.item() * inputs.size(0)
                    running_dice += dice_l.item() * inputs.size(0)
                    running_softmax += softmax_loss.item() * inputs.size(0)

                    # times+=1
                    # if times==2:
                    #     break

                # end of an epoch
                # pdb.set_trace()

                # epoch_loss = running_loss / dataset_sizes[phase]
                epoch_dice_l = running_dice / dataset_sizes[phase]
                epoch_softmax = running_softmax / dataset_sizes[phase]
                epoch_loss = epoch_dice_l + epoch_softmax

                if phase == 'train':
                    scheduler.step()

                loss_str = '\n{} Epoch {}: TotalLoss: {:.4f}   SoftmaxLoss: {:.4f} Dice Loss: {:.4f} \n'.format(
                    phase, epoch, epoch_loss, epoch_softmax, epoch_dice_l) + '\n'
                print(loss_str)

                logs_ptr.write(loss_str + '\n')

                # deep copy the model
                if phase == 'val' and epoch_loss > best_loss:
                    print('Val Dice better than Best Dice')
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())

        except:
            # save model
            save_model(epoch, best_model_wts, optimizer, scheduler, epoch_loss, PATH)
            exit(0)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val DICE: {:4f}'.format(best_loss))

    # save model
    save_model(num_epochs,
               best_model_wts,
               optimizer,
               scheduler, loss, PATH)




# Third Experiment

def single_scale_roi_align_experiment(model, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=25, viz = False):
    #out dirs
    base_dir = Path.cwd() / 'outputs' / 'single_scale_roi_align_experiment'
    output_tracking_dir = base_dir / 'output_tracking'
    logs_dir = base_dir / 'logs'
    model_dir = base_dir / 'model'

    upsampler = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

    model = model.to(device)

    logs_dir.mkdir(parents= True, exist_ok= True)
    model_dir.mkdir(parents= True, exist_ok=True)
    output_tracking_dir.mkdir(parents=True, exist_ok=True)

    since = time.time()
    PATH = str(model_dir / (model.name+'.pth'))
    epo = 1

    if Path(PATH).is_file():
        checkpoint = torch.load(PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epo = checkpoint['epoch']
        loss = checkpoint['loss']
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        print('Resuming from epoch ' + str(epo) + ', LOSS: ', loss)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 3.0

    logs_ptr = open(str(logs_dir/ 'train_logs'), 'a')

    # pdb.set_trace()
    for epoch in range(epo, epo + num_epochs):
        epoch_str = 'Epoch {}/{}'.format(epoch, epo + num_epochs - 1) + '\n\n'
        print(epoch_str)
        logs_ptr.write(epoch_str)

        print('-' * 10)

        try:
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode

                running_softmax = 0.0
                running_dice = 0.0

                # Iterate over data.
                times = 0

                for mini_batch, (patch_boxes, scans, labels) in enumerate(dataloaders[phase]):

                    inputs = scans.to(device)

                    # labels size is (batch_size, 1, 224, 224)
                    label224 = labels.to(device)

                    # patch boxes
                    patch_boxes = patch_boxes.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):

                        log_softmax_outputs224 = model((inputs, patch_boxes))  # shape of pred224 is (batch_size, 2, 224, 224)

                        softmax_loss = F.nll_loss(log_softmax_outputs224, label224.squeeze().long(),
                                                  weight=class_weights)

                        softmax_outputs224 = torch.exp(log_softmax_outputs224)

                        _, pred224_argmax = torch.max(softmax_outputs224, dim=1, keepdim=True)  # (batch_size, 1, 28,28)
                        pred224_argmax = pred224_argmax.float()

                        dice_l = dice_loss(input=pred224_argmax, target=label224)

                        # dice_l = dice_loss(input=outputs28, target=mask28)

                        total_loss = 0.9 * dice_l + 0.1 * softmax_loss

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            total_loss.backward()
                            optimizer.step()

                    if phase == 'train':
                        step_str = '{} Step {}- Loss: {:.4f}, Dice Loss: {:.4f}, Softmax Loss: {:.4f}'\
                              .format(phase, mini_batch + 1,total_loss, dice_l, softmax_loss)

                        print(step_str)

                        logs_ptr.write(step_str+'\n')

                    if phase == 'val' and viz:
                        for item in range(label224.size(0)):
                            # get the path for saving the intermediate outputs
                            epoch_tracking_path = output_tracking_dir / str(epoch)

                            if not epoch_tracking_path.is_dir():
                                epoch_tracking_path.mkdir(parents=True, exist_ok=False)

                            actual_predicted(label224[item][0].numpy(),
                                             pred224_argmax[item][0].detach().numpy(),
                                             str(epoch_tracking_path / (str(mini_batch * label224.size(0) + item) + '.jpg') )
                                             )

                    # statistics
                    # running_loss += step_loss.item() * inputs.size(0)
                    running_dice += dice_l.item() * inputs.size(0)
                    running_softmax += softmax_loss.item() * inputs.size(0)

                    # times+=1
                    # if times==2:
                    #     break

                # end of an epoch
                # pdb.set_trace()

                # epoch_loss = running_loss / dataset_sizes[phase]
                epoch_dice_l = running_dice / dataset_sizes[phase]
                epoch_softmax = running_softmax / dataset_sizes[phase]
                epoch_loss = epoch_dice_l + epoch_softmax

                if phase == 'train':
                    scheduler.step()

                loss_str = '\n{} Epoch {}: TotalLoss: {:.4f}   SoftmaxLoss: {:.4f} Dice Loss: {:.4f} \n'.format(
                    phase, epoch, epoch_loss, epoch_softmax, epoch_dice_l) + '\n'
                print(loss_str)

                logs_ptr.write(loss_str + '\n')

                # deep copy the model
                if phase == 'val' and epoch_loss > best_loss:
                    print('Val Dice better than Best Dice')
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())

        except:
            # save model
            save_model(epoch, best_model_wts, optimizer, scheduler, epoch_loss, PATH)
            exit(0)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val DICE: {:4f}'.format(best_loss))

    # save model
    save_model(num_epochs,
               best_model_wts,
               optimizer,
               scheduler, epoch_loss, PATH)





# Experiment with Classifier Head

def classifier_experiment(model, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=25, viz = False):
    #out dirs
    base_dir = Path.cwd() / 'outputs' / 'classifier_experiment'
    output_tracking_dir = base_dir / 'output_tracking'
    logs_dir = base_dir / 'logs'
    model_dir = base_dir / 'model'

    model = model.to(device)

    logs_dir.mkdir(parents= True, exist_ok= True)
    model_dir.mkdir(parents= True, exist_ok=True)
    output_tracking_dir.mkdir(parents=True, exist_ok=True)

    since = time.time()
    PATH = str(model_dir / (model.name+'.pth'))
    epo = 1

    if Path(PATH).is_file():
        checkpoint = torch.load(PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epo = checkpoint['epoch']
        loss = checkpoint['loss']
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        print('Resuming from epoch ' + str(epo) + ', LOSS: ', loss)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 3.0
    best_acc = 0
    logs_ptr = open(str(logs_dir/ 'train_logs'), 'a')
    epoch_softmax = 0

    # pdb.set_trace()

    for epoch in range(epo, epo + num_epochs):
        epoch_str = 'Epoch {}/{}'.format(epoch, epo + num_epochs - 1) + '\n\n'
        print(epoch_str)
        logs_ptr.write(epoch_str)


        print('-' * 10)

        try:
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode

                running_softmax = 0.0
                running_corrects = 0

                # Iterate over data.
                times = 0

                # pdb.set_trace()

                for mini_batch, (inputs, label224, cls) in enumerate(dataloaders[phase]):

                    # pdb.set_trace()

                    inputs = inputs.to(device)

                    # labels size is (batch_size, 1, 224, 224)
                    # label224 = label224.to(device)

                    cls = cls.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):

                        log_softmax_cls_output = model(inputs)['classifier']  # shape of pred output is (batch_size, 2, 1, 1)

                        softmax_loss = F.nll_loss(log_softmax_cls_output, cls.squeeze().long(),
                                                  weight=class_weights)

                        softmax_output = torch.exp(log_softmax_cls_output)

                        _, pred_argmax = torch.max(softmax_output, dim=1, keepdim=True)  # (batch_size, 1, 28,28)

                        pred_argmax = pred_argmax.float()

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            softmax_loss.backward()
                            optimizer.step()

                    step_corrects = torch.sum(pred_argmax == cls).double()
                    step_acc = step_corrects / cls.size(0)

                    if phase == 'train':
                        step_str = '{} Step {}- Loss: {:.4f}, Accuracy: {:.4f} %'\
                              .format(phase, mini_batch + 1, softmax_loss, step_acc)

                        print(step_str)

                        logs_ptr.write(step_str+'\n')

                    # statistics
                    running_softmax += softmax_loss.item() * inputs.size(0)
                    running_corrects += step_corrects

                    # times+=1
                    #
                    # if times ==2:
                    #     break


                # end of an epoch
                # pdb.set_trace()

                epoch_softmax = running_softmax / dataset_sizes[phase]
                epoch_acc = running_corrects / dataset_sizes[phase]

                if phase == 'train':
                    scheduler.step()

                loss_str = '\n{} Epoch {}: SoftmaxLoss: {:.4f}, Accuracy: {:.4f} %\n'.format(
                    phase, epoch, epoch_softmax, epoch_acc) + '\n'

                print(loss_str)

                logs_ptr.write(loss_str + '\n')

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    print('Val Dice better than Best Dice')
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

        except:
            # save model
            save_model(epoch, best_model_wts, optimizer, scheduler, epoch_softmax, PATH)
            exit(0)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val DICE: {:4f}'.format(best_loss))

    # save model
    save_model(num_epochs,
               best_model_wts,
               optimizer,
               scheduler, epoch_softmax, PATH)



# single scale RoI align

def single_scale_roi_align(model, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=25, viz= False):
    #out dirs
    base_dir = Path.cwd() / 'outputs' / 'single_scale_roi_align'
    output_tracking_dir = base_dir / 'output_tracking'
    logs_dir = base_dir / 'logs'
    model_dir = base_dir / 'model'

    model = model.to(device)

    upsampler = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

    logs_dir.mkdir(parents= True, exist_ok= True)
    model_dir.mkdir(parents= True, exist_ok=True)
    output_tracking_dir.mkdir(parents=True, exist_ok=True)

    since = time.time()
    PATH = str(model_dir / (model.name+'.pth'))
    epo = 1

    if Path(PATH).is_file():
        checkpoint = torch.load(PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epo = checkpoint['epoch']
        loss = checkpoint['loss']
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        print('Resuming from epoch ' + str(epo) + ', LOSS: ', loss)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 3.0

    logs_ptr = open(str(logs_dir/ 'train_logs'), 'a')

    # pdb.set_trace()
    for epoch in range(epo, epo + num_epochs):
        epoch_str = 'Epoch {}/{}'.format(epoch, epo + num_epochs - 1) + '\n\n'
        print(epoch_str)
        logs_ptr.write(epoch_str)

        print('-' * 10)

        try:
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode

                running_softmax = 0.0
                running_dice = 0.0

                # Iterate over data.
                times = 0

                for mini_batch, (inputs, label224, label28) in enumerate(dataloaders[phase]):

                    inputs = inputs.to(device)

                    # labels size is (batch_size, 1, 224, 224)
                    label28 = label28.to(device)

                    label224 = label224.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):

                        log_softmax_outputs28 = model(inputs)  # shape of pred28 is (batch_size, 2, 28, 28)

                        softmax_loss = F.nll_loss(log_softmax_outputs28, label28.round().squeeze().long(),
                                                  weight=class_weights)

                        softmax_outputs28 = torch.exp(log_softmax_outputs28)
                        torch_pred28_prob = get_prob_map28(softmax_outputs28)
                        torch_pred224_prob = upsampler(torch_pred28_prob)

                        rounded_pred224_prob_for_dice = torch.round(torch_pred224_prob)
                        # return format is (batch_size, 1, 224, 224)

                        dice_l = dice_loss(input=rounded_pred224_prob_for_dice, target=label224)

                        # dice_l = dice_loss(input=outputs28, target=mask28)

                        total_loss = 0.7 * dice_l + 0.3 * softmax_loss

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            total_loss.backward()
                            optimizer.step()

                    if phase == 'train':
                        step_str = '{} Step {}- Loss: {:.4f}, Dice Loss: {:.4f}, Softmax Loss: {:.4f}'\
                              .format(phase, mini_batch + 1,total_loss, dice_l, softmax_loss)

                        print(step_str)

                        logs_ptr.write(step_str+'\n')

                    if phase == 'val' and viz:
                        for item in range(label28.size(0)):
                            # get the path for saving the intermediate outputs
                            epoch_tracking_path = output_tracking_dir / str(epoch)

                            if not epoch_tracking_path.is_dir():
                                epoch_tracking_path.mkdir(parents=True, exist_ok=False)

                            actual_predicted(label224[item][0].numpy(),
                                             rounded_pred224_prob_for_dice[item][0].detach().numpy(),
                                             str(epoch_tracking_path / (str(mini_batch * label28.size(0) + item) + '.jpg') )
                                             )

                    # statistics
                    # running_loss += step_loss.item() * inputs.size(0)
                    running_dice += dice_l.item() * inputs.size(0)
                    running_softmax += softmax_loss.item() * inputs.size(0)

                    # times+=1
                    # if times==2:
                    #     break

                # end of an epoch
                # pdb.set_trace()

                # epoch_loss = running_loss / dataset_sizes[phase]
                epoch_dice_l = running_dice / dataset_sizes[phase]
                epoch_softmax = running_softmax / dataset_sizes[phase]
                epoch_loss = epoch_dice_l + epoch_softmax

                if phase == 'train':
                    scheduler.step()

                loss_str = '\n{} Epoch {}: TotalLoss: {:.4f}   SoftmaxLoss: {:.4f} Dice Loss: {:.4f} \n'.format(
                    phase, epoch, epoch_loss, epoch_softmax, epoch_dice_l) + '\n'
                print(loss_str)

                logs_ptr.write(loss_str + '\n')

                # deep copy the model
                if phase == 'val' and epoch_loss > best_loss:
                    print('Val Dice better than Best Dice')
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())

        except:
            # save model
            save_model(epoch, best_model_wts, optimizer, scheduler, epoch_loss, PATH)
            exit(0)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val DICE: {:4f}'.format(best_loss))

    # save model
    save_model(num_epochs,
               best_model_wts,
               optimizer,
               scheduler, loss, PATH)






# Experiment with Final ENHANCED RoI align - Experiment 5

def final_roi_align(model, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=25, viz = False):
    # Steps
    '''
    First thing, write code to generate all the patches as 28x28 from the MRI scan. do it with anchor boxes. write a Transfrom for that.
        - for an scan generate 10 RoIs.
        - the generator will return this:- MRI_scan224, MRI_label224, plus (all anchor boxes of MRI_scan28x28, MRI_label28x28)
        - if the sample has some lesion, return all the RoIs with that lesion. return just the (x1, y1, x2, y2) of the boxes in 224x224 map.
        - if the sample has no lesion, then return 10 RoIs of no lesion zone.
    Then perform these new set of actions on the sub-level data transform:-
        - it has to take the 224x224 tensor, and the rois, and then do the roi align to generate these level of feature maps.
        - now, view(-1, m, n) and randomize all the samples, for all (m, n) maps levels.
        - it has to run a simple algorithm to get the class as 0 or 1 for every patch.

    Second, get RoI maps for the same 28x28 roi from the feature maps of the CNN using RoI align. and by passing through the deconv nets.
        - so, the model() nn.module has perform all this.
        - it has to run deconv nets as pytorch.nn modules for these levels of patches dims to result in uniform 28x28 maps.
        - concat all the 28x28 predicted masks from these feature levels, make one small 3x3 or 3x3 conv and 1x1 conv until it ends up here.
        - it has to return 28x28 predictions for all feature levels individually plus the max class voting result from these preds, as one mask
          plus the classification head

    Third, frame the loss function with the classifier head and the segmentor head.
        - train the classifier for all samples.
        - run a simple algorithm to collect only those samples with non-zero lesion based on the patch classifier label.
        - run piecewise loss for every patch mask to prediction.
          Also, double it up with a secondary, loss function.
    '''







def save_model(epoch, best_model_wts, optimizer, scheduler, loss, PATH):
    print('Saving model @ epoch = ', epoch)
    torch.save({
        'epoch': epoch,
        'model_state_dict': best_model_wts,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
    }, PATH)
