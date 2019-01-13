from __future__ import print_function, division
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, models, datasets
from pathlib import Path
import nibabel as nib
from torch.optim import lr_scheduler
import math
from skimage import measure
from utils import resize
import skimage
from skimage.transform import resize as sk_resize
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
from torchvision.ops import RoIAlign

plt.ion()   # interactive mode

# ROOT_DIR = Path('/Users/hariharan/PycharmProjects/SISS/')

ROOT_DIR = Path.cwd().parent.parent.parent

def read_nib(path):
    'read from the path'
    img = nib.load(str(path))
    data = img.get_fdata()
    return np.array(data)


def read_single_scan(scan_slice, train = True):
    'read the first 4 CT scan slices and 1 mask'
    scan_idx, slice_idx = scan_slice
    scan_dir = ROOT_DIR / 'data' / ('train' if train else 'val') / str(scan_idx)
    scan_data = []
    paths = sorted([x for x in scan_dir.iterdir()])
    for path in paths:
        if path.is_dir():
            scan_type = path / (path.name + '.nii')
            slice = read_nib(scan_type)[:, :, slice_idx]
            scan_data.append(slice)

    return np.stack(scan_data, axis=-1)

def test_scan(scan_idx, train = True):
    scan_dir = ROOT_DIR / 'data' / ('train' if train else 'val') / str(scan_idx)
    paths = sorted([x for x in scan_dir.iterdir()])
    stack = read_nib(paths[0] / (paths[0].name + '.nii'))
    return stack.shape[-1]

class SISSDataset(Dataset):
    """SISS dataset."""

    def __init__(self, num_slices, num_scans, root_dir, train = True, transform=None):
        """
        Args:
            num_slices (int): 154 for number of slices
            num_scans (int): 3 scans available
            root_dir (string): Directory with all the NIB scan blobs.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.train = train
        self.num_slices = num_slices
        self.num_scans = num_scans
        self.total_samples = num_slices * num_scans
        self.sample_to_path = lambda x: (math.floor(x / num_slices) + 1, x % num_slices)
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['background', 'lesion']

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()

        sample = read_single_scan(self.sample_to_path(idx), self.train)

        if self.transform:
            sample = self.transform(sample)

        return sample


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        # Decouple sample into id and image stack components
        # idx, sample = sample

        h, w = sample.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(sample, (new_h, new_w))

        return img


class RandomRotate(object):
    """Rotate randomly the image in a sample."""
    def __init__(self, max_deg):
        self.max_deg = max_deg

    def __call__(self, sample):
        # Decouple sample into id and image stack components
        # idx, sample = sample

        # generate a random angle between 0 to 20 degrees
        angle = np.random.uniform(0, 1) * self.max_deg

        # apply rotation
        sample = transform.rotate(sample, angle)

        return sample


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        # Decouple sample into id and image stack components
        # idx, sample = sample

        h, w = sample.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        sample = sample[top: top + new_h,
                 left: left + new_w, :]

        return sample


class Normalize(object):
    """
    Normalise every slice of the image stack by apply 0-1 min max normalization.

    """
    def __init__(self):
        pass

    def normalizeSlice(self, image, newMin=0, newMax=1):
        '''
        Function to normalize a image using the min-max normalization

        Arguments:
            image {np.array} -- a numpy array of information

        Keyword Arguments:
            newMin {int} -- the new min value for normalization (default: {0})
            newMax {int} -- the new max value for normalization (default: {1})

        Returns:
            normalizedImage -- the normalized image as numpy array
        '''

        oldMin = np.nanmin(image)
        oldMax = np.nanmax(image)
        # Special case
        if oldMin == oldMax:
            # Case which we can make it as zero
            if oldMax <= 0:
                return np.zeros(shape=image.shape, dtype=np.float32)
            # Lost hope case
            print("[+] oldMin: %f, oldMax: %f" % (oldMin, oldMax))
            return image
        normalizedImage = ((image - oldMin) / (oldMax - oldMin)) * (newMax - newMin) + newMin

        return normalizedImage


    def __call__(self, sample):
        mean = [0.485, 0.456, 0.406, 0.449]
        std = [0.229, 0.224, 0.225, 0.226]

        for i in range(4):
            sample[:, :, i] = (self.normalizeSlice(sample[:, :, i]) - mean[i])/std[i]

        return sample


class ToTupleTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # Decouple sample into id and image stack components
        # idx, sample = sample

        # get the scans and the mask label.
        scans, label = sample[:, :, :-1], sample[:, :, -1]

        # get one hot representation for mask
        # label_onehot = (np.arange(2) == label[..., None]).astype(np.int64)

        #make datatypes consistent
        scans, label = scans.astype(np.float32), label.astype(np.int64)

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        scans = scans.transpose((2, 0, 1))

        scans, label = torch.from_numpy(scans), torch.from_numpy(label)

        return scans, label


class ToMultiFloatMaskValues(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # Decouple sample into id and image stack components
        # idx, sample = sample

        # get the scans and the mask label.
        scans, label224 = sample[:, :, :-1], sample[:, :, -1]

        # get one hot representation for mask
        # label_onehot = (np.arange(2) == label[..., None]).astype(np.int64)

        #make datatypes consistent
        label28 = sk_resize(label224, (28, 28)).astype(np.float32)

        scans, label28, label224 = scans.astype(np.float32), label28, label224.astype(np.float32)

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        scans, label224, label28 = scans.transpose((2, 0, 1)), label224[np.newaxis,...], label28[np.newaxis,...]

        scans, label28, label224= torch.from_numpy(scans), torch.from_numpy(label28), torch.from_numpy(label224)

        return scans, label224, label28



class FinalRoIAlignExperiment(object):

    def __init__(self):
        self.valid_anchor_boxes = self.get_valid_anchor_boxes().astype(np.float32)
        # extracting RoI of size 56x56 on 224x224
        self.scale_base_roi_align = RoIAlign((56, 56), spatial_scale=1.0, sampling_ratio=2)  # base map is of size 56x56
        self.scale_1_roi_align = RoIAlign((28, 28), spatial_scale=1.0, sampling_ratio=2)     # wil look as 28x28 on the 112x112 image
        self.scale_2_roi_align = RoIAlign((14, 14), spatial_scale=1.0, sampling_ratio=2)     # will look at 14x14 on the 56x56 image
        self.scale_3_roi_align = RoIAlign((7, 7), spatial_scale=1.0, sampling_ratio=2)      # will look at 7x7 on the 28x28 image

    def get_valid_anchor_boxes(self):

        fe_size = 224 // 8  # (224 to 28) is 224/8 = 28
        ctr = np.zeros((fe_size * fe_size, 2))
        sub_sample_ratio = 224 // 28  # it is the height and width stride of the anchor centres.
        # we want to visit each and every point of the feature map and create a set of anchors.
        # for our case, 28x28 points on the original map will be the anchor centres.

        # Aspect Ratio of an anchor box is basically width/height. aspect ratio will always be 1 (square box)
        # Scales are bigger as the anchor box is from the base box (i.e. 512 x 512 box is twice as big as 256 x 256).
        # for scale, its better to go at 8 times.

        import math
        ar = 1.0
        scale = 56 / sub_sample_ratio  # to ensure that we get 25 on 112, 12.5 on 56 and 6.25 on 28

        # every 1x1 pixel on the 28x28 map corresponds to 8x8 on the original image. need to get the center of every 8x8 region.
        width_b = scale * math.sqrt(ar) * sub_sample_ratio
        height_b = scale * sub_sample_ratio / math.sqrt(ar)

        # Generate all the center points for all the boxes.
        ctr = np.zeros((fe_size * fe_size, 2))
        ctr_x = np.arange(sub_sample_ratio, (fe_size + 1) * sub_sample_ratio, sub_sample_ratio)
        ctr_y = ctr_x.copy()

        index = 0
        for x in range(len(ctr_x)):
            for y in range(len(ctr_y)):
                ctr[index, 0] = ctr_x[x] - sub_sample_ratio / 2
                ctr[index, 1] = ctr_y[y] - sub_sample_ratio / 2
                index += 1

        anchors = np.zeros((fe_size * fe_size, 4))
        anchors[:, 0] = ctr[:, 0] - height_b / 2.
        anchors[:, 1] = ctr[:, 1] - width_b / 2.
        anchors[:, 2] = ctr[:, 0] + height_b / 2.
        anchors[:, 3] = ctr[:, 1] + width_b / 2.
        # %%
        valid_anchor_boxes_indices = np.where(
            (anchors[:, 0] >= 0) &
            (anchors[:, 1] >= 0) &
            (anchors[:, 2] <= 224) &
            (anchors[:, 3] <= 224)
        )[0]
        valid_anchor_boxes = anchors[valid_anchor_boxes_indices]

        return valid_anchor_boxes


    def get_max_ious_boxes_labels(self, scans, label224):
        max_boxes = 16
        mask = label224

        # If there is some lesion on the mask, that is, if
        if len(np.unique(mask)) != 1:
            masked_labels = skimage.measure.label(mask)

            # instances are encoded as different colors
            obj_ids = np.unique(masked_labels)
            # first id is the background, so remove it
            obj_ids = obj_ids[1:]

            # split the color-encoded mask into a set
            # of binary masks
            masks = masked_labels == obj_ids[:, None, None]

            # get bounding box coordinates for each mask
            num_objs = len(obj_ids)

            # num objs
            print(obj_ids, num_objs)

            boxes = []
            for i in range(num_objs):
                pos = np.where(masks[i])
                xmin = np.min(pos[0])
                xmax = np.max(pos[0])
                ymin = np.min(pos[1])
                ymax = np.max(pos[1])
                boxes.append([xmin, ymin, xmax, ymax])

            # only choose the top 10 boxes from this.
            ious = np.empty((len(self.valid_anchor_boxes), len(boxes)), dtype=np.float32)
            ious.fill(0)
            for num1, i in enumerate(self.valid_anchor_boxes):
                xa1, ya1, xa2, ya2 = i
                anchor_area = (ya2 - ya1) * (xa2 - xa1)
                for num2, j in enumerate(boxes):
                    xb1, yb1, xb2, yb2 = j
                    box_area = (yb2 - yb1) * (xb2 - xb1)
                    inter_x1 = max([xb1, xa1])
                    inter_y1 = max([yb1, ya1])
                    inter_x2 = min([xb2, xa2])
                    inter_y2 = min([yb2, ya2])
                    if (inter_x1 < inter_x2) and (inter_y1 < inter_y2):
                        iter_area = (inter_y2 - inter_y1 + 1) * \
                                    (inter_x2 - inter_x1 + 1)
                        iou = iter_area / \
                              (anchor_area + box_area - iter_area)
                    else:
                        iou = 0.

                    ious[num1, num2] = iou

            # choose the highest valued bounding boxes

            patches_for_objs = max_boxes // num_objs
            maxarg_ious = np.argsort(ious, axis=0)[::-1]

            selected_ious_args = []
            for obj in range(num_objs):
                obj_max_indices = maxarg_ious[:patches_for_objs, obj].tolist()
                maxarg_ious = np.delete(maxarg_ious, obj_max_indices, axis=0)
                selected_ious_args.extend(obj_max_indices)

            # Return, the selected anchor boxes coords and the class_labels
            sel_anchors = self.valid_anchor_boxes[selected_ious_args]
            # and the all ones class labels
            class_labels = [1.0] * max_boxes

            return sel_anchors, class_labels

        # so there's no lesion at all in any part of the mask
        else:
            # box_for_scan_area
            cornerVal = scans[0, 0, 0]
            pos = np.where(scans[0, :, :] != cornerVal)
            if len(pos[0]):
                x1_scan = np.min(pos[0])
                x2_scan = np.max(pos[0])
                y1_scan = np.min(pos[1])
                y2_scan = np.max(pos[1])
            else:
                x1_scan, y1_scan, x2_scan, y2_scan = [0, 0, 223, 223]

            # filter valid bounding boxes
            valid_anchor_boxes_indices = np.where(
                (self.valid_anchor_boxes[:, 0] >= x1_scan) &
                (self.valid_anchor_boxes[:, 1] >= y1_scan) &
                (self.valid_anchor_boxes[:, 2] <= x2_scan) &
                (self.valid_anchor_boxes[:, 3] <= y2_scan)
            )[0]

            sel_anchors = self.valid_anchor_boxes[np.random.choice(valid_anchor_boxes_indices, max_boxes)]
            class_labels = [0.0] * max_boxes

            return sel_anchors, class_labels

    def __call__(self, sample):

        # scans and the labelled mask are in (4, 224, 224) and (1,224,224)
        scans, label224 = sample[:, :, :-1], sample[:, :, -1]

        # consistent datatypes
        scans, label224 = scans.astype(np.float32), label224.astype(np.float32)

        # get the data objs into an object of shape (4, h, w) and (1, h, w)
        scans, label224 = scans.transpose((2, 0, 1)), label224[np.newaxis,...]

        # get 10 anchor boxes formatted.
        anchor_boxes, class_labels = self.get_max_ious_boxes_labels(scans, label224)

        scans, label224 = torch.from_numpy(scans), torch.from_numpy(label224)

        # image of size 224.
        #every anchor box has a size of 56x56
        cut_boxes = torch.from_numpy(anchor_boxes)
        base, scale1, scale2, scale3 = self.scale_base_roi_align(label224.unsqueeze(dim=0), [cut_boxes]), \
                                       self.scale_1_roi_align(label224.unsqueeze(0), [cut_boxes]), \
                                       self.scale_2_roi_align(label224.unsqueeze(0), [cut_boxes]), \
                                       self.scale_3_roi_align(label224.unsqueeze(0), [cut_boxes])
        # will return boxes of shape (batch_size, 1, h, w)

        # items to return from here, are the anchor boxes for all the cuts, the class_labels, the image scan 224,
        # the anchor cut labels of 56(base), 28 for 112, 14 for 56x56, 7 for 28x28

        return cut_boxes, class_labels, scans, (base, scale1, scale2, scale3)



class ToClassifierTuple(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # Decouple sample into id and image stack components
        # idx, sample = sample

        # get the scans and the mask label.
        scans, label224 = sample[:, :, :-1], sample[:, :, -1]

        # get one hot representation for mask
        # label_onehot = (np.arange(2) == label[..., None]).astype(np.int64)

        scans, label224 = scans.astype(np.float32), label224.astype(np.float32)

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        scans, label224 = scans.transpose((2, 0, 1)), label224[np.newaxis,...]

        class_ = np.array([1.0], dtype=np.float32) if np.sum(label224 == 1) else np.array([0.0], dtype=np.float32)

        scans, label224, class_= torch.from_numpy(scans), torch.from_numpy(label224), torch.from_numpy(class_)

        return scans, label224, class_
